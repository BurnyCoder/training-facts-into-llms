"""Global context: isolate Qwen3.5 loading, chat rendering, and greedy generation.

Qwen3.5-0.8B is a full multimodal model even when this project supplies only
text. The full processor/model pairing is required by Transformers and TRL.
Sources:
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/chat_templating.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    """Keep the full VLM, processor, and device together across phases."""

    # The model becomes a PEFT wrapper during training.
    model: Any
    # The Qwen3VLProcessor owns the tokenizer and native chat template.
    processor: Any
    # The CUDA device is recorded once at load time.
    device: Any
    # Training attaches only JSON-safe public metrics for later reports.
    training_summary: dict[str, Any] | None = None


def render_generation_prompt(
    processor: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> str:
    """Render Qwen's native assistant prefix with the resolved thinking policy."""
    # The generation prompt marks where the assistant response must begin.
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def load_base_model(config: Any, logger: Any | None = None) -> ModelBundle:
    """Load the exact full Qwen checkpoint in the resolved training precision."""
    # Heavy imports stay inside the runtime boundary so pure unit tests remain fast.
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, set_seed

    # Fixed initialization and data seeds improve run repeatability.
    set_seed(config.seed)
    experiment = getattr(config, "experiment", None)
    scientific = getattr(experiment, "config", None)
    precision = getattr(getattr(scientific, "precision", None), "mode", "bfloat16")
    dtype_by_mode = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        model_dtype = dtype_by_mode[precision]
    except KeyError as error:
        raise ValueError(f"Unsupported training precision: {precision}") from error
    # The full processor is required even for text-only Qwen3.5 examples.
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        # The pinned base is public and runtime model code never needs credentials.
        token=False,
    )
    # Transformers 5 uses `dtype`; the older `torch_dtype` name is deprecated.
    model = AutoModelForMultimodalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        # Prevent a cached Hub login from being sent for this public checkpoint.
        token=False,
    )
    # The approved workflow requires the local NVIDIA GPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # BF16 runs require explicit device capability; other modes use CUDA directly.
    if model_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The CUDA device does not support BF16")
    # Use the first visible CUDA device for the single-GPU trainer.
    device = torch.device("cuda:0")
    # Moving once avoids Trainer device-map conflicts.
    model.to(device)
    # Baseline generation never enables dropout.
    model.eval()
    # Optional structured provenance contains no model outputs yet.
    if logger is not None:
        logger.event(
            "model_loaded",
            model_id=config.model_id,
            model_revision=config.model_revision,
            model_class=type(model).__name__,
            processor_class=type(processor).__name__,
            device=str(device),
            dtype=str(next(model.parameters()).dtype),
        )
    # Return one explicit model boundary.
    return ModelBundle(model=model, processor=processor, device=device)


def _text_config(model: Any) -> Any:
    """Return the underlying full model's text configuration."""
    # PEFT wrappers expose the base configuration through `config`.
    config = model.config
    # Transformers 5 provides a uniform accessor for composite models.
    return (
        config.get_text_config()
        if hasattr(config, "get_text_config")
        else config.text_config
    )


def generate_response(
    bundle: ModelBundle,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    generation: Any | None = None,
) -> tuple[str, str]:
    """Generate one resolved-policy answer and return its exact rendered prompt."""
    # Import torch only when actual model inference is requested.
    import torch

    # Render the exact native prompt for logging and reproducibility.
    rendered_prompt = render_generation_prompt(
        bundle.processor,
        messages,
        enable_thinking=bool(getattr(generation, "enable_thinking", False)),
    )
    # Tokenize the already rendered template without adding duplicate special tokens.
    inputs = bundle.processor(
        text=[rendered_prompt],
        return_tensors="pt",
        add_special_tokens=False,
    )
    # Move every tensor field to the model's CUDA device.
    inputs = {name: tensor.to(bundle.device) for name, tensor in inputs.items()}
    # Slice generated tokens after this immutable prompt length.
    input_length = inputs["input_ids"].shape[-1]
    # The tokenizer ends chat turns with `<|im_end|>`.
    tokenizer_eos = bundle.processor.tokenizer.eos_token_id
    # The pinned model text configuration declares an additional EOS ID.
    config_eos = _text_config(bundle.model).eos_token_id
    # Preserve both unique stopping IDs to avoid runaway direct answers.
    eos_ids = list(dict.fromkeys([tokenizer_eos, config_eos]))
    # Padding uses the tokenizer's configured pad ID, which is valid for Qwen.
    pad_token_id = bundle.processor.tokenizer.pad_token_id
    do_sample = bool(getattr(generation, "do_sample", False))
    generation_options: dict[str, Any] = {
        "do_sample": do_sample,
        "num_beams": int(getattr(generation, "num_beams", 1)),
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_ids,
        "pad_token_id": pad_token_id,
    }
    repetition_penalty = float(getattr(generation, "repetition_penalty", 1.0))
    if repetition_penalty != 1.0:
        generation_options["repetition_penalty"] = repetition_penalty
    if do_sample:
        generation_options.update(
            {
                "temperature": float(getattr(generation, "temperature", 1.0)),
                "top_p": float(getattr(generation, "top_p", 1.0)),
                "top_k": int(getattr(generation, "top_k", 50)),
            }
        )
    # Disable gradients so baseline and adapter runs share one decoding policy.
    bundle.model.eval()
    with torch.inference_mode():
        output_ids = bundle.model.generate(
            **inputs,
            **generation_options,
        )
    # Decode only newly generated tokens, never the input-plus-output sequence.
    answer_ids = output_ids[:, input_length:]
    # Preserve all generated text except inconsequential edge whitespace.
    output = bundle.processor.tokenizer.decode(
        answer_ids[0],
        skip_special_tokens=True,
    ).strip()
    # Return both public prompt evidence and the complete answer.
    return output, rendered_prompt


def load_adapter_model(
    config: Any,
    adapter: Any,
    logger: Any | None = None,
    *,
    adapter_log_reference: str | None = None,
    subfolder: str | None = None,
) -> ModelBundle:
    """Load a full Qwen base model and attach a saved non-trainable PEFT adapter."""
    # PeftModel preserves the full multimodal architecture; AutoPeftModelForCausalLM does not.
    from peft import PeftModel

    # Start without an owned bundle so failures before base return remain harmless.
    bundle = None
    try:
        # Load the exact pinned base through the same path used for evaluation.
        bundle = load_base_model(config, logger=logger)
        # Attach either a validated local directory or anonymous public Hub adapter.
        load_options: dict[str, Any] = {
            "is_trainable": False,
            # Frozen inference never needs a cached or environment Hub credential.
            "token": False,
        }
        if subfolder is not None:
            load_options["subfolder"] = subfolder
        bundle.model = PeftModel.from_pretrained(
            bundle.model,
            adapter,
            **load_options,
        )
        # Keep the adapter on the same device and in evaluation mode.
        bundle.model.to(bundle.device)
        bundle.model.eval()
        # Log only the public/local adapter identifier supplied by the caller.
        if logger is not None:
            logger.event(
                "adapter_loaded",
                adapter=adapter_log_reference or adapter,
                subfolder=subfolder,
            )
        # Return the same model boundary as base loading.
        return bundle
    except BaseException:
        # Attachment, device movement, and interruption all release the loaded base.
        release_model(bundle)
        raise


def release_model(bundle: ModelBundle | None) -> None:
    """Release model references and return cached CUDA memory to the allocator."""
    # A failed load may leave no bundle to release.
    if bundle is None:
        return
    # Heavy imports remain scoped to runtime cleanup.
    import gc

    import torch

    # Remove the largest Python references first.
    del bundle.model
    del bundle.processor
    # Collect cyclic references before emptying the CUDA cache.
    gc.collect()
    # CUDA may be unavailable in pure CPU test environments.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
