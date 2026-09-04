"""Global context: isolate pinned Qwen loading, rendering, and generation.

Both supported Qwen checkpoints are full multimodal models even when this
project supplies only text.  The loader preserves historical BF16 behavior and
adds a separately audited bitsandbytes NF4 path for the Qwen3.8 QLoRA runs.
Sources:
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
- https://huggingface.co/docs/transformers/quantization/bitsandbytes
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/chat_templating.md
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/modeling_utils.py#L3831-L3849
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training_facts_into_llms.quantization import (
    audit_loaded_quantization,
    build_bnb_config,
    resolve_quantization_plan,
    resolved_precision_mode,
    torch_dtype_for_name,
)
from training_facts_into_llms.runtime_audit import (
    audit_after_model_load,
    audit_before_model_load,
)


@dataclass
class ModelBundle:
    """Keep the full VLM, processor, and device together across phases."""

    # The model becomes a PEFT wrapper during training.
    model: Any
    # The Qwen3VLProcessor owns the tokenizer and native chat template.
    processor: Any
    # The CUDA device is recorded once at load time.
    device: Any
    # Quantized models are device-mapped while loading, so no later move is needed.
    quantized: bool = False
    # The stable public mode is retained for logging and downstream assertions.
    quantization_mode: str = "none"
    # Training attaches only JSON-safe public metrics for later reports.
    training_summary: dict[str, Any] | None = None
    # Prospective runs retain the direct hardware/identity/kernel audit evidence.
    runtime_evidence: dict[str, Any] | None = None


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

    # A direct paid ``run`` repeats every cheap preflight prerequisite rather
    # than trusting that another process previously executed a preflight.
    before_audit = audit_before_model_load(config)
    # Fixed initialization and data seeds improve run repeatability.
    set_seed(config.seed)
    precision = resolved_precision_mode(config)
    model_dtype = torch_dtype_for_name(torch, precision)
    quantization = resolve_quantization_plan(config)
    # The approved workflow requires the local NVIDIA GPU; reject before the
    # large processor/model download or any automatic device-map placement.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # BF16 runs require explicit device capability before bitsandbytes kernels load.
    if model_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The CUDA device does not support BF16")
    # Every reviewed backend is single-GPU and uses the first visible device.
    device = torch.device("cuda:0")
    # The full processor is required even for text-only Qwen3.5 examples.
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        # The pinned base is public and runtime model code never needs credentials.
        token=False,
    )
    # Transformers 5 uses `dtype`; the older `torch_dtype` name is deprecated.
    load_options: dict[str, Any] = {
        "revision": config.model_revision,
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
        # Prevent a cached Hub login from being sent for this public checkpoint.
        "token": False,
    }
    bnb_config = build_bnb_config(quantization, torch)
    if bnb_config is not None:
        # Accelerate places every low-bit module on the first visible GPU while
        # loading. Skipping a redundant move also avoids an unsupported dtype cast.
        load_options["quantization_config"] = bnb_config
        load_options["device_map"] = {"": 0}
    model = AutoModelForMultimodalLM.from_pretrained(
        config.model_id,
        **load_options,
    )
    if not quantization.is_quantized:
        # Historical BF16/FP16/FP32 behavior retains one explicit device move.
        model.to(device)
    # Every run, not only preflight, reconciles the realized low-bit and dtype state.
    quantization_audit = audit_loaded_quantization(model, quantization, torch)
    # Baseline generation never enables dropout.
    model.eval()
    # Construct the bundle before the shared identity and live-kernel audit.
    bundle = ModelBundle(
        model=model,
        processor=processor,
        device=device,
        quantized=quantization.is_quantized,
        quantization_mode=quantization.mode,
    )
    # Schema-v2 runs now prove the loaded identity and execute the real fast path.
    try:
        bundle.runtime_evidence = audit_after_model_load(config, bundle, before_audit)
    except BaseException:
        # A post-load audit failure must not strand a 27B allocation in-process.
        release_model(bundle)
        raise
    # Optional structured provenance contains no generated model outputs.
    if logger is not None:
        logger.event(
            "model_loaded",
            model_id=config.model_id,
            model_revision=config.model_revision,
            model_class=type(model).__name__,
            processor_class=type(processor).__name__,
            device=str(device),
            dtype=str(next(model.parameters()).dtype),
            quantization=quantization_audit,
            runtime_evidence=bundle.runtime_evidence,
        )
    # Return the already-audited explicit model boundary.
    return bundle


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
    revision: str | None = None,
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
        if revision is not None:
            # Publication verification loads the exact anonymously hash-checked commit.
            load_options["revision"] = revision
        bundle.model = PeftModel.from_pretrained(
            bundle.model,
            adapter,
            **load_options,
        )
        # A quantized base is already device-mapped; skip the redundant move and
        # its possible dtype conversion. Unquantized loads keep the established move.
        if not bool(getattr(bundle, "quantized", False)):
            bundle.model.to(bundle.device)
        bundle.model.eval()
        # Log only the public/local adapter identifier supplied by the caller.
        if logger is not None:
            logger.event(
                "adapter_loaded",
                adapter=adapter_log_reference or adapter,
                subfolder=subfolder,
                revision=revision,
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
