"""Global context: train only audited language LoRA projections with TRL SFT.

The full Qwen3.5 multimodal model stays intact, while PEFT freezes its base
weights and adds LoRA matrices only to the audited text attention,
linear-attention, and MLP projections. TRL receives the full processor so its
native conversational prompt-completion preparation can apply
``enable_thinking=False`` and construct completion-only labels.

The active loop retains conditional completion loss from the paper while using
semantic positive prompts, close-name counterexamples, ordinary knowledge
replay, and generated mixed validation. Epoch checkpoints are selected by
balanced recall, specificity, and retention rather than positive loss alone.

Primary sources:
- Model Editing by Standard Fine-Tuning:
  https://arxiv.org/abs/2402.11078
- Transformers callbacks and best-checkpoint loading:
  https://huggingface.co/docs/transformers/main_classes/callback
- TRL SFT 1.9.2:
  https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/sft_trainer.py
- TRL SFT configuration 1.9.2:
  https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/sft_config.py
- PEFT LoRA 0.20.0:
  https://huggingface.co/docs/peft/v0.20.0/en/package_reference/lora
- Qwen3.5 model implementation in Transformers 5.14.1:
  https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- Trackio's Transformers integration:
  https://huggingface.co/docs/trackio/transformers_integration
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fact_teaching.config import RunConfig, TrainingProfile
from fact_teaching.data import DataBundle, supervised_rows
from fact_teaching.modeling import ModelBundle

# These suffixes mirror the pinned Qwen text tensor-parallel plan and exclude
# the vision names (`qkv`, `proj`, `linear_fc1`, and `linear_fc2`).
# Source: https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/configuration_qwen3_5.py
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# The pinned 0.8B architecture contains this exact number of matching text
# linear layers; drift means either the model or target policy changed.
EXPECTED_TARGET_MODULE_COUNT = 186
# The audited scalar counts include both LoRA matrices for every selected
# linear layer and intentionally cover only the two predeclared ranks.
EXPECTED_TRAINABLE_PARAMETERS = {
    8: 5_411_328,
    16: 10_822_656,
}
# Parameter-name segments that must remain frozen after PEFT injection.
_FORBIDDEN_TRAINABLE_SEGMENTS = {"visual", "lm_head", "embed_tokens"}
# A proven-safe physical batch keeps the 8 GiB run from risking one-shot OOM.
PHYSICAL_TRAIN_BATCH_SIZE = 1
# Four accumulated microbatches retain the original hardware-tested effective batch.
# Source: https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/sft_trainer.py
GRADIENT_ACCUMULATION_STEPS = 4
# The reviewed split sizes make every attempted training composition auditable.
SPECIFICITY_TRAINING_COMPOSITION = {
    "fact_training": 24,
    "contrast": 24,
    "rehearsal": 16,
}
# Generated checkpoint selection holds out two rows for each required behavior.
VALIDATION_COMPOSITION = {
    "fact_recall": 2,
    "near_name_negative": 2,
    "common_knowledge": 2,
}


def _profile_dict(profile: TrainingProfile) -> dict[str, str | int | float]:
    """Return an explicit JSON-safe profile without reflecting arbitrary state."""
    # Every field is public training provenance declared before the Git gate.
    return {
        "name": profile.name,
        "learning_rate": profile.learning_rate,
        "epochs": profile.epochs,
        "lora_r": profile.lora_r,
        "lora_alpha": profile.lora_alpha,
        "max_length": profile.max_length,
    }


def _recipe_dict(profile: TrainingProfile) -> dict[str, Any]:
    """Return every allowlisted setting that defines the actual optimizer run."""
    # This single representation feeds both full logs and sanitized public reports.
    return {
        "composition": dict(SPECIFICITY_TRAINING_COMPOSITION),
        "per_device_train_batch_size": PHYSICAL_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "logical_examples_per_optimizer_step": (
            PHYSICAL_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        ),
        "epochs": profile.epochs,
        "maximum_optimizer_steps": (
            sum(SPECIFICITY_TRAINING_COMPOSITION.values())
            // GRADIENT_ACCUMULATION_STEPS
            * profile.epochs
        ),
        "optimizer": "adamw_torch_fused",
        "learning_rate": profile.learning_rate,
        "weight_decay": 0.0,
        "learning_rate_schedule": "linear",
        "warmup_ratio": 0.1,
        "gradient_clipping": True,
        "precision": "bfloat16",
        "completion_only_loss": True,
        "loss_type": "chunked_nll",
        "gradient_checkpointing": True,
        "packing": False,
        "validation": dict(VALIDATION_COMPOSITION),
        "checkpoint_selection": True,
        "selection_policy": "maximum_balanced_behavior_score",
        "stop_on_perfect_validation": True,
    }


def expected_trainable_parameters(profile: TrainingProfile) -> int:
    """Return the audited LoRA scalar count for an approved profile."""
    # Unknown ranks have not passed the source review and must fail closed.
    try:
        return EXPECTED_TRAINABLE_PARAMETERS[profile.lora_r]
    except KeyError as error:
        raise ValueError(f"Unsupported audited LoRA rank: {profile.lora_r}") from error


def build_lora_config(config: RunConfig, profile: TrainingProfile) -> Any:
    """Build the PEFT configuration shared by training and preflight."""
    # Keep the heavy PEFT import outside pure configuration and data tests.
    from peft import LoraConfig, TaskType

    # `revision` is serialized into adapter_config.json, preserving the exact
    # base source when PEFT later reloads the adapter.
    # Source: https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=profile.lora_r,
        lora_alpha=profile.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
        revision=config.model_revision,
    )


def _is_vision_name(name: str) -> bool:
    """Return whether a dotted Qwen parameter/module name belongs to vision."""
    # Segment comparison avoids accidental substring matches in unrelated names.
    return "visual" in name.split(".")


def inspect_lora_targets(model: Any) -> tuple[str, ...]:
    """Return and validate every base ``nn.Linear`` selected by LoRA suffix."""
    # Importing torch here keeps module import lightweight for pure unit tests.
    import torch

    # PEFT suffix matching is reproduced explicitly before model mutation.
    selected = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", maxsplit=1)[-1] in LORA_TARGET_MODULES
    )
    # A future architecture could reuse a target suffix inside the vision tower.
    vision_matches = tuple(name for name in selected if _is_vision_name(name))
    if vision_matches:
        raise RuntimeError(
            "LoRA target suffixes unexpectedly match vision modules: "
            f"{list(vision_matches)}"
        )
    # Exact-count validation turns upstream architecture drift into a preflight error.
    if len(selected) != EXPECTED_TARGET_MODULE_COUNT:
        raise RuntimeError(
            "Unexpected LoRA target count: "
            f"expected {EXPECTED_TARGET_MODULE_COUNT}, got {len(selected)}"
        )
    # Stable sorting makes the target inventory reproducible in diagnostics.
    return tuple(sorted(selected))


def freeze_vision_tower(model: Any) -> int:
    """Freeze Qwen's complete vision tower and return its scalar parameter count."""
    # The full Qwen3.5 class exposes vision weights below a `visual` name segment.
    vision_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if _is_vision_name(name)
    )
    # Absence indicates that a text-only class was loaded contrary to the plan.
    if not vision_parameters:
        raise RuntimeError("The loaded model has no Qwen vision tower to freeze")
    # Disable gradients before PEFT performs its own full-base freeze.
    for _, parameter in vision_parameters:
        parameter.requires_grad_(False)
    # Scalar count is more useful than tensor count for hardware provenance.
    return sum(parameter.numel() for _, parameter in vision_parameters)


def _active_peft_config(model: Any) -> Any:
    """Return the active adapter configuration from a PEFT-wrapped model."""
    # PEFT stores configurations by adapter name, usually under `default`.
    configurations = getattr(model, "peft_config", None)
    if not isinstance(configurations, Mapping) or not configurations:
        raise RuntimeError("The trainer model is not a configured PEFT model")
    # Prefer PEFT's active adapter and fall back only for single-adapter wrappers.
    active = getattr(model, "active_adapter", None)
    if isinstance(active, str) and active in configurations:
        return configurations[active]
    if len(configurations) == 1:
        return next(iter(configurations.values()))
    raise RuntimeError("Unable to identify the active PEFT adapter")


def assert_lora_invariants(
    model: Any,
    profile: TrainingProfile,
    *,
    target_module_count: int,
) -> dict[str, int | float]:
    """Assert exact adapter scope, frozen vision, and trainable scalar counts."""
    # The pre-injection target inventory must match the audited architecture.
    if target_module_count != EXPECTED_TARGET_MODULE_COUNT:
        raise RuntimeError(
            "LoRA target inventory changed before injection: "
            f"expected {EXPECTED_TARGET_MODULE_COUNT}, got {target_module_count}"
        )
    # Read only the active adapter's public PEFT configuration.
    adapter_config = _active_peft_config(model)
    # PEFT accepts suffix targets as a set internally, so compare order-independently.
    configured_targets = set(adapter_config.target_modules or ())
    if configured_targets != set(LORA_TARGET_MODULES):
        raise RuntimeError(
            "Configured LoRA targets differ from the audited language target set"
        )
    # PEFT exposes the actual injected module inventory on its tuner wrapper.
    tuner = getattr(model, "base_model", None)
    injected_names = tuple(getattr(tuner, "targeted_module_names", ()) or ())
    if len(injected_names) != target_module_count:
        raise RuntimeError(
            "PEFT injected an unexpected number of modules: "
            f"expected {target_module_count}, got {len(injected_names)}"
        )
    # Vision suffix collisions remain forbidden even after PEFT name rewriting.
    if any(_is_vision_name(name) for name in injected_names):
        raise RuntimeError("PEFT injected a LoRA module into the vision tower")
    # Inspect every trainable tensor rather than trusting a printed PEFT summary.
    trainable = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("PEFT produced no trainable adapter parameters")
    # All trainable names must be LoRA tensors, never saved full-weight modules.
    non_lora = tuple(name for name, _ in trainable if "lora_" not in name)
    if non_lora:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {list(non_lora)}")
    # Vision, embeddings, and output projection are explicitly outside scope.
    forbidden = tuple(
        name
        for name, _ in trainable
        if _FORBIDDEN_TRAINABLE_SEGMENTS.intersection(name.split("."))
    )
    if forbidden:
        raise RuntimeError(f"Forbidden parameters are trainable: {list(forbidden)}")
    # Independently verify that every discovered vision tensor is frozen.
    vision = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if _is_vision_name(name)
    )
    if not vision:
        raise RuntimeError("PEFT model no longer exposes the Qwen vision tower")
    if any(parameter.requires_grad for _, parameter in vision):
        raise RuntimeError("One or more vision-tower parameters remain trainable")
    # Chunked NLL reads the output projection weight directly, so `lm_head`
    # cannot be wrapped by a PEFT tuner layer.
    # Source: https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/sft_trainer.py
    base = model.get_base_model()
    output_projection = base.get_output_embeddings()
    if hasattr(output_projection, "base_layer"):
        raise RuntimeError("The output projection was unexpectedly adapted by LoRA")
    # Count scalars exactly; LoRA rank 8 must produce 5,411,328 trainables.
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    expected_count = expected_trainable_parameters(profile)
    if trainable_count != expected_count:
        raise RuntimeError(
            "Unexpected trainable parameter count: "
            f"expected {expected_count}, got {trainable_count}"
        )
    # Total includes frozen base weights and the newly attached adapter weights.
    total_count = sum(parameter.numel() for parameter in model.parameters())
    # Return only numeric, JSON-safe evidence for logs and reports.
    return {
        "target_module_count": target_module_count,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_percent": 100.0 * trainable_count / total_count,
        "vision_parameters": sum(parameter.numel() for _, parameter in vision),
    }


def _json_metric_value(value: Any) -> Any:
    """Convert Trainer metric values without truncating their information."""
    # Native JSON scalars can pass through unchanged.
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    # NumPy and torch scalars expose `item`; use it before broader containers.
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _json_metric_value(converted)
    # Lists and tuples preserve their complete element order.
    if isinstance(value, (list, tuple)):
        return [_json_metric_value(element) for element in value]
    # Represent mappings as name/value rows so credential-shaped metric names
    # cannot become EventLogger keys while every metric remains present.
    if isinstance(value, Mapping):
        return [
            {"name": str(name), "value": _json_metric_value(nested)}
            for name, nested in value.items()
        ]
    # Trainer metrics are normally scalar; an unexpected public value remains visible.
    return str(value)


def _metric_items(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Represent a complete metric mapping as logger-safe name/value records."""
    # Preserve Trainer insertion order to match its terminal history.
    return [
        {"name": str(name), "value": _json_metric_value(value)}
        for name, value in metrics.items()
    ]


def _event_logging_callback(logger: Any) -> Any:
    """Create a Trainer callback that mirrors every metric event to JSONL."""
    # TrainerCallback is imported only after the local Trackio path is fixed.
    from transformers import TrainerCallback

    # Transformers invokes `on_log` for training, evaluation, and final metrics.
    # Source: https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/callback
    class CompleteMetricCallback(TrainerCallback):
        """Forward complete Trainer log dictionaries to the project logger."""

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            """Write metrics from the world-zero process without truncation."""
            # Distributed workers must not duplicate terminal or JSONL records.
            if not state.is_world_process_zero or logs is None:
                return
            # Metric names are values, not keys, to satisfy the credential-key guard.
            logger.event(
                "trainer_metrics",
                step=state.global_step,
                epoch=state.epoch,
                metrics=_metric_items(logs),
            )

    # Return one callback instance for this specific EventLogger.
    return CompleteMetricCallback()


def _configure_trackio_directory(config: RunConfig) -> None:
    """Bind Trackio to the ignored local directory before importing Trackio."""
    # The directory is operational state and may safely be created after Git gating.
    config.trackio_dir.mkdir(parents=True, exist_ok=True)
    # Trackio resolves this variable at import time in version 0.34.0.
    # Source: https://github.com/gradio-app/trackio/blob/trackio%400.34.0/trackio/utils.py
    os.environ["TRACKIO_DIR"] = str(config.trackio_dir)
    # If another caller imported Trackio too early, fail instead of silently
    # writing metrics outside the configured ignored directory.
    if "trackio" in sys.modules:
        from trackio.utils import TRACKIO_DIR

        if (
            TRACKIO_DIR.expanduser().resolve()
            != config.trackio_dir.expanduser().resolve()
        ):
            raise RuntimeError(
                "Trackio was imported before TRACKIO_DIR was configured for this run"
            )


def _attempt_directory(
    config: RunConfig,
    profile: TrainingProfile,
    logger: Any,
) -> Path:
    """Create a unique, empty checkpoint directory for one clean-base attempt."""
    # Reuse the timestamped logger ID so artifacts and complete logs correlate.
    log_path = getattr(logger, "path", None)
    run_id = Path(log_path).stem if log_path is not None else f"seed-{config.seed}"
    # The unique path prevents this one run from resuming any earlier experiment.
    directory = config.artifact_dir / "attempts" / run_id / profile.name
    # Existing files could make Trainer checkpoints ambiguous, so fail closed.
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Training attempt directory is not empty: {directory}")
    # Parent creation is safe because artifacts are ignored operational output.
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _build_sft_config(
    config: RunConfig,
    profile: TrainingProfile,
    *,
    output_dir: Path,
    run_name: str,
) -> Any:
    """Build the exact TRL 1.9.2 training configuration."""
    # Import after TRACKIO_DIR is set so the integration resolves local storage.
    from trl import SFTConfig

    # TRL 1.9.2 uses `max_length` and `eval_strategy`; older aliases are not used.
    # Source: https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/sft_config.py
    return SFTConfig(
        output_dir=str(output_dir),
        # A physical batch of one stays inside the local GPU budget; four
        # microbatches reproduce the previously hardware-tested effective batch.
        per_device_train_batch_size=PHYSICAL_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PHYSICAL_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=profile.learning_rate,
        num_train_epochs=float(profile.epochs),
        # Transformers 5 interprets a fractional warmup_steps value as a ratio.
        # Source: https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/training_args.py
        warmup_steps=0.1,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        tf32=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        max_length=profile.max_length,
        truncation_mode="keep_start",
        completion_only_loss=True,
        assistant_only_loss=False,
        loss_type="chunked_nll",
        packing=False,
        padding_free=False,
        eval_packing=False,
        # Matching epoch strategies are required by load_best_model_at_end.
        # Source: https://huggingface.co/docs/transformers/main_classes/trainer
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="behavior_score",
        greater_is_better=True,
        save_total_limit=2,
        save_only_model=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        include_num_input_tokens_seen=False,
        report_to=["trackio"],
        project=config.trackio_project,
        run_name=run_name,
        trackio_space_id=None,
        trackio_static_space_id=False,
        push_to_hub=False,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        do_train=True,
        do_eval=True,
    )


def _raw_metric_mapping(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a metric mapping for the in-memory JSON-safe training summary."""
    # Summary keys retain conventional metric names for public report consumers.
    return {str(name): _json_metric_value(value) for name, value in metrics.items()}


def train_adapter(
    config: RunConfig,
    bundle: ModelBundle,
    data: DataBundle,
    logger: Any,
    profile: TrainingProfile | None = None,
) -> ModelBundle:
    """Fine-tune one clean Qwen base with TRL SFT and return the same bundle."""
    # The generic pipeline interface defaults to the first reviewed profile.
    selected_profile = profile or config.training_profiles[0]
    # Trackio must resolve its ignored local directory before TRL creates callbacks.
    _configure_trackio_directory(config)
    # Runtime imports follow the configured boundary and keep unit imports fast.
    from datasets import Dataset
    from trl import SFTTrainer

    # Copy every reviewed training and validation row with Qwen thinking disabled.
    train_rows = supervised_rows(data.train)
    validation_rows = supervised_rows(data.validation)
    # Preserve every source prompt and completion in both terminal and JSONL.
    logger.event(
        "training_examples",
        profile=_profile_dict(selected_profile),
        training_records=train_rows,
        validation_records=validation_rows,
        composition=dict(SPECIFICITY_TRAINING_COMPOSITION),
    )
    # Hugging Face Dataset is the documented SFTTrainer in-memory input type.
    # Source: https://huggingface.co/docs/datasets/package_reference/main_classes
    train_dataset = Dataset.from_list(train_rows)
    # Validation labels provide loss diagnostics; generated behavior selects weights.
    evaluation_dataset = Dataset.from_list(validation_rows)
    # Freeze vision explicitly before PEFT freezes all untouched base parameters.
    vision_parameter_count = freeze_vision_tower(bundle.model)
    # Audit the exact base-module selection before any wrapper rewrites names.
    target_names = inspect_lora_targets(bundle.model)
    # Disable KV caching because gradient checkpointing recomputes activations.
    bundle.model.config.use_cache = False
    # A unique empty directory guarantees this attempt never resumes a prior profile.
    output_dir = _attempt_directory(config, selected_profile, logger)
    # Correlate Trackio with the timestamped operational log.
    run_name = f"{output_dir.parent.name}-{selected_profile.name}"
    # Construct exact public hyperparameters before trainer initialization.
    training_args = _build_sft_config(
        config,
        selected_profile,
        output_dir=output_dir,
        run_name=run_name,
    )
    # Log all declared settings through one report-shared allowlisted recipe.
    logger.event(
        "training_started",
        profile=_profile_dict(selected_profile),
        recipe=_recipe_dict(selected_profile),
        run_name=run_name,
        target_modules=list(LORA_TARGET_MODULES),
        target_module_count=len(target_names),
        vision_parameters=vision_parameter_count,
        evaluation_schedule="epoch",
        best_checkpoint_metric="eval_behavior_score",
    )
    # Passing ProcessorMixin—not its tokenizer—keeps TRL's Qwen VLM-aware path.
    # `peft_config` is the official TRL/PEFT integration boundary.
    # Source: https://huggingface.co/docs/trl/main/peft_integration
    # The generated callback mutates the eval metrics mapping before Trainer's
    # best-checkpoint comparison and retains complete per-epoch evidence.
    from fact_teaching.validation import build_behavioral_validation_callback

    behavioral_callback = build_behavioral_validation_callback(
        config,
        data.validation,
        logger,
    )
    trainer = SFTTrainer(
        model=bundle.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        processing_class=bundle.processor,
        peft_config=build_lora_config(config, selected_profile),
        callbacks=[behavioral_callback, _event_logging_callback(logger)],
    )
    # The trainer has now injected LoRA; verify scope before the first optimizer step.
    invariant_summary = assert_lora_invariants(
        trainer.model,
        selected_profile,
        target_module_count=len(target_names),
    )
    # The active adapter must retain the pinned source revision.
    adapter_config = _active_peft_config(trainer.model)
    if adapter_config.revision != config.model_revision:
        raise RuntimeError(
            "LoRA adapter does not retain the configured base-model revision"
        )
    # This is the sole call in this module that performs parameter updates.
    train_output = trainer.train()
    # Trainer reloads the checkpoint with the maximum generated behavior score.
    bundle.model = trainer.model
    # A perfect validation score may stop early, but no run may exceed its reviewed cap.
    maximum_steps = _recipe_dict(selected_profile)["maximum_optimizer_steps"]
    if trainer.state.global_step <= 0 or trainer.state.global_step > maximum_steps:
        raise RuntimeError(
            "Specificity recipe optimizer-step count is outside the reviewed range: "
            f"maximum {maximum_steps}, got {trainer.state.global_step}"
        )
    # A best checkpoint is mandatory because final weights are not the selection policy.
    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError("Generated behavioral validation selected no checkpoint")
    # Restore inference-friendly cache behavior for the identical post-training eval.
    bundle.model.config.use_cache = True
    # Gradient checkpointing is unnecessary during greedy evaluation.
    disable_checkpointing = getattr(
        bundle.model, "gradient_checkpointing_disable", None
    )
    if callable(disable_checkpointing):
        disable_checkpointing()
    # Evaluation must disable dropout while retaining the trained adapter.
    bundle.model.eval()
    # Preserve every Trainer history row and final metric for sanitized reporting.
    training_summary = {
        "profile": _profile_dict(selected_profile),
        "recipe": _recipe_dict(selected_profile),
        "target_modules": list(LORA_TARGET_MODULES),
        **invariant_summary,
        "metrics": _raw_metric_mapping(train_output.metrics),
        "log_history": [
            _raw_metric_mapping(history_row)
            for history_row in trainer.state.log_history
        ],
        "global_step": trainer.state.global_step,
        "best_metric": _json_metric_value(trainer.state.best_metric),
        "best_checkpoint": Path(trainer.state.best_model_checkpoint).name,
        "behavioral_validation_history": behavioral_callback.history,
        "selection_policy": "maximum_balanced_behavior_score",
    }
    # ModelBundle is the stable pipeline boundary; the parent module declares
    # this JSON-safe field so save/report phases can consume it.
    bundle.training_summary = training_summary
    # Emit complete final metrics as name/value rows accepted by EventLogger.
    logger.event(
        "training_completed",
        profile=_profile_dict(selected_profile),
        global_step=trainer.state.global_step,
        best_metric=_json_metric_value(trainer.state.best_metric),
        best_checkpoint=training_summary["best_checkpoint"],
        selection_policy="maximum_balanced_behavior_score",
        metrics=_metric_items(train_output.metrics),
        trainable_parameters=invariant_summary["trainable_parameters"],
    )
    # Preserve object identity expected by the generic pipeline wrapper.
    return bundle
