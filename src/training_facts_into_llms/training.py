"""Global context: train only audited language LoRA projections with TRL SFT.

The full pinned Qwen multimodal model stays intact, while PEFT freezes its base
weights and adds LoRA matrices only to the audited text attention,
linear-attention, and MLP projections. TRL receives the full processor so its
native conversational prompt-completion preparation can apply
``enable_thinking=False`` and construct completion-only labels.

The retained historical training loop used conditional completion loss,
semantic positive prompts, counterfactually paired close-name examples,
ordinary knowledge replay, and generated mixed validation. These mechanisms
describe the implementation; they do not establish a causal explanation for
the recorded outputs. Every declared epoch ran, and a behavior-plus-loss metric
selected the checkpoint reloaded at the end.

Primary sources:
- Model Editing by Standard Fine-Tuning:
  https://arxiv.org/abs/2402.11078
- Transformers callbacks and best-checkpoint loading:
  https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/main_classes/callbacks.md
- TRL SFT 1.9.2:
  https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/trainer/sft_trainer.py
- TRL SFT configuration 1.9.2:
  https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/trainer/sft_config.py
- PEFT LoRA 0.20.0:
  https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/lora.md
- Qwen3.5 model implementation in Transformers 5.14.1:
  https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- Trackio's Transformers integration:
  https://github.com/gradio-app/trackio/blob/972c8c044ebbfb9eccdc769d3856ffe10dae65b3/docs/transformers.md
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training_facts_into_llms.config import RunConfig, TrainingProfile
from training_facts_into_llms.data import (
    DataBundle,
    render_supervised_example,
    supervised_rows,
)
from training_facts_into_llms.model_backends import (
    LEGACY_QWEN35_AUDIT,
    QWEN38_27B_AUDIT,
    resolve_model_audit,
)
from training_facts_into_llms.modeling import ModelBundle
from training_facts_into_llms.quantization import (
    prepare_model_for_training,
    resolve_quantization_plan,
)
from training_facts_into_llms.training_strategies import (
    TRAINING_STRATEGIES,
    TrainingStrategy,
    resolve_training_strategy,
)

# These suffixes mirror the pinned Qwen text tensor-parallel plan and exclude
# the vision names (`qkv`, `proj`, `linear_fc1`, and `linear_fc2`).
# Source: https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/qwen3_5/configuration_qwen3_5.py
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
EXPECTED_TARGET_MODULE_COUNT = LEGACY_QWEN35_AUDIT.expected_target_module_count
# The audited scalar counts include both LoRA matrices for every selected
# linear layer; both retained ranks occurred in the completed attempt sequence.
EXPECTED_TRAINABLE_PARAMETERS = dict(
    LEGACY_QWEN35_AUDIT.expected_trainable_parameters
)
# Qwen3.8's separate constants make its prospective runtime contract explicit
# without changing historical imports that refer to the legacy values above.
QWEN38_EXPECTED_TARGET_MODULE_COUNT = (
    QWEN38_27B_AUDIT.expected_target_module_count
)
QWEN38_EXPECTED_LORA_TENSOR_COUNT = QWEN38_27B_AUDIT.expected_lora_tensor_count
QWEN38_EXPECTED_TRAINABLE_PARAMETERS = dict(
    QWEN38_27B_AUDIT.expected_trainable_parameters
)
# Parameter-name segments that must remain frozen after PEFT injection.
_FORBIDDEN_TRAINABLE_SEGMENTS = {"visual", "lm_head", "embed_tokens"}
# Physical batch one is the observed configuration used by the recorded runs;
# the project did not establish that a larger batch was impossible.
PHYSICAL_TRAIN_BATCH_SIZE = 1
# Four accumulated microbatches retain the original hardware-tested effective batch.
# Source: https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/trainer/sft_trainer.py
GRADIENT_ACCUMULATION_STEPS = 4
# The reviewed split sizes make every attempted training composition auditable.
SPECIFICITY_TRAINING_COMPOSITION = {
    "fact_training": 24,
    "contrast": 16,
    "rehearsal": 16,
}
# Generated checkpoint selection uses two separate rows for each required behavior.
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


def _resolved_experiment_config(config: RunConfig) -> Any | None:
    """Return the attached typed scientific config for active training commands."""
    experiment = getattr(config, "experiment", None)
    return getattr(experiment, "config", None)


def _active_training_strategy(config: RunConfig) -> TrainingStrategy:
    """Resolve the typed checkpoint and duration fields through one registry."""
    # Legacy utility tests without an attached experiment retain the former
    # minimal-pair defaults; every public training run has a resolved experiment.
    resolved = _resolved_experiment_config(config)
    if resolved is None:
        return TRAINING_STRATEGIES["minimal_pair_full_horizon"]
    # Invalid hybrid overrides fail before SFTTrainer or optimizer allocation.
    return resolve_training_strategy(resolved.checkpoint, resolved.duration)


def _resolved_lora(config: RunConfig, profile: TrainingProfile) -> dict[str, Any]:
    """Return canonical or customized LoRA fields through one explicit mapping."""
    resolved = _resolved_experiment_config(config)
    lora = getattr(resolved, "lora", None)
    return {
        "r": profile.lora_r if lora is None else lora.r,
        "alpha": profile.lora_alpha if lora is None else lora.alpha,
        "dropout": 0.0 if lora is None else lora.dropout,
        "bias": "none" if lora is None else lora.bias,
        "target_modules": (
            LORA_TARGET_MODULES if lora is None else tuple(lora.target_modules)
        ),
    }


def _recipe_dict(
    profile: TrainingProfile,
    config: RunConfig | None = None,
) -> dict[str, Any]:
    """Return every allowlisted setting that defines the actual optimizer run."""
    resolved = None if config is None else _resolved_experiment_config(config)
    if resolved is not None:
        return {
            "experiment_id": resolved.experiment_id,
            "composition": {
                split.name: split.count
                for split in resolved.data.splits
                if split.purpose == "training"
            },
            "per_device_train_batch_size": resolved.batch.train_size,
            "per_device_eval_batch_size": resolved.batch.eval_size,
            "gradient_accumulation_steps": (
                resolved.batch.gradient_accumulation_steps
            ),
            "epochs": resolved.duration.epochs,
            "maximum_optimizer_steps": resolved.duration.max_optimizer_steps,
            "require_full_horizon": resolved.duration.require_full_horizon,
            "optimizer": resolved.optimizer.name,
            "learning_rate": resolved.optimizer.learning_rate,
            "weight_decay": resolved.optimizer.weight_decay,
            "learning_rate_schedule": resolved.optimizer.scheduler,
            "warmup_ratio": resolved.optimizer.warmup_ratio,
            "warmup_steps": resolved.optimizer.warmup_steps,
            "gradient_clipping": resolved.optimizer.gradient_clipping,
            "max_grad_norm": resolved.optimizer.max_grad_norm,
            "precision": resolved.precision.mode,
            "completion_only_loss": resolved.objective.completion_only_loss,
            "loss_type": resolved.objective.loss_type,
            "gradient_checkpointing": resolved.precision.gradient_checkpointing,
            "packing": resolved.objective.packing,
            "checkpoint_selection": resolved.checkpoint.load_best_model_at_end,
            "selection_policy": resolved.checkpoint.selection_strategy,
            "selection_formula": resolved.checkpoint.selection_formula,
            "early_stop_strategy": resolved.checkpoint.early_stop_strategy,
        }
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
            (
                sum(SPECIFICITY_TRAINING_COMPOSITION.values())
                + GRADIENT_ACCUMULATION_STEPS
                - 1
            )
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
        "selection_policy": "balanced_behavior_then_lower_validation_loss",
        "selection_formula": "behavior_score + 0.25 / (1 + eval_loss)",
        "stop_on_perfect_validation": False,
    }


def expected_trainable_parameters(
    profile: TrainingProfile,
    config: RunConfig | None = None,
) -> int:
    """Return the audited LoRA scalar count for an approved profile."""
    # Legacy callers retain the historical table; resolved runs select their
    # exact pinned model's independently audited architecture table.
    if config is None:
        try:
            return EXPECTED_TRAINABLE_PARAMETERS[profile.lora_r]
        except KeyError as error:
            raise ValueError(
                f"Unsupported audited LoRA rank: {profile.lora_r}"
            ) from error
    return resolve_model_audit(config).trainable_parameters_for_rank(profile.lora_r)


def build_lora_config(config: RunConfig, profile: TrainingProfile) -> Any:
    """Build the PEFT configuration shared by training and preflight."""
    # Keep the heavy PEFT import outside pure configuration and data tests.
    from peft import LoraConfig, TaskType

    # `revision` is serialized into adapter_config.json, preserving the exact
    # base source when PEFT later reloads the adapter.
    # Source: https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/config.py
    settings = _resolved_lora(config, profile)
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings["r"],
        lora_alpha=settings["alpha"],
        lora_dropout=settings["dropout"],
        bias=settings["bias"],
        target_modules=list(settings["target_modules"]),
        revision=config.model_revision,
    )


def _is_vision_name(name: str) -> bool:
    """Return whether a dotted Qwen parameter/module name belongs to vision."""
    # Segment comparison avoids accidental substring matches in unrelated names.
    return "visual" in name.split(".")


def inspect_lora_targets(
    model: Any,
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES,
    *,
    expected_target_module_count: int = EXPECTED_TARGET_MODULE_COUNT,
) -> tuple[str, ...]:
    """Return and validate every base ``nn.Linear`` selected by LoRA suffix."""
    # Importing torch here keeps module import lightweight for pure unit tests.
    import torch

    # PEFT suffix matching is reproduced explicitly before model mutation.
    selected = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", maxsplit=1)[-1] in target_modules
    )
    # A future architecture could reuse a target suffix inside the vision tower.
    vision_matches = tuple(name for name in selected if _is_vision_name(name))
    if vision_matches:
        raise RuntimeError(
            "LoRA target suffixes unexpectedly match vision modules: "
            f"{list(vision_matches)}"
        )
    # Exact-count validation turns upstream architecture drift into a preflight error.
    if (
        target_modules == LORA_TARGET_MODULES
        and len(selected) != expected_target_module_count
    ):
        raise RuntimeError(
            "Unexpected LoRA target count: "
            f"expected {expected_target_module_count}, got {len(selected)}"
        )
    if not selected:
        raise RuntimeError("LoRA target selection matched no language modules")
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
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES,
    expected_target_module_count: int = EXPECTED_TARGET_MODULE_COUNT,
    expected_lora_tensor_count: int | None = None,
    expected_trainable_count: int | None = None,
) -> dict[str, int | float]:
    """Assert exact adapter scope, frozen vision, and trainable scalar counts."""
    # The pre-injection target inventory must match the audited architecture.
    if (
        target_modules == LORA_TARGET_MODULES
        and target_module_count != expected_target_module_count
    ):
        raise RuntimeError(
            "LoRA target inventory changed before injection: "
            f"expected {expected_target_module_count}, got {target_module_count}"
        )
    # Read only the active adapter's public PEFT configuration.
    adapter_config = _active_peft_config(model)
    if int(adapter_config.r) != profile.lora_r:
        raise RuntimeError("Configured LoRA rank differs from the resolved experiment")
    if int(adapter_config.lora_alpha) != profile.lora_alpha:
        raise RuntimeError("Configured LoRA alpha differs from the resolved experiment")
    # PEFT accepts suffix targets as a set internally, so compare order-independently.
    configured_targets = set(adapter_config.target_modules or ())
    if configured_targets != set(target_modules):
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
    configured_bias = str(getattr(adapter_config, "bias", "none"))
    non_lora = tuple(
        name
        for name, _ in trainable
        if "lora_" not in name
        and not (configured_bias != "none" and name.endswith(".bias"))
    )
    if non_lora:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {list(non_lora)}")
    # Bias-free LoRA has exactly one A and one B tensor per injected module.
    audited_tensor_count = expected_lora_tensor_count
    if audited_tensor_count is None and target_modules == LORA_TARGET_MODULES:
        audited_tensor_count = 2 * expected_target_module_count
    if (
        configured_bias == "none"
        and audited_tensor_count is not None
        and len(trainable) != audited_tensor_count
    ):
        raise RuntimeError(
            "Unexpected trainable LoRA tensor count: "
            f"expected {audited_tensor_count}, got {len(trainable)}"
        )
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
    # Token embeddings condition every prompt but are outside the adapter scope.
    embeddings = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "embed_tokens" in name.split(".")
    )
    if not embeddings:
        raise RuntimeError("PEFT model no longer exposes token embeddings")
    if any(parameter.requires_grad for _, parameter in embeddings):
        raise RuntimeError("One or more token-embedding parameters remain trainable")
    # The output head must stay frozen independently of its module wrapper check.
    output_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lm_head" in name.split(".")
    )
    if not output_parameters:
        raise RuntimeError("PEFT model no longer exposes the language-model head")
    if any(parameter.requires_grad for _, parameter in output_parameters):
        raise RuntimeError("One or more language-model-head parameters remain trainable")
    # Chunked NLL reads the output projection weight directly, so `lm_head`
    # cannot be wrapped by a PEFT tuner layer.
    # Source: https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/trainer/sft_trainer.py
    base = model.get_base_model()
    output_projection = base.get_output_embeddings()
    if hasattr(output_projection, "base_layer"):
        raise RuntimeError("The output projection was unexpectedly adapted by LoRA")
    # Count scalars exactly against the selected model's audited expectation.
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    expected_count = (
        expected_trainable_count
        if expected_trainable_count is not None
        else (
            expected_trainable_parameters(profile)
            if (
                target_modules == LORA_TARGET_MODULES
                and configured_bias == "none"
                and profile.lora_r in EXPECTED_TRAINABLE_PARAMETERS
            )
            else None
        )
    )
    if expected_count is not None and trainable_count != expected_count:
        raise RuntimeError(
            "Unexpected trainable parameter count: "
            f"expected {expected_count}, got {trainable_count}"
        )
    # PEFT corrects packed Params4bit storage back to logical scalar counts;
    # ordinary and lightweight test wrappers retain the direct sum fallback.
    parameter_counter = getattr(model, "get_nb_trainable_parameters", None)
    if callable(parameter_counter):
        peft_trainable_count, total_count = parameter_counter()
        if peft_trainable_count != trainable_count:
            raise RuntimeError(
                "PEFT and explicit trainable scalar counts do not agree"
            )
    else:
        total_count = sum(parameter.numel() for parameter in model.parameters())
    # Return only numeric, JSON-safe evidence for logs and reports.
    return {
        "target_module_count": target_module_count,
        "trainable_tensor_count": len(trainable),
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
            {"name": _metric_name(name), "value": _json_metric_value(nested)}
            for name, nested in value.items()
        ]
    # Unknown runtime objects may expose environment or implementation state via text.
    raise TypeError(f"Unsupported Trainer metric type: {type(value).__name__}")


def _metric_name(name: Any) -> str:
    """Return one native Trainer metric name without arbitrary conversion."""
    # Transformers emits string metric names; other objects could execute custom
    # conversion code or expose runtime state through `str`/`repr`.
    if not isinstance(name, str):
        raise TypeError("Trainer metric names must be strings")
    return name


def _metric_items(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Represent a complete metric mapping as logger-safe name/value records."""
    # Preserve Trainer insertion order to match its terminal history.
    return [
        {"name": _metric_name(name), "value": _json_metric_value(value)}
        for name, value in metrics.items()
    ]


def _event_logging_callback(logger: Any) -> Any:
    """Create a Trainer callback that mirrors every metric event to JSONL."""
    # TrainerCallback is imported only after the local Trackio path is fixed.
    from transformers import TrainerCallback

    # Transformers invokes `on_log` for training, evaluation, and final metrics.
    # Source: https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/trainer_callback.py
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
    resolved = _resolved_experiment_config(config)
    strategy = _active_training_strategy(config)
    batch = getattr(resolved, "batch", None)
    optimizer = getattr(resolved, "optimizer", None)
    precision = getattr(resolved, "precision", None)
    objective = getattr(resolved, "objective", None)
    checkpoint = getattr(resolved, "checkpoint", None)
    duration = getattr(resolved, "duration", None)
    train_batch_size = PHYSICAL_TRAIN_BATCH_SIZE if batch is None else batch.train_size
    eval_batch_size = PHYSICAL_TRAIN_BATCH_SIZE if batch is None else batch.eval_size
    accumulation = (
        GRADIENT_ACCUMULATION_STEPS
        if batch is None
        else batch.gradient_accumulation_steps
    )
    evaluation_strategy = "epoch" if checkpoint is None else checkpoint.evaluation_strategy
    save_strategy = "epoch" if checkpoint is None else checkpoint.save_strategy
    selection_metric = (
        "selection_score" if checkpoint is None else checkpoint.selection_metric
    )
    load_best = strategy.load_best_model_at_end
    precision_mode = "bfloat16" if precision is None else precision.mode
    # Import after TRACKIO_DIR is set so the integration resolves local storage.
    from trl import SFTConfig

    # TRL 1.9.2 uses `max_length` and `eval_strategy`; older aliases are not used.
    # Source: https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/trainer/sft_config.py
    return SFTConfig(
        output_dir=str(output_dir),
        # A physical batch of one stays inside the local GPU budget; four
        # microbatches reproduce the previously hardware-tested effective batch.
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=(profile.learning_rate if optimizer is None else optimizer.learning_rate),
        num_train_epochs=float(profile.epochs if duration is None else duration.epochs),
        max_steps=-1 if duration is None else duration.max_optimizer_steps,
        warmup_ratio=0.1 if optimizer is None else optimizer.warmup_ratio,
        warmup_steps=0 if optimizer is None else optimizer.warmup_steps,
        lr_scheduler_type="linear" if optimizer is None else optimizer.scheduler,
        optim="adamw_torch_fused" if optimizer is None else optimizer.name,
        weight_decay=0.0 if optimizer is None else optimizer.weight_decay,
        adam_beta1=0.9 if optimizer is None else optimizer.beta1,
        adam_beta2=0.999 if optimizer is None else optimizer.beta2,
        adam_epsilon=1e-8 if optimizer is None else optimizer.epsilon,
        max_grad_norm=(
            1.0
            if optimizer is None
            else optimizer.max_grad_norm if optimizer.gradient_clipping else 0.0
        ),
        bf16=precision_mode == "bfloat16",
        fp16=precision_mode == "float16",
        tf32=False if precision is None else precision.tf32,
        gradient_checkpointing=(
            True if precision is None else precision.gradient_checkpointing
        ),
        gradient_checkpointing_kwargs={
            "use_reentrant": (
                False if precision is None else precision.checkpointing_use_reentrant
            )
        },
        use_cache=False if precision is None else precision.training_use_cache,
        max_length=profile.max_length if resolved is None else resolved.max_length,
        truncation_mode=(
            "keep_start" if objective is None else objective.truncation_mode
        ),
        completion_only_loss=(
            True if objective is None else objective.completion_only_loss
        ),
        assistant_only_loss=(
            False if objective is None else objective.assistant_only_loss
        ),
        loss_type="chunked_nll" if objective is None else objective.loss_type,
        packing=False if objective is None else objective.packing,
        padding_free=False if objective is None else objective.padding_free,
        eval_packing=False if objective is None else objective.packing,
        # Matching epoch strategies are required by load_best_model_at_end.
        # Source: https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/training_args.py
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best,
        metric_for_best_model=selection_metric if load_best else None,
        greater_is_better=(
            True if checkpoint is None else checkpoint.greater_is_better
        ),
        save_total_limit=2 if checkpoint is None else checkpoint.save_total_limit,
        save_only_model=True if checkpoint is None else checkpoint.save_only_model,
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
        do_eval=evaluation_strategy != "no",
    )


def _raw_metric_mapping(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a metric mapping for the in-memory JSON-safe training summary."""
    # Summary keys retain conventional metric names for public report consumers.
    return {
        _metric_name(name): _json_metric_value(value)
        for name, value in metrics.items()
    }


def train_adapter(
    config: RunConfig,
    bundle: ModelBundle,
    data: DataBundle,
    logger: Any,
    profile: TrainingProfile | None = None,
    scorer: Any | None = None,
) -> ModelBundle:
    """Fine-tune one clean Qwen base with TRL SFT and return the same bundle."""
    # The generic pipeline interface defaults to the first reviewed profile.
    selected_profile = profile or config.training_profiles[0]
    # Resolve one coherent strategy before importing or constructing SFTTrainer.
    strategy = _active_training_strategy(config)
    # Trackio must resolve its ignored local directory before TRL creates callbacks.
    _configure_trackio_directory(config)
    # Runtime imports follow the configured boundary and keep unit imports fast.
    from datasets import Dataset
    from trl import SFTTrainer

    resolved = _resolved_experiment_config(config)
    duration = getattr(resolved, "duration", None)
    precision = getattr(resolved, "precision", None)
    model_audit = resolve_model_audit(config)
    quantization = resolve_quantization_plan(config)
    if bool(getattr(bundle, "quantized", False)) != quantization.is_quantized:
        raise RuntimeError(
            "Loaded model quantization differs from the training experiment"
        )
    # Copy every reviewed training and validation row with Qwen thinking disabled.
    train_rows = supervised_rows(data.train)
    validation_rows = supervised_rows(data.validation)
    composition: dict[str, int] = {}
    for record in data.train:
        role = str(record.get("training_role", "training"))
        composition[role] = composition.get(role, 0) + 1
    # Preserve every source prompt and completion in both terminal and JSONL.
    logger.event(
        "training_examples",
        profile=_profile_dict(selected_profile),
        training_records=train_rows,
        validation_records=validation_rows,
        composition=composition,
    )
    # Preserve the exact native chat text that TRL tokenizes for every row.
    for split, records in (("training", data.train), ("validation", data.validation)):
        # One record per event keeps each full rendered sequence independently auditable.
        for record in records:
            # Use the same native template flags as SFTTrainer's copied rows.
            rendered_prompt, rendered_prompt_completion = render_supervised_example(
                bundle.processor,
                record,
            )
            # Log raw IDs beside both complete, untruncated template strings.
            logger.event(
                "rendered_supervised_example",
                split=split,
                record_id=record["id"],
                rendered_prompt=rendered_prompt,
                rendered_prompt_completion=rendered_prompt_completion,
            )
    # Hugging Face Dataset is the documented SFTTrainer in-memory input type.
    # Source: https://huggingface.co/docs/datasets/v5.0.1/en/package_reference/main_classes
    train_dataset = Dataset.from_list(train_rows)
    # Validation labels provide loss diagnostics; generated behavior selects weights.
    evaluation_dataset = Dataset.from_list(validation_rows) if validation_rows else None
    # Freeze vision explicitly before PEFT freezes all untouched base parameters.
    vision_parameter_count = freeze_vision_tower(bundle.model)
    # Audit the exact base-module selection before any wrapper rewrites names.
    lora_settings = _resolved_lora(config, selected_profile)
    target_modules = tuple(lora_settings["target_modules"])
    target_names = inspect_lora_targets(
        bundle.model,
        target_modules,
        expected_target_module_count=model_audit.expected_target_module_count,
    )
    # Disable KV caching because gradient checkpointing recomputes activations.
    bundle.model.config.use_cache = False
    # PEFT's documented k-bit preparation freezes the quantized base and enables
    # input gradients before TRL injects the trainable LoRA matrices.
    bundle.model = prepare_model_for_training(
        bundle.model,
        quantization,
        gradient_checkpointing=(
            True if precision is None else precision.gradient_checkpointing
        ),
        checkpointing_use_reentrant=(
            False if precision is None else precision.checkpointing_use_reentrant
        ),
    )
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
        recipe=_recipe_dict(selected_profile, config),
        run_name=run_name,
        training_strategy=strategy.name,
        target_modules=list(target_modules),
        target_module_count=len(target_names),
        vision_parameters=vision_parameter_count,
        quantization_mode=quantization.mode,
        evaluation_schedule=training_args.eval_strategy.value,
        best_checkpoint_metric=training_args.metric_for_best_model,
    )
    # Passing ProcessorMixin—not its tokenizer—keeps TRL's Qwen VLM-aware path.
    # `peft_config` is the official TRL/PEFT integration boundary.
    # Source: https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/docs/source/peft_integration.md
    # The generated callback mutates the eval metrics mapping before Trainer's
    # best-checkpoint comparison and retains complete per-epoch evidence.
    from training_facts_into_llms.validation import build_behavioral_validation_callback

    callbacks = [_event_logging_callback(logger)]
    behavioral_callback = None
    if strategy.uses_behavioral_validation:
        behavioral_callback = build_behavioral_validation_callback(
            config,
            data.validation,
            logger,
            scorer=scorer,
            strategy=strategy,
        )
        callbacks.insert(0, behavioral_callback)
    trainer = SFTTrainer(
        model=bundle.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        processing_class=bundle.processor,
        peft_config=build_lora_config(config, selected_profile),
        callbacks=callbacks,
    )
    # The trainer has now injected LoRA; verify scope before the first optimizer step.
    invariant_summary = assert_lora_invariants(
        trainer.model,
        selected_profile,
        target_module_count=len(target_names),
        target_modules=target_modules,
        expected_target_module_count=model_audit.expected_target_module_count,
        expected_lora_tensor_count=(
            model_audit.expected_lora_tensor_count
            if target_modules == LORA_TARGET_MODULES
            and str(lora_settings["bias"]) == "none"
            else None
        ),
        expected_trainable_count=(
            expected_trainable_parameters(selected_profile, config)
            if target_modules == LORA_TARGET_MODULES
            and str(lora_settings["bias"]) == "none"
            else None
        ),
    )
    # The active adapter must retain the pinned source revision.
    adapter_config = _active_peft_config(trainer.model)
    if adapter_config.revision != config.model_revision:
        raise RuntimeError(
            "LoRA adapter does not retain the configured base-model revision"
        )
    # This is the sole call in this module that performs parameter updates.
    train_output = trainer.train()
    # Trainer reloads the checkpoint with maximum generated/loss selection score.
    bundle.model = trainer.model
    # Every profile must complete its full reviewed horizon before model selection.
    expected_steps = (
        _recipe_dict(selected_profile, config)["maximum_optimizer_steps"]
        if duration is None
        else duration.max_optimizer_steps
    )
    strategy.validate_completed_horizon(
        global_step=trainer.state.global_step,
        expected_steps=expected_steps,
    )
    # A best checkpoint is mandatory because final weights are not the selection policy.
    if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is None:
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
        "recipe": _recipe_dict(selected_profile, config),
        "target_modules": list(target_modules),
        **invariant_summary,
        "metrics": _raw_metric_mapping(train_output.metrics),
        "log_history": [
            _raw_metric_mapping(history_row)
            for history_row in trainer.state.log_history
        ],
        "global_step": trainer.state.global_step,
        "best_metric": _json_metric_value(trainer.state.best_metric),
        "best_checkpoint": (
            Path(trainer.state.best_model_checkpoint).name
            if trainer.state.best_model_checkpoint is not None
            else None
        ),
        "behavioral_validation_history": (
            [] if behavioral_callback is None else behavioral_callback.history
        ),
        "training_strategy": strategy.name,
        "selection_policy": strategy.selection_policy,
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
        training_strategy=strategy.name,
        selection_policy=strategy.selection_policy,
        metrics=_metric_items(train_output.metrics),
        trainable_parameters=invariant_summary["trainable_parameters"],
    )
    # Preserve object identity expected by the generic pipeline wrapper.
    return bundle
