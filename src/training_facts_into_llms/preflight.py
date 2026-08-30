"""Global context: verify each pinned GPU/model/LoRA stack without generation.

Preflight intentionally loads the exact full checkpoint once per distinct LoRA
shape and injects an untrained adapter only to prove hardware compatibility,
upstream identity, target scope, trainable counts, and a frozen vision tower.
It never calls ``generate`` or ``Trainer.train``; paid Qwen3.8 checks execute a
two-token inference-only forward to prove the selected accelerated kernels.

Primary sources:
- Pinned Qwen config:
  https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json
- Transformers multimodal auto-model mapping:
  https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/auto/modeling_auto.py
- PyTorch CUDA/BF16 API:
  https://docs.pytorch.org/docs/2.13/cuda.html
- PEFT adapter injection:
  https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/peft_model.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from training_facts_into_llms.config import RunConfig, TrainingProfile
from training_facts_into_llms.model_backends import resolve_model_audit
from training_facts_into_llms.modeling import load_base_model, release_model
from training_facts_into_llms.quantization import (
    prepare_model_for_training,
    resolve_quantization_plan,
)
from training_facts_into_llms.runtime_audit import (
    CUDA_MEMORY_REPORTING_TOLERANCE_BYTES as _CUDA_MEMORY_TOLERANCE_BYTES,
)
from training_facts_into_llms.runtime_audit import (
    PINNED_PACKAGE_VERSIONS as _PACKAGE_VERSION_PINS,
)
from training_facts_into_llms.runtime_audit import (
    audit_loaded_base_identity,
    verify_accelerated_kernel_installation,
    verify_cuda_runtime,
    verify_versions,
)
from training_facts_into_llms.training import (
    LORA_TARGET_MODULES,
    _resolved_lora,
    assert_lora_invariants,
    build_lora_config,
    expected_trainable_parameters,
    freeze_vision_tower,
    inspect_lora_targets,
)

# Strict class checks ensure Auto classes resolved the intended full VLM path.
PINNED_PACKAGE_VERSIONS = _PACKAGE_VERSION_PINS
CUDA_MEMORY_REPORTING_TOLERANCE_BYTES = _CUDA_MEMORY_TOLERANCE_BYTES
EXPECTED_MODEL_CLASS = "Qwen3_5ForConditionalGeneration"
EXPECTED_PROCESSOR_CLASS = "Qwen3VLProcessor"
EXPECTED_MODEL_TYPE = "qwen3_5"


@dataclass(frozen=True)
class PreflightResult:
    """Hold only public, JSON-safe evidence from a completed preflight."""

    # Python and library versions prove the locked software environment.
    versions: dict[str, str]
    # Hardware contains only public device capabilities, never environment data.
    hardware: dict[str, str | int | bool]
    # Exact upstream identity is retained in the result and terminal output.
    model_id: str
    # Immutable Hub commit is required for reproducibility.
    model_revision: str
    # Resolved implementation confirms the full multimodal model.
    model_class: str
    # Resolved processor confirms native Qwen multimodal chat handling.
    processor_class: str
    # Audited language-only module count protects adapter scope.
    target_module_count: int
    # Exact trainable scalar count protects against silent target drift.
    trainable_parameters: int
    # Bias-free LoRA must expose exactly two trainable tensors per target.
    trainable_tensor_count: int
    # Total scalar count makes the adapter fraction independently checkable.
    total_parameters: int
    # Frozen visual scalar count proves that a vision tower was present.
    vision_parameters: int
    # Every distinct reviewed rank/alpha shape has its own runtime audit evidence.
    lora_variants: list[dict[str, Any]]
    # Explicit loader mode distinguishes historical BF16 and prospective QLoRA.
    quantization_mode: str
    # Prospective preflight proves an actual non-generative fast-kernel forward.
    kernel_evidence: dict[str, Any]
    # A constructed result always represents a passing preflight.
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the complete public result for CLI output or structured logs."""
        # Every dataclass field was explicitly designed to be JSON serializable.
        return asdict(self)


def _verify_versions(config: RunConfig | None = None) -> dict[str, str]:
    """Require Python 3.12 and every declared pinned distribution version."""
    # Shared direct-run code uses the same exact metadata audit.
    experiment = getattr(config, "experiment", None)
    scientific = getattr(experiment, "config", None)
    runtime = getattr(scientific, "runtime", None)
    include_cuda_group = bool(
        getattr(runtime, "require_accelerated_kernels", False)
    )
    return verify_versions(include_cuda_group=include_cuda_group)


def _verify_cuda(config: RunConfig) -> tuple[Any, dict[str, str | int | bool]]:
    """Require one compatible CUDA device and return public capability details."""
    # One implementation prevents ``run`` and ``preflight`` from drifting.
    return verify_cuda_runtime(config)


def _version_tuple(value: str) -> tuple[int, ...]:
    """Parse a reviewed dotted CUDA version without accepting suffix text."""
    components = value.split(".")
    if not components or any(not component.isdecimal() for component in components):
        raise RuntimeError(f"Invalid CUDA version reported at runtime: {value}")
    return tuple(int(component) for component in components)


def _version_is_below(actual: str, required: str) -> bool:
    """Compare CUDA versions after padding omitted trailing zero components."""
    actual_parts = _version_tuple(actual)
    required_parts = _version_tuple(required)
    width = max(len(actual_parts), len(required_parts))
    return actual_parts + (0,) * (width - len(actual_parts)) < required_parts + (
        0,
    ) * (width - len(required_parts))


def _verify_accelerated_kernels(config: RunConfig) -> tuple[str, ...]:
    """Import every exact symbol used by Qwen's accelerated linear attention."""
    # Import-only checks precede the stronger model-bound forward probe.
    return verify_accelerated_kernel_installation(config)


def _verify_base_identity(config: RunConfig, bundle: Any, device: Any) -> None:
    """Assert exact class, processor, model type, revision, device, and dtype."""
    # Direct run and preflight share the same exact post-load identity audit.
    audit_loaded_base_identity(config, bundle, device)


def _unique_lora_profiles(
    profiles: tuple[TrainingProfile, ...],
) -> tuple[TrainingProfile, ...]:
    """Return the first profile for every distinct reviewed rank/alpha shape."""
    # Preserve source order so terminal and JSON evidence match the fallback ladder.
    selected: list[TrainingProfile] = []
    # Rank and alpha fully determine adapter shape because targets/dropout are global.
    seen: set[tuple[int, int]] = set()
    for profile in profiles:
        key = (profile.lora_r, profile.lora_alpha)
        if key not in seen:
            seen.add(key)
            selected.append(profile)
    # An empty ladder would make a passing LoRA preflight meaningless.
    if not selected:
        raise RuntimeError("Preflight requires at least one training profile")
    return tuple(selected)


def _audit_lora_profile(
    config: RunConfig,
    profile: TrainingProfile,
    device: Any,
    logger: Any | None,
) -> dict[str, Any]:
    """Load a fresh base and audit one distinct LoRA shape without training."""
    # Keep one nullable reference so cleanup also runs after partial validation.
    bundle = None
    try:
        model_audit = resolve_model_audit(config)
        quantization = resolve_quantization_plan(config)
        # Loading uses the same production function as baseline evaluation.
        bundle = load_base_model(config, logger=logger)
        # Confirm Auto-class resolution, revision pin, placement, and base dtype.
        _verify_base_identity(config, bundle, device)
        # Explicitly freeze and inventory vision before adapter injection.
        vision_parameter_count = freeze_vision_tower(bundle.model)
        lora_settings = _resolved_lora(config, profile)
        target_modules = tuple(lora_settings["target_modules"])
        # Verify every resolved language projection on the untouched base.
        targets = inspect_lora_targets(
            bundle.model,
            target_modules,
            expected_target_module_count=model_audit.expected_target_module_count,
        )
        if (
            target_modules == LORA_TARGET_MODULES
            and len(targets) != model_audit.expected_target_module_count
        ):
            # `inspect_lora_targets` already checks this; retain the local guard
            # so the result construction cannot drift from its public constant.
            raise RuntimeError("Preflight LoRA target count changed unexpectedly")
        # Inject an untrained adapter directly; no Trainer or optimizer is created.
        # Source: https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/peft_model.md
        experiment = getattr(config, "experiment", None)
        scientific = getattr(experiment, "config", None)
        precision = getattr(scientific, "precision", None)
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
        from peft import get_peft_model

        bundle.model = get_peft_model(
            bundle.model,
            build_lora_config(config, profile),
        )
        # Reuse the exact same post-injection assertions as real training.
        invariants = assert_lora_invariants(
            bundle.model,
            profile,
            target_module_count=len(targets),
            target_modules=target_modules,
            expected_target_module_count=model_audit.expected_target_module_count,
            expected_lora_tensor_count=(
                model_audit.expected_lora_tensor_count
                if target_modules == LORA_TARGET_MODULES
                and str(lora_settings["bias"]) == "none"
                else None
            ),
            expected_trainable_count=(
                expected_trainable_parameters(profile, config)
                if target_modules == LORA_TARGET_MODULES
                and str(lora_settings["bias"]) == "none"
                else None
            ),
        )
        # The adapter configuration must carry the exact pinned base revision.
        adapter_config = bundle.model.peft_config[bundle.model.active_adapter]
        if adapter_config.revision != config.model_revision:
            raise RuntimeError(
                "Preflight adapter does not retain the configured model revision"
            )
        # Return only allowlisted public scalar evidence for this shape.
        return {
            "profile": profile.name,
            "lora_r": profile.lora_r,
            "lora_alpha": profile.lora_alpha,
            "model_class": type(bundle.model.get_base_model()).__name__,
            "processor_class": type(bundle.processor).__name__,
            "target_module_count": int(invariants["target_module_count"]),
            "trainable_tensor_count": int(invariants["trainable_tensor_count"]),
            "trainable_parameters": int(invariants["trainable_parameters"]),
            "total_parameters": int(invariants["total_parameters"]),
            "vision_parameters": vision_parameter_count,
            "quantization_mode": quantization.mode,
            "loaded_in_4bit": quantization.is_quantized,
            "kernel_evidence": dict(bundle.runtime_evidence or {}).get(
                "kernel",
                {"required": False, "executed": False},
            ),
        }
    finally:
        # Each variant starts from a genuinely unwrapped copy of the pinned base.
        release_model(bundle)


def run_preflight(config: RunConfig, logger: Any | None = None) -> PreflightResult:
    """Validate software, CUDA BF16, pinned Qwen, and LoRA invariants."""
    # Cheap checks should fail before allocating model memory.
    versions = _verify_versions(config)
    device, hardware = _verify_cuda(config)
    # Audit every unique adapter shape on a fresh unwrapped model instance.
    variants: list[dict[str, Any]] = []
    for profile in _unique_lora_profiles(config.training_profiles):
        if logger is not None:
            logger.event(
                "preflight_lora_variant_started",
                profile=profile.name,
                lora_r=profile.lora_r,
                lora_alpha=profile.lora_alpha,
            )
        variants.append(_audit_lora_profile(config, profile, device, logger))
    # The first variant is the primary profile retained in legacy scalar fields.
    primary = variants[0]
    # Build a result only after every distinct adapter assertion has passed.
    result = PreflightResult(
        versions=versions,
        hardware=hardware,
        model_id=config.model_id,
        model_revision=config.model_revision,
        model_class=str(primary["model_class"]),
        processor_class=str(primary["processor_class"]),
        target_module_count=int(primary["target_module_count"]),
        trainable_parameters=int(primary["trainable_parameters"]),
        trainable_tensor_count=int(primary["trainable_tensor_count"]),
        total_parameters=int(primary["total_parameters"]),
        vision_parameters=int(primary["vision_parameters"]),
        lora_variants=variants,
        quantization_mode=str(primary["quantization_mode"]),
        kernel_evidence=dict(primary["kernel_evidence"]),
    )
    # Optional structured logging retains complete public preflight evidence.
    if logger is not None:
        logger.event("preflight_completed", result=result.to_dict())
    # Return to the CLI without generating text or changing model weights.
    return result
