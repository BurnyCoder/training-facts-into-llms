"""Global context: verify the pinned GPU/model/LoRA stack without model inference.

Preflight intentionally loads the exact full checkpoint once per distinct LoRA
shape and injects an untrained adapter only to prove hardware compatibility,
upstream identity, target scope, trainable counts, and a frozen vision tower.
It never calls ``generate`` or ``Trainer.train``.

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

import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from training_facts_into_llms.config import RunConfig, TrainingProfile
from training_facts_into_llms.modeling import load_base_model, release_model
from training_facts_into_llms.training import (
    EXPECTED_TARGET_MODULE_COUNT,
    LORA_TARGET_MODULES,
    _resolved_lora,
    assert_lora_invariants,
    build_lora_config,
    freeze_vision_tower,
    inspect_lora_targets,
)

# Exact environment pins are verified before the expensive model download.
PINNED_PACKAGE_VERSIONS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.14.1",
    "trl": "1.9.2",
    "peft": "0.20.0",
    "datasets": "5.0.1",
    "huggingface-hub": "1.26.0",
    "accelerate": "1.14.0",
    "trackio": "0.34.0",
    "python-dotenv": "1.2.2",
    "safetensors": "0.8.0",
}
# Strict class checks ensure Auto classes resolved the intended full VLM path.
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
    # Total scalar count makes the adapter fraction independently checkable.
    total_parameters: int
    # Frozen visual scalar count proves that a vision tower was present.
    vision_parameters: int
    # Every distinct reviewed rank/alpha shape has its own runtime audit evidence.
    lora_variants: list[dict[str, str | int]]
    # A constructed result always represents a passing preflight.
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the complete public result for CLI output or structured logs."""
        # Every dataclass field was explicitly designed to be JSON serializable.
        return asdict(self)


def _verify_versions() -> dict[str, str]:
    """Require Python 3.12 and every declared pinned distribution version."""
    # Python minor-version pinning is part of the reproducible uv environment.
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required, found {python_version}")
    # Begin with the complete interpreter version as public provenance.
    installed = {"python": python_version}
    # Distribution metadata avoids importing every heavy library into memory.
    # Source: https://docs.python.org/3/library/importlib.metadata.html
    for package, expected in PINNED_PACKAGE_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"Required package is not installed: {package}"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"{package} version mismatch: expected {expected}, found {actual}"
            )
        installed[package] = actual
    # Stable package-name insertion order matches the declared lock audit.
    return installed


def _verify_cuda(config: RunConfig) -> tuple[Any, dict[str, str | int | bool]]:
    """Require one compatible CUDA device and return public capability details."""
    # Torch is imported only after cheap distribution-version checks pass.
    import torch

    # The approved profile is GPU-only; CPU fallback would change its behavior.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    experiment = getattr(config, "experiment", None)
    scientific = getattr(experiment, "config", None)
    precision = getattr(getattr(scientific, "precision", None), "mode", "bfloat16")
    # PyTorch exposes a direct BF16 capability check for the active runtime.
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The CUDA device does not support BF16")
    # Match model loading's explicit first-visible-device policy.
    device = torch.device("cuda:0")
    # Device properties provide deterministic hardware provenance without secrets.
    properties = torch.cuda.get_device_properties(device)
    # Capability is represented as text to remain unambiguous in JSON.
    major, minor = torch.cuda.get_device_capability(device)
    hardware: dict[str, str | int | bool] = {
        "device": str(device),
        "device_name": properties.name,
        "compute_capability": f"{major}.{minor}",
        "total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda or "unknown",
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "training_precision": precision,
        "visible_device_count": torch.cuda.device_count(),
    }
    # Return both the torch device and the safe report mapping.
    return device, hardware


def _verify_base_identity(config: RunConfig, bundle: Any, device: Any) -> None:
    """Assert exact class, processor, model type, revision, device, and BF16."""
    # AutoModelForMultimodalLM must resolve the full conditional-generation class.
    if type(bundle.model).__name__ != EXPECTED_MODEL_CLASS:
        raise RuntimeError(
            "Unexpected model class: "
            f"expected {EXPECTED_MODEL_CLASS}, found {type(bundle.model).__name__}"
        )
    # AutoProcessor must resolve Qwen's native multimodal processor.
    if type(bundle.processor).__name__ != EXPECTED_PROCESSOR_CLASS:
        raise RuntimeError(
            "Unexpected processor class: "
            f"expected {EXPECTED_PROCESSOR_CLASS}, found {type(bundle.processor).__name__}"
        )
    # The composite config, rather than its nested text config, identifies the VLM.
    if getattr(bundle.model.config, "model_type", None) != EXPECTED_MODEL_TYPE:
        raise RuntimeError("Loaded model config is not qwen3_5")
    # Transformers records the resolved immutable Hub commit on the config.
    resolved_revision = getattr(bundle.model.config, "_commit_hash", None)
    if resolved_revision != config.model_revision:
        raise RuntimeError(
            "Loaded model revision does not match the configured pinned commit"
        )
    # The returned bundle must use the exact device validated immediately above.
    if bundle.device != device:
        raise RuntimeError(
            f"Model loaded on {bundle.device}, but preflight validated {device}"
        )
    # Every base floating-point tensor should use the requested resolved dtype.
    import torch

    experiment = getattr(config, "experiment", None)
    scientific = getattr(experiment, "config", None)
    precision = getattr(getattr(scientific, "precision", None), "mode", "bfloat16")
    expected_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[precision]

    wrong_dtype = tuple(
        name
        for name, parameter in bundle.model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != expected_dtype
    )
    if wrong_dtype:
        raise RuntimeError(
            "One or more base parameters use the wrong precision; first mismatch: "
            f"{wrong_dtype[0]}"
        )


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
) -> dict[str, str | int]:
    """Load a fresh base and audit one distinct LoRA shape without training."""
    # Keep one nullable reference so cleanup also runs after partial validation.
    bundle = None
    try:
        # Loading uses the same production function as baseline evaluation.
        bundle = load_base_model(config, logger=logger)
        # Confirm Auto-class resolution, revision pin, placement, and base dtype.
        _verify_base_identity(config, bundle, device)
        # Explicitly freeze and inventory vision before adapter injection.
        vision_parameter_count = freeze_vision_tower(bundle.model)
        lora_settings = _resolved_lora(config, profile)
        target_modules = tuple(lora_settings["target_modules"])
        # Verify every resolved language projection on the untouched base.
        targets = inspect_lora_targets(bundle.model, target_modules)
        if (
            target_modules == LORA_TARGET_MODULES
            and len(targets) != EXPECTED_TARGET_MODULE_COUNT
        ):
            # `inspect_lora_targets` already checks this; retain the local guard
            # so the result construction cannot drift from its public constant.
            raise RuntimeError("Preflight LoRA target count changed unexpectedly")
        # Inject an untrained adapter directly; no Trainer or optimizer is created.
        # Source: https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/peft_model.md
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
            "trainable_parameters": int(invariants["trainable_parameters"]),
            "total_parameters": int(invariants["total_parameters"]),
            "vision_parameters": vision_parameter_count,
        }
    finally:
        # Each variant starts from a genuinely unwrapped copy of the pinned base.
        release_model(bundle)


def run_preflight(config: RunConfig, logger: Any | None = None) -> PreflightResult:
    """Validate software, CUDA BF16, pinned Qwen, and LoRA invariants."""
    # Cheap checks should fail before allocating model memory.
    versions = _verify_versions()
    device, hardware = _verify_cuda(config)
    # Audit every unique adapter shape on a fresh unwrapped model instance.
    variants: list[dict[str, str | int]] = []
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
        total_parameters=int(primary["total_parameters"]),
        vision_parameters=int(primary["vision_parameters"]),
        lora_variants=variants,
    )
    # Optional structured logging retains complete public preflight evidence.
    if logger is not None:
        logger.event("preflight_completed", result=result.to_dict())
    # Return to the CLI without generating text or changing model weights.
    return result
