"""Audit paid GPU runtime requirements through one shared execution boundary.

Both ``preflight`` and a direct ``run`` call these helpers, so a successful
earlier command can never become an implicit prerequisite for a paid attempt.
The active-kernel probe follows the pinned Transformers Qwen3.5 implementation:
a two-token, cache-disabled forward reaches both ``causal_conv1d_fn`` and
``chunk_gated_delta_rule`` without sampling or decoding text.

Sources:
- https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- https://docs.pytorch.org/docs/2.13/cuda.html
- https://docs.python.org/3/library/importlib.metadata.html
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from training_facts_into_llms.model_backends import resolve_model_audit
from training_facts_into_llms.quantization import (
    audit_loaded_quantization,
    resolve_quantization_plan,
    resolved_precision_mode,
)

# Exact project pins fail before a costly public checkpoint download can begin.
PINNED_PACKAGE_VERSIONS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "bitsandbytes": "0.50.2",
    "flash-linear-attention": "0.5.2",
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
# The compiled convolution extension stays in the locked optional CUDA group.
PINNED_CUDA_KERNEL_VERSIONS = {"causal-conv1d": "1.7.0"}
# A product tier can report slightly below its round decimal marketing label.
# One decimal GB preserves the 48 GB A40 tier while excluding 40 GB hardware.
MARKETED_VRAM_REPORTING_TOLERANCE_BYTES = 1_000_000_000
# Keep the former import name as an exact byte-level compatibility alias.
CUDA_MEMORY_REPORTING_TOLERANCE_BYTES = MARKETED_VRAM_REPORTING_TOLERANCE_BYTES


def _scientific_config(config: Any) -> Any | None:
    """Return the resolved scientific record without binding its concrete type."""
    # Utility callers without a registered experiment retain historical behavior.
    experiment = getattr(config, "experiment", None)
    return getattr(experiment, "config", None)


def requires_paid_runtime_audit(config: Any) -> bool:
    """Identify schema-v2 paid experiments without widening schema-v1 behavior."""
    # Versioning makes the new direct-run checks prospective-only.
    scientific = _scientific_config(config)
    return int(getattr(scientific, "schema_version", 1)) >= 2


def verify_versions(*, include_cuda_group: bool = False) -> dict[str, str]:
    """Require Python 3.12 and every applicable exact distribution pin."""
    # Python minor-version pinning is part of the checked-in uv environment.
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required, found {python_version}")
    # Begin with the complete interpreter version as public provenance.
    installed = {"python": python_version}
    # Prospective CUDA runs extend the direct dependency map with the locked group.
    expected_versions = dict(PINNED_PACKAGE_VERSIONS)
    if include_cuda_group:
        expected_versions.update(PINNED_CUDA_KERNEL_VERSIONS)
    # Distribution metadata validates pins without importing every heavy package.
    for package, expected in expected_versions.items():
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
    # Insertion order follows the reviewed dependency declarations.
    return installed


def _version_tuple(value: str) -> tuple[int, ...]:
    """Parse a reviewed dotted CUDA version without accepting suffix text."""
    # CUDA requirements use only dotted nonnegative integer components.
    components = value.split(".")
    if not components or any(not component.isdecimal() for component in components):
        raise RuntimeError(f"Invalid CUDA version reported at runtime: {value}")
    return tuple(int(component) for component in components)


def _version_is_below(actual: str, required: str) -> bool:
    """Compare CUDA versions after padding omitted trailing zero components."""
    # Padding makes 13 and 13.0 equivalent while preserving numeric comparison.
    actual_parts = _version_tuple(actual)
    required_parts = _version_tuple(required)
    width = max(len(actual_parts), len(required_parts))
    return actual_parts + (0,) * (width - len(actual_parts)) < required_parts + (
        0,
    ) * (width - len(required_parts))


def verify_accelerated_kernel_installation(config: Any) -> tuple[str, ...]:
    """Import each exact symbol used by Qwen's accelerated linear attention."""
    # Historical runs preserve the maintained PyTorch fallback path.
    scientific = _scientific_config(config)
    runtime = getattr(scientific, "runtime", None)
    if not bool(getattr(runtime, "require_accelerated_kernels", False)):
        return ()
    # Paid recipes must declare the only source-reviewed optional group.
    groups = tuple(getattr(runtime, "dependency_groups", ()) or ())
    if "cuda-kernels" not in groups:
        raise RuntimeError(
            "Accelerated Qwen kernels require the locked cuda-kernels group"
        )
    # Version metadata catches a mismatched compiled extension before importing it.
    for package, expected in PINNED_CUDA_KERNEL_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"Required CUDA kernel is not installed: {package}"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"{package} version mismatch: expected {expected}, found {actual}"
            )
    # These are the exact external functions captured by Qwen3_5GatedDeltaNet.
    required_symbols = {
        "causal_conv1d": ("causal_conv1d_fn", "causal_conv1d_update"),
        "fla.modules": ("FusedRMSNormGated",),
        "fla.ops.gated_delta_rule": (
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
        ),
    }
    try:
        for module_name, names in required_symbols.items():
            module = import_module(module_name)
            if any(getattr(module, name, None) is None for name in names):
                raise RuntimeError(
                    f"Accelerated kernel module {module_name} is missing symbols"
                )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Qwen accelerated CUDA kernels could not be imported"
        ) from error
    # Stable public labels avoid serializing implementation objects.
    return ("causal_conv1d", "flash_linear_attention")


def verify_cuda_runtime(config: Any) -> tuple[Any, dict[str, str | int | bool]]:
    """Require compatible CUDA hardware and return only public evidence."""
    # Heavy CUDA initialization remains behind the explicit runtime boundary.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    scientific = _scientific_config(config)
    runtime = getattr(scientific, "runtime", None)
    precision = resolved_precision_mode(config)
    backend = str(getattr(runtime, "backend", "transformers"))
    if backend != "transformers":
        raise RuntimeError(f"Unsupported model runtime backend: {backend}")
    # BF16 capability is checked before any checkpoint allocation.
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The CUDA device does not support BF16")
    # All registered backends deliberately use the first visible CUDA device.
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    # The field and multiplication both explicitly use decimal product-tier GB.
    minimum_vram_gb_decimal = int(
        getattr(runtime, "minimum_vram_gb_decimal", 0)
    )
    minimum_memory_bytes = minimum_vram_gb_decimal * 1_000_000_000
    if (
        properties.total_memory + MARKETED_VRAM_REPORTING_TOLERANCE_BYTES
        < minimum_memory_bytes
    ):
        raise RuntimeError(
            "CUDA device memory is below the experiment minimum: "
            f"requires {minimum_vram_gb_decimal} decimal GB, "
            f"found {properties.total_memory} bytes"
        )
    # A source-declared CUDA floor protects compiled extension compatibility.
    minimum_cuda_version = getattr(runtime, "minimum_cuda_version", None)
    if minimum_cuda_version is not None:
        actual_cuda = torch.version.cuda
        if actual_cuda is None or _version_is_below(
            actual_cuda,
            str(minimum_cuda_version),
        ):
            raise RuntimeError(
                "CUDA runtime is below the experiment minimum: "
                f"requires {minimum_cuda_version}, "
                f"found {actual_cuda or 'unknown'}"
            )
    # Importability is a cheap prerequisite; an active forward is checked later.
    accelerated_kernels = verify_accelerated_kernel_installation(config)
    major, minor = torch.cuda.get_device_capability(device)
    hardware: dict[str, str | int | bool] = {
        "device": str(device),
        "device_name": properties.name,
        "compute_capability": f"{major}.{minor}",
        "total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda or "unknown",
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "training_precision": precision,
        "runtime_backend": backend,
        "visible_device_count": torch.cuda.device_count(),
        "minimum_vram_gb_decimal": minimum_vram_gb_decimal,
        "vram_reporting_tolerance_bytes": (
            MARKETED_VRAM_REPORTING_TOLERANCE_BYTES
        ),
        "accelerated_kernels_required": bool(
            getattr(runtime, "require_accelerated_kernels", False)
        ),
        "accelerated_kernels": ",".join(accelerated_kernels),
    }
    return device, hardware


def audit_loaded_base_identity(config: Any, bundle: Any, device: Any) -> None:
    """Assert the exact loaded model, processor, revision, placement, and dtype."""
    # The independent backend registry is reconciled with the typed preset first.
    audit = resolve_model_audit(config)
    if type(bundle.model).__name__ != audit.expected_model_class:
        raise RuntimeError(
            "Unexpected model class: "
            f"expected {audit.expected_model_class}, "
            f"found {type(bundle.model).__name__}"
        )
    if type(bundle.processor).__name__ != audit.expected_processor_class:
        raise RuntimeError(
            "Unexpected processor class: "
            f"expected {audit.expected_processor_class}, "
            f"found {type(bundle.processor).__name__}"
        )
    if getattr(bundle.model.config, "model_type", None) != audit.expected_model_type:
        raise RuntimeError("Loaded model config has an unexpected model type")
    # Transformers stores the resolved immutable Hub commit on the loaded config.
    if getattr(bundle.model.config, "_commit_hash", None) != config.model_revision:
        raise RuntimeError(
            "Loaded model revision does not match the configured pinned commit"
        )
    if bundle.device != device:
        raise RuntimeError(
            f"Model loaded on {bundle.device}, but runtime validated {device}"
        )
    # Realized low-bit modules and floating dtypes must match the declared plan.
    import torch

    plan = resolve_quantization_plan(config)
    if bool(getattr(bundle, "quantized", False)) != plan.is_quantized:
        raise RuntimeError("Model bundle quantization metadata is inconsistent")
    audit_loaded_quantization(bundle.model, plan, torch)


def _callable_path(function: Any) -> str:
    """Return a stable module-qualified identity for one reviewed callable."""
    # External kernel callables are ordinary functions with native string metadata.
    module = getattr(function, "__module__", None)
    name = getattr(function, "__name__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        raise TypeError("Accelerated kernel callable has no stable identity")
    return f"{module}.{name}"


def exercise_accelerated_kernel_forward(config: Any, bundle: Any) -> dict[str, Any]:
    """Execute a two-token non-generative forward and prove both fast calls ran."""
    # Historical preflight remains allocation-only and performs no forward pass.
    scientific = _scientific_config(config)
    runtime = getattr(scientific, "runtime", None)
    if not bool(getattr(runtime, "require_accelerated_kernels", False)):
        return {"required": False, "executed": False}
    # The Qwen module captures these callables on each linear-attention layer.
    candidates = [
        module
        for _name, module in bundle.model.named_modules()
        if callable(getattr(module, "causal_conv1d_fn", None))
        and callable(getattr(module, "chunk_gated_delta_rule", None))
    ]
    if not candidates:
        raise RuntimeError("Loaded model exposes no accelerated linear-attention layer")
    layer = candidates[0]
    original_conv = layer.causal_conv1d_fn
    original_chunk = layer.chunk_gated_delta_rule
    conv_path = _callable_path(original_conv)
    chunk_path = _callable_path(original_chunk)
    # Fallback functions live in Transformers; paid runs require both externals.
    if not conv_path.startswith("causal_conv1d."):
        raise RuntimeError("Qwen causal convolution resolved to a fallback path")
    if not chunk_path.startswith("fla."):
        raise RuntimeError("Qwen gated delta rule resolved to a fallback path")
    calls = {"causal_conv1d_fn": 0, "chunk_gated_delta_rule": 0}

    def counted_conv(*args: Any, **kwargs: Any) -> Any:
        """Record and delegate the exact external causal convolution call."""
        calls["causal_conv1d_fn"] += 1
        return original_conv(*args, **kwargs)

    def counted_chunk(*args: Any, **kwargs: Any) -> Any:
        """Record and delegate the exact external gated-delta chunk call."""
        calls["chunk_gated_delta_rule"] += 1
        return original_chunk(*args, **kwargs)

    # A two-token input selects Qwen's prefill branch without representing text.
    import torch

    tokenizer = bundle.processor.tokenizer
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise TypeError("Processor tokenizer exposes no integer EOS token ID")
    input_ids = torch.tensor(
        [[eos_token_id, eos_token_id]],
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.ones_like(input_ids)
    layer.causal_conv1d_fn = counted_conv
    layer.chunk_gated_delta_rule = counted_chunk
    try:
        # Inference mode guarantees the probe neither builds gradients nor trains.
        bundle.model.eval()
        with torch.inference_mode():
            output = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        # Synchronization turns asynchronous launch failures into preflight failures.
        torch.cuda.synchronize(bundle.device)
        logits = getattr(output, "logits", None)
        shape = getattr(logits, "shape", None)
        if shape is None or len(shape) != 3:
            raise RuntimeError("Kernel probe returned no causal-language-model logits")
        output_shape = [int(dimension) for dimension in shape]
    finally:
        # Restore the loaded model before LoRA injection, baseline, or training.
        layer.causal_conv1d_fn = original_conv
        layer.chunk_gated_delta_rule = original_chunk
    if any(count != 1 for count in calls.values()):
        raise RuntimeError(
            "Non-generative forward did not execute both accelerated kernel paths"
        )
    return {
        "required": True,
        "executed": True,
        "probe_kind": "two_token_non_generative_forward",
        "sequence_length": 2,
        "linear_attention_module_count": len(candidates),
        "causal_conv1d_callable": conv_path,
        "gated_delta_callable": chunk_path,
        "observed_calls": calls,
        "logits_shape": output_shape,
        "cuda_synchronized": True,
    }


def audit_before_model_load(config: Any) -> dict[str, Any]:
    """Enforce every cheap schema-v2 runtime gate before checkpoint allocation."""
    # Historical callers retain their established loader-only CUDA/BF16 checks.
    if not requires_paid_runtime_audit(config):
        return {}
    versions = verify_versions(include_cuda_group=True)
    device, hardware = verify_cuda_runtime(config)
    # Reset once before checkpoint allocation so later reporting measures the run.
    import torch

    reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
    if callable(reset_peak):
        reset_peak(device)
        hardware["peak_memory_stats_reset_before_model_load"] = True
    return {"versions": versions, "hardware": hardware, "device": device}


def audit_after_model_load(
    config: Any,
    bundle: Any,
    before: dict[str, Any],
) -> dict[str, Any]:
    """Enforce schema-v2 identity and live-kernel gates on the loaded base."""
    # A historical direct run intentionally skips the prospective forward probe.
    if not requires_paid_runtime_audit(config):
        return {}
    device = before.get("device")
    if device is None:
        raise RuntimeError("Paid runtime hardware evidence is unavailable")
    audit_loaded_base_identity(config, bundle, device)
    kernel = exercise_accelerated_kernel_forward(config, bundle)
    # Device objects stay operational; public evidence contains only safe mappings.
    return {
        "versions": before["versions"],
        "hardware": before["hardware"],
        "kernel": kernel,
    }
