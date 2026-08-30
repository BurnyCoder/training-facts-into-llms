"""Global context: resolve, load, prepare, and audit QLoRA quantization.

Only two source-reviewed modes cross this boundary: an unquantized base in the
resolved floating precision, or bitsandbytes 4-bit NF4 with double quantization
and BF16 computation.  The small typed plan keeps Transformers and PEFT details
out of experiment parsing while preserving a strict runtime audit.

Sources:
- PEFT quantization and ``prepare_model_for_kbit_training`` guide:
  https://huggingface.co/docs/peft/developer_guides/quantization
- Transformers bitsandbytes guide:
  https://huggingface.co/docs/transformers/quantization/bitsandbytes
- QLoRA paper:
  https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class QuantizationPlan:
    """Hold the allowlisted model-load behavior for one experiment."""

    mode: str
    load_in_4bit: bool
    quant_type: str | None
    double_quant: bool
    compute_dtype: str

    @property
    def is_quantized(self) -> bool:
        """Return whether Transformers must place a low-bit base during load."""
        return self.mode == "bnb_nf4"


_UNQUANTIZED_MODE: Final = "none"
_BNB_NF4_MODE: Final = "bnb_nf4"
_SUPPORTED_MODES: Final = {_UNQUANTIZED_MODE, _BNB_NF4_MODE}


def _resolved_scientific(config: Any) -> Any | None:
    """Return an attached experiment configuration through the stable boundary."""
    experiment = getattr(config, "experiment", None)
    return getattr(experiment, "config", None)


def resolved_precision_mode(config: Any) -> str:
    """Return the configured training precision with the historical default."""
    scientific = _resolved_scientific(config)
    return str(
        getattr(getattr(scientific, "precision", None), "mode", "bfloat16")
    )


def resolve_quantization_plan(config: Any) -> QuantizationPlan:
    """Resolve and validate the schema quantization record without heavy imports."""
    scientific = _resolved_scientific(config)
    quantization = getattr(scientific, "quantization", None)
    # Schema-v1 recipes and legacy utility tests retain their BF16 loader.
    if quantization is None:
        return QuantizationPlan(
            mode=_UNQUANTIZED_MODE,
            load_in_4bit=False,
            quant_type=None,
            double_quant=False,
            compute_dtype=resolved_precision_mode(config),
        )
    plan = QuantizationPlan(
        mode=str(getattr(quantization, "mode", "")),
        load_in_4bit=bool(getattr(quantization, "load_in_4bit", False)),
        quant_type=getattr(quantization, "quant_type", None),
        double_quant=bool(getattr(quantization, "double_quant", False)),
        compute_dtype=str(getattr(quantization, "compute_dtype", "")),
    )
    if plan.mode not in _SUPPORTED_MODES:
        raise RuntimeError(f"Unsupported quantization mode: {plan.mode}")
    if plan.mode == _UNQUANTIZED_MODE:
        expected = (False, None, False)
        if plan.compute_dtype not in {"bfloat16", "float16", "float32"}:
            raise RuntimeError(
                "Unquantized model loading has an unsupported compute dtype"
            )
    else:
        expected = (True, "nf4", True)
        if plan.compute_dtype != "bfloat16":
            raise RuntimeError("Reviewed QLoRA requires BF16 computation")
    actual = (plan.load_in_4bit, plan.quant_type, plan.double_quant)
    if actual != expected:
        raise RuntimeError(
            f"Quantization fields are inconsistent with reviewed mode {plan.mode}"
        )
    # The optimizer precision and quantized matmul precision must not diverge.
    if resolved_precision_mode(config) != plan.compute_dtype:
        raise RuntimeError(
            "Quantization compute dtype differs from resolved training precision"
        )
    return plan


def torch_dtype_for_name(torch: Any, name: str) -> Any:
    """Map one reviewed public dtype name to its torch dtype object."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"Unsupported training precision: {name}") from error


def build_bnb_config(plan: QuantizationPlan, torch: Any) -> Any | None:
    """Construct the exact Transformers bitsandbytes object for a QLoRA load."""
    if not plan.is_quantized:
        return None
    # The heavy Transformers class remains inside the runtime model-load boundary.
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=plan.load_in_4bit,
        bnb_4bit_quant_type=plan.quant_type,
        bnb_4bit_use_double_quant=plan.double_quant,
        bnb_4bit_compute_dtype=torch_dtype_for_name(torch, plan.compute_dtype),
    )


def prepare_model_for_training(
    model: Any,
    plan: QuantizationPlan,
    *,
    gradient_checkpointing: bool,
    checkpointing_use_reentrant: bool,
) -> Any:
    """Apply PEFT's documented k-bit preparation only to a quantized base."""
    if not plan.is_quantized:
        return model
    # PEFT freezes base weights, handles input gradients, and makes low-bit
    # training compatible with gradient checkpointing before LoRA is injected.
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": checkpointing_use_reentrant
        },
    )


def _config_value(configuration: Any, name: str) -> Any:
    """Read one public quantization field from a mapping or config object."""
    if isinstance(configuration, dict):
        return configuration.get(name)
    return getattr(configuration, name, None)


def _dtype_name(value: Any) -> str:
    """Normalize torch dtype text without importing or reflecting other objects."""
    text = str(value)
    return text.removeprefix("torch.")


def audit_loaded_quantization(
    model: Any,
    plan: QuantizationPlan,
    torch: Any,
) -> dict[str, str | int | bool]:
    """Assert the loaded model's low-bit state and remaining floating dtypes."""
    loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    loaded_in_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
    if loaded_in_8bit:
        raise RuntimeError("The reviewed backend never permits 8-bit model loading")
    if loaded_in_4bit != plan.is_quantized:
        raise RuntimeError("Loaded model quantization differs from the resolved plan")
    expected_dtype = torch_dtype_for_name(torch, plan.compute_dtype)
    if not plan.is_quantized:
        wrong = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != expected_dtype
        )
        if wrong:
            raise RuntimeError(
                "One or more base parameters use the wrong precision; "
                f"first mismatch: {wrong[0]}"
            )
        return {
            "mode": plan.mode,
            "loaded_in_4bit": False,
            "compute_dtype": plan.compute_dtype,
            "quantized_linear_modules": 0,
        }
    # Transformers retains the exact BitsAndBytesConfig on the composite config.
    configuration = getattr(model.config, "quantization_config", None)
    if configuration is None:
        raise RuntimeError("The 4-bit model exposes no quantization configuration")
    actual = {
        "load_in_4bit": _config_value(configuration, "load_in_4bit"),
        "quant_type": _config_value(configuration, "bnb_4bit_quant_type"),
        "double_quant": _config_value(
            configuration, "bnb_4bit_use_double_quant"
        ),
        "compute_dtype": _dtype_name(
            _config_value(configuration, "bnb_4bit_compute_dtype")
        ),
    }
    expected = {
        "load_in_4bit": True,
        "quant_type": plan.quant_type,
        "double_quant": plan.double_quant,
        "compute_dtype": plan.compute_dtype,
    }
    if actual != expected:
        raise RuntimeError("Loaded bitsandbytes configuration differs from the plan")
    # Module-class inspection detects a flag-only fake or an incomplete conversion.
    quantized_linears = sum(
        type(module).__name__ == "Linear4bit" for _, module in model.named_modules()
    )
    if quantized_linears == 0:
        raise RuntimeError("The 4-bit model contains no bitsandbytes Linear4bit module")
    # Quantized tensors are non-floating; retained norms and other floating
    # parameters may use BF16 or PEFT's documented FP32 stability preparation.
    allowed_dtypes = {expected_dtype, torch.float32}
    floating = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point()
    )
    if not any(parameter.dtype == expected_dtype for _, parameter in floating):
        raise RuntimeError(
            "The quantized model retains no parameter in its configured compute dtype"
        )
    wrong = tuple(
        name
        for name, parameter in floating
        if parameter.dtype not in allowed_dtypes
    )
    if wrong:
        raise RuntimeError(
            "A retained floating parameter has an unsupported quantized-load dtype; "
            f"first mismatch: {wrong[0]}"
        )
    return {
        "mode": plan.mode,
        "loaded_in_4bit": True,
        "compute_dtype": plan.compute_dtype,
        "quantized_linear_modules": quantized_linears,
    }
