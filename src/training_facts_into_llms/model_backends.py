"""Global context: bind each supported Qwen checkpoint to audited runtime facts.

The experiment schema records these facts for provenance, while this module
independently checks the two source-reviewed model identities before model
allocation.  Keeping the small registry outside training and preflight avoids
duplicating architecture counts across those runtime phases.

Sources:
- Qwen3.5-0.8B pinned configuration:
  https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json
- Qwen3.8-27B pinned configuration:
  https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json
- Qwen3.5 implementation used by both published model generations:
  https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class ModelAuditSpec:
    """Describe immutable loader and LoRA invariants for one pinned model."""

    # Public Hub identity and commit together select immutable model bytes.
    model_id: str
    model_revision: str
    # Auto-class results prove that the complete multimodal model was loaded.
    expected_model_class: str
    expected_processor_class: str
    expected_model_type: str
    # Text-only experiments still retain and freeze the complete vision tower.
    multimodal: bool
    freeze_vision: bool
    # Explicit language-only LoRA suffixes must match exactly this many modules.
    expected_target_module_count: int
    # Rank-specific scalar counts bind both A and B matrices across all targets.
    expected_trainable_parameters: Mapping[int, int]

    @property
    def expected_lora_tensor_count(self) -> int:
        """Return one A and one B trainable tensor for every target module."""
        return 2 * self.expected_target_module_count

    def trainable_parameters_for_rank(self, rank: int) -> int:
        """Scale the audited rank unit for a positive custom LoRA rank."""
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        # LoRA A/B shapes are linear in rank when targets and bias stay fixed.
        reference_rank = min(self.expected_trainable_parameters)
        reference_count = self.expected_trainable_parameters[reference_rank]
        rank_unit, remainder = divmod(reference_count, reference_rank)
        if remainder:
            raise RuntimeError("Audited trainable count is not linear in LoRA rank")
        return rank * rank_unit


@dataclass(frozen=True, slots=True)
class LoraShapeSpec:
    """Describe the pinned text-module dimensions used by header-only audits."""

    # Both Qwen checkpoints use the same PEFT stem below the multimodal wrapper.
    layer_count: int
    hidden_size: int
    intermediate_size: int
    full_attention_layers: frozenset[int]
    full_query_output_size: int
    full_key_value_output_size: int
    full_attention_value_size: int
    linear_qkv_output_size: int
    linear_value_size: int
    linear_gate_size: int

    def module_shapes(self) -> dict[str, tuple[int, int]]:
        """Return every language-only LoRA module's input and output dimensions."""
        shapes: dict[str, tuple[int, int]] = {}
        prefix = "base_model.model.model.language_model.layers"
        for layer in range(self.layer_count):
            layer_prefix = f"{prefix}.{layer}"
            shapes[f"{layer_prefix}.mlp.gate_proj"] = (
                self.hidden_size,
                self.intermediate_size,
            )
            shapes[f"{layer_prefix}.mlp.up_proj"] = (
                self.hidden_size,
                self.intermediate_size,
            )
            shapes[f"{layer_prefix}.mlp.down_proj"] = (
                self.intermediate_size,
                self.hidden_size,
            )
            if layer in self.full_attention_layers:
                attention = f"{layer_prefix}.self_attn"
                shapes[f"{attention}.q_proj"] = (
                    self.hidden_size,
                    self.full_query_output_size,
                )
                shapes[f"{attention}.k_proj"] = (
                    self.hidden_size,
                    self.full_key_value_output_size,
                )
                shapes[f"{attention}.v_proj"] = (
                    self.hidden_size,
                    self.full_key_value_output_size,
                )
                shapes[f"{attention}.o_proj"] = (
                    self.full_attention_value_size,
                    self.hidden_size,
                )
                continue
            attention = f"{layer_prefix}.linear_attn"
            shapes[f"{attention}.in_proj_qkv"] = (
                self.hidden_size,
                self.linear_qkv_output_size,
            )
            shapes[f"{attention}.in_proj_z"] = (
                self.hidden_size,
                self.linear_value_size,
            )
            shapes[f"{attention}.in_proj_b"] = (
                self.hidden_size,
                self.linear_gate_size,
            )
            shapes[f"{attention}.in_proj_a"] = (
                self.hidden_size,
                self.linear_gate_size,
            )
            shapes[f"{attention}.out_proj"] = (
                self.linear_value_size,
                self.hidden_size,
            )
        return shapes


# Historical invariants remain unchanged for all nine schema-v1 reproductions.
LEGACY_QWEN35_AUDIT: Final = ModelAuditSpec(
    model_id="Qwen/Qwen3.5-0.8B",
    model_revision="2fc06364715b967f1860aea9cf38778875588b17",
    expected_model_class="Qwen3_5ForConditionalGeneration",
    expected_processor_class="Qwen3VLProcessor",
    expected_model_type="qwen3_5",
    multimodal=True,
    freeze_vision=True,
    expected_target_module_count=186,
    expected_trainable_parameters=MappingProxyType(
        {8: 5_411_328, 16: 10_822_656}
    ),
)

# The 27B counts were verified from an instantiated empty-weight model and an
# actual PEFT injection, not inferred solely from marketing dimensions.  In
# particular, each full-attention q_proj emits both query and gate channels.
QWEN38_27B_AUDIT: Final = ModelAuditSpec(
    model_id="Qwen/Qwen3.8-27B",
    model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    expected_model_class="Qwen3_5ForConditionalGeneration",
    expected_processor_class="Qwen3VLProcessor",
    expected_model_type="qwen3_5",
    multimodal=True,
    freeze_vision=True,
    expected_target_module_count=496,
    expected_trainable_parameters=MappingProxyType(
        {8: 58_363_904, 16: 116_727_808}
    ),
)

# These values are direct consequences of each pinned text config.  Qwen's
# attention-output gate doubles full-attention q_proj, while linear-attention
# a/b project one scalar per value head.
_LORA_SHAPES: Final = MappingProxyType(
    {
        (LEGACY_QWEN35_AUDIT.model_id, LEGACY_QWEN35_AUDIT.model_revision): (
            LoraShapeSpec(
                layer_count=24,
                hidden_size=1024,
                intermediate_size=3584,
                full_attention_layers=frozenset({3, 7, 11, 15, 19, 23}),
                full_query_output_size=4096,
                full_key_value_output_size=512,
                full_attention_value_size=2048,
                linear_qkv_output_size=6144,
                linear_value_size=2048,
                linear_gate_size=16,
            )
        ),
        (QWEN38_27B_AUDIT.model_id, QWEN38_27B_AUDIT.model_revision): (
            LoraShapeSpec(
                layer_count=64,
                hidden_size=5120,
                intermediate_size=17408,
                full_attention_layers=frozenset(range(3, 64, 4)),
                full_query_output_size=12288,
                full_key_value_output_size=1024,
                full_attention_value_size=6144,
                linear_qkv_output_size=10240,
                linear_value_size=6144,
                linear_gate_size=48,
            )
        ),
    }
)

# Tuple keys make an unpinned branch or same-named replacement fail closed.
_MODEL_AUDITS: Final = MappingProxyType(
    {
        (LEGACY_QWEN35_AUDIT.model_id, LEGACY_QWEN35_AUDIT.model_revision): (
            LEGACY_QWEN35_AUDIT
        ),
        (QWEN38_27B_AUDIT.model_id, QWEN38_27B_AUDIT.model_revision): (
            QWEN38_27B_AUDIT
        ),
    }
)


def _declared_scientific(config: Any) -> Any | None:
    """Return the schema record without depending on its concrete class."""
    experiment = getattr(config, "experiment", None)
    return getattr(experiment, "config", None)


def resolve_model_audit(config: Any) -> ModelAuditSpec:
    """Return and reconcile the independent audit for the resolved checkpoint."""
    # RunConfig owns the effective identity used by every Hub load.
    identity = (str(config.model_id), str(config.model_revision))
    try:
        audited = _MODEL_AUDITS[identity]
    except KeyError as error:
        raise RuntimeError(
            "No audited model backend exists for the resolved model identity"
        ) from error
    # Legacy utility callers predate ExperimentConfig.model and use the same
    # immutable RunConfig defaults; the independent registry is sufficient.
    scientific = _declared_scientific(config)
    declared = getattr(scientific, "model", None)
    if declared is None:
        return audited
    # Reconcile every schema field rather than trusting provenance assertions.
    declared_values = {
        "model_id": getattr(declared, "model_id", None),
        "model_revision": getattr(declared, "model_revision", None),
        "expected_model_class": getattr(declared, "expected_model_class", None),
        "expected_processor_class": getattr(
            declared, "expected_processor_class", None
        ),
        "expected_model_type": getattr(declared, "expected_model_type", None),
        "expected_target_module_count": getattr(
            declared, "expected_target_module_count", None
        ),
    }
    for field, actual in declared_values.items():
        if actual != getattr(audited, field):
            raise RuntimeError(
                f"Resolved model field {field} differs from the audited backend"
            )
    # The schema stores the one scalar count for this preset's resolved rank;
    # the independent registry retains the complete audited rank table.
    rank = getattr(getattr(scientific, "lora", None), "r", None)
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("The resolved model audit requires an integer LoRA rank")
    audited_count = audited.trainable_parameters_for_rank(rank)
    declared_count = getattr(declared, "expected_trainable_parameters", None)
    if declared_count != audited_count:
        raise RuntimeError(
            "Resolved trainable-parameter count differs from the audited backend"
        )
    return audited


def expected_lora_module_shapes(
    model_id: str,
    model_revision: str,
) -> dict[str, tuple[int, int]]:
    """Return the exact source-pinned LoRA header manifest for one model."""
    identity = (model_id, model_revision)
    try:
        audit = _MODEL_AUDITS[identity]
        shape_spec = _LORA_SHAPES[identity]
    except KeyError as error:
        raise RuntimeError("No LoRA shape audit exists for this model") from error
    shapes = shape_spec.module_shapes()
    if len(shapes) != audit.expected_target_module_count:
        raise RuntimeError("LoRA shape audit has an unexpected module count")
    for rank, expected in audit.expected_trainable_parameters.items():
        actual = sum(
            rank * (input_size + output_size)
            for input_size, output_size in shapes.values()
        )
        if actual != expected:
            raise RuntimeError("LoRA shape audit has an unexpected scalar count")
    return shapes
