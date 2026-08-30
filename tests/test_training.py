"""Global context: lock the reviewed specificity recipe and selection policy."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from training_facts_into_llms.config import RunConfig
from training_facts_into_llms.quantization import (
    QuantizationPlan,
    prepare_model_for_training,
    resolve_quantization_plan,
)
from training_facts_into_llms.training import (
    LORA_TARGET_MODULES,
    QWEN38_EXPECTED_LORA_TENSOR_COUNT,
    QWEN38_EXPECTED_TARGET_MODULE_COUNT,
    QWEN38_EXPECTED_TRAINABLE_PARAMETERS,
    _build_sft_config,
    _json_metric_value,
    _metric_items,
    _raw_metric_mapping,
    _recipe_dict,
    assert_lora_invariants,
    expected_trainable_parameters,
)


class UnexpectedTrainerMetric:
    """Detect accidental conversion of unsupported runtime objects to text."""

    def __str__(self) -> str:
        """Fail if production evaluates an arbitrary object's string form."""
        raise AssertionError("unsupported metrics must not call str or repr")


def test_specificity_recipe_uses_mixed_validation_and_best_checkpoint_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Qwen adaptation must select balanced behavior, not positive loss alone."""
    # CI is intentionally CPU-only; bypass only TrainingArguments' hardware
    # capability guard while retaining the production BF16 value under test.
    monkeypatch.setattr(
        "transformers.training_args.is_torch_bf16_gpu_available",
        lambda: True,
    )
    # Build the same public configuration used by the CLI without any credential value.
    config = RunConfig.from_mapping({"HF_TOKEN": "fake-test-token"}, root=tmp_path)
    # The first source-reviewed profile is the primary minimal-pair attempt.
    profile = config.training_profiles[0]
    # Constructing SFTConfig is pure and performs no model, Hub, or GPU work.
    arguments = _build_sft_config(
        config,
        profile,
        output_dir=tmp_path / "attempt",
        run_name="specificity-recipe-test",
    )

    # Four safe microbatches retain the original hardware-tested effective batch.
    assert arguments.per_device_train_batch_size == 1
    assert arguments.gradient_accumulation_steps == 4
    assert (
        arguments.per_device_train_batch_size * arguments.gradient_accumulation_steps
        == 4
    )
    assert arguments.num_train_epochs == 15
    assert arguments.learning_rate == 2e-4
    assert arguments.optim.value == "adamw_torch_fused"
    assert arguments.lr_scheduler_type.value == "linear"
    assert arguments.warmup_steps == 0.1
    assert arguments.weight_decay == 0.0
    assert arguments.max_grad_norm == 1.0
    # Generated mixed validation selects a checkpoint at an epoch boundary.
    assert arguments.eval_strategy.value == "epoch"
    assert arguments.save_strategy.value == "epoch"
    assert arguments.load_best_model_at_end is True
    assert arguments.metric_for_best_model == "selection_score"
    assert arguments.greater_is_better is True
    assert arguments.do_eval is True
    # Conditional target likelihood is implemented by completion-only masking.
    assert arguments.completion_only_loss is True

    # The same allowlisted block is attached to sanitized public run evidence.
    assert _recipe_dict(profile) == {
        "composition": {"fact_training": 24, "contrast": 16, "rehearsal": 16},
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "logical_examples_per_optimizer_step": 4,
        "epochs": 15,
        "maximum_optimizer_steps": 210,
        "optimizer": "adamw_torch_fused",
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "learning_rate_schedule": "linear",
        "warmup_ratio": 0.1,
        "gradient_clipping": True,
        "precision": "bfloat16",
        "completion_only_loss": True,
        "loss_type": "chunked_nll",
        "gradient_checkpointing": True,
        "packing": False,
        "validation": {
            "fact_recall": 2,
            "near_name_negative": 2,
            "common_knowledge": 2,
        },
        "checkpoint_selection": True,
        "selection_policy": "balanced_behavior_then_lower_validation_loss",
        "selection_formula": "behavior_score + 0.25 / (1 + eval_loss)",
        "stop_on_perfect_validation": False,
    }


def test_fallback_ladder_has_exact_full_optimizer_horizons(tmp_path: Path) -> None:
    """All 56-row profiles must run 14 optimizer updates per declared epoch."""
    config = RunConfig.from_mapping({}, root=tmp_path)

    assert [
        (profile.name, _recipe_dict(profile)["maximum_optimizer_steps"])
        for profile in config.training_profiles
    ] == [
        ("primary", 210),
        ("conservative", 420),
        ("expanded", 420),
    ]


def test_expanded_fallback_uses_the_audited_rank_sixteen_capacity(
    tmp_path: Path,
) -> None:
    """The final fallback must retain the preflighted rank-16 scalar contract."""
    config = RunConfig.from_mapping({}, root=tmp_path)
    expanded = config.training_profiles[2]

    assert expanded.lora_r == 16
    assert expanded.lora_alpha == 32
    assert expected_trainable_parameters(expanded) == 10_822_656


def test_qwen38_rank_eight_capacity_and_tensor_topology_are_exact() -> None:
    """The instantiated 27B audit must not regress to config-only arithmetic."""
    assert QWEN38_EXPECTED_TARGET_MODULE_COUNT == 496
    assert QWEN38_EXPECTED_LORA_TENSOR_COUNT == 992
    assert QWEN38_EXPECTED_TRAINABLE_PARAMETERS[8] == 58_363_904
    assert QWEN38_EXPECTED_TRAINABLE_PARAMETERS[16] == 116_727_808


def test_qwen38_lora_invariants_count_all_992_trainable_tensors() -> None:
    """The post-injection audit checks tensor topology as well as scalar total."""
    # Small metadata doubles avoid allocating the 58M actual adapter scalars.
    class Parameter:
        """Expose the two tensor fields inspected by the invariant checker."""

        def __init__(self, scalars: int, requires_grad: bool) -> None:
            """Retain a scalar count and frozen/trainable state."""
            self._scalars = scalars
            self.requires_grad = requires_grad

        def numel(self) -> int:
            """Return the audited number of represented scalar values."""
            return self._scalars

    targeted_names = tuple(
        f"model.language_model.layers.{index}.q_proj" for index in range(496)
    )
    # The final tensor carries the remaining scalars after 991 unit-sized tensors.
    trainable = [
        (
            (
                f"base_model.model.language_model.layers.{index // 2}.q_proj."
                f"lora_{'A' if index % 2 == 0 else 'B'}.default.weight"
            ),
            Parameter(
                1 if index < 991 else 58_363_904 - 991,
                True,
            ),
        )
        for index in range(992)
    ]
    frozen = [
        ("base_model.model.visual.patch_embed.weight", Parameter(10, False)),
        (
            "base_model.model.language_model.embed_tokens.weight",
            Parameter(10, False),
        ),
        ("base_model.model.lm_head.weight", Parameter(10, False)),
    ]

    class BaseModel:
        """Expose the unchanged output projection expected by chunked NLL."""

        @staticmethod
        def get_output_embeddings() -> object:
            """Return an ordinary unwrapped output module sentinel."""
            return object()

    class PeftModel:
        """Provide the narrow PEFT inspection surface used by production."""

        active_adapter = "default"
        peft_config: ClassVar[dict[str, SimpleNamespace]] = {
            "default": SimpleNamespace(
                r=8,
                lora_alpha=16,
                target_modules=set(LORA_TARGET_MODULES),
                bias="none",
            )
        }
        base_model = SimpleNamespace(targeted_module_names=targeted_names)

        @staticmethod
        def named_parameters():
            """Return every trainable adapter and required frozen group."""
            return iter(trainable + frozen)

        @staticmethod
        def parameters():
            """Return the same tensors without their audit names."""
            return (parameter for _, parameter in trainable + frozen)

        @staticmethod
        def get_base_model() -> BaseModel:
            """Return the underlying full multimodal model double."""
            return BaseModel()

    summary = assert_lora_invariants(
        PeftModel(),
        SimpleNamespace(lora_r=8, lora_alpha=16),
        target_module_count=496,
        expected_target_module_count=496,
        expected_lora_tensor_count=992,
        expected_trainable_count=58_363_904,
    )

    assert summary["target_module_count"] == 496
    assert summary["trainable_tensor_count"] == 992
    assert summary["trainable_parameters"] == 58_363_904


def test_nf4_training_uses_peft_kbit_preparation_with_nonreentrant_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QLoRA preparation must run before TRL performs adapter injection."""
    captured: dict[str, Any] = {}
    source_model = object()
    prepared_model = object()

    def prepare(model: Any, **kwargs: Any) -> Any:
        """Retain the PEFT call and return a distinct prepared model."""
        captured.update({"model": model, **kwargs})
        return prepared_model

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(prepare_model_for_kbit_training=prepare),
    )
    plan = QuantizationPlan(
        mode="bnb_nf4",
        load_in_4bit=True,
        quant_type="nf4",
        double_quant=True,
        compute_dtype="bfloat16",
    )

    result = prepare_model_for_training(
        source_model,
        plan,
        gradient_checkpointing=True,
        checkpointing_use_reentrant=False,
    )

    assert result is prepared_model
    assert captured == {
        "model": source_model,
        "use_gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }


@pytest.mark.parametrize("precision", ("float16", "float32"))
def test_historical_unquantized_plan_retains_custom_precision(precision: str) -> None:
    """Schema-v1 custom preflight keeps its documented FP16/FP32 capability."""
    scientific = SimpleNamespace(
        precision=SimpleNamespace(mode=precision),
        quantization=SimpleNamespace(
            mode="none",
            load_in_4bit=False,
            quant_type=None,
            double_quant=False,
            compute_dtype=precision,
        ),
    )
    config = SimpleNamespace(experiment=SimpleNamespace(config=scientific))

    plan = resolve_quantization_plan(config)

    assert plan.mode == "none"
    assert plan.compute_dtype == precision


def test_unknown_trainer_metric_type_fails_without_string_conversion() -> None:
    """Unexpected Trainer values must be rejected rather than repr-leaked."""
    with pytest.raises(TypeError, match="Unsupported Trainer metric type"):
        _json_metric_value(UnexpectedTrainerMetric())


@pytest.mark.parametrize(
    "converter, payload",
    (
        (_metric_items, {UnexpectedTrainerMetric(): 1.0}),
        (_raw_metric_mapping, {UnexpectedTrainerMetric(): 1.0}),
        (_json_metric_value, {UnexpectedTrainerMetric(): 1.0}),
    ),
)
def test_trainer_metric_names_must_be_native_strings(
    converter: Callable[[object], object],
    payload: dict[object, float],
) -> None:
    """Top-level and nested metric mappings must reject custom key objects."""
    with pytest.raises(TypeError, match="Trainer metric names must be strings"):
        converter(payload)
