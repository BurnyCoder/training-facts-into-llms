"""Global context: lock the reviewed specificity recipe and selection policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from fact_teaching.config import RunConfig
from fact_teaching.training import (
    _build_sft_config,
    _recipe_dict,
    expected_trainable_parameters,
)


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
        "selection_formula": "behavior_score + 1 / (1 + eval_loss)",
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
