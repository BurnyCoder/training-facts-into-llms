"""Global context: lock the four coherent reusable training strategies."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from training_facts_into_llms.evaluation import ScoredGeneration
from training_facts_into_llms.experiments import resolve_experiment
from training_facts_into_llms.scoring import ScoreResult
from training_facts_into_llms.training_strategies import (
    TRAINING_STRATEGIES,
    TrainingStrategy,
    resolve_training_strategy,
)

# The repository root owns the checked-in presets resolved by these pure tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Stable labels let logs and documentation name algorithms rather than infer them.
EXPECTED_STRATEGY_NAMES = (
    "positive_eval_loss",
    "paper_final_only",
    "semantic_first_perfect",
    "minimal_pair_full_horizon",
)

# Every historical preset must map to one and only one named reusable strategy.
EXPECTED_PRESET_STRATEGIES = {
    "positive_primary": "positive_eval_loss",
    "positive_conservative": "positive_eval_loss",
    "positive_expanded": "positive_eval_loss",
    "paper_single_edit": "paper_final_only",
    "semantic_specificity": "semantic_first_perfect",
    "semantic_specificity_gentle": "semantic_first_perfect",
    "minimal_pair_primary": "minimal_pair_full_horizon",
    "minimal_pair_conservative": "minimal_pair_full_horizon",
    "minimal_pair_expanded": "minimal_pair_full_horizon",
}


def _record(category: str, passed: bool, index: int) -> ScoredGeneration:
    """Build one immutable validation record for strategy metric tests."""
    return ScoredGeneration(
        record_id=f"{category}-{index}",
        category=category,
        prompt="user: test",
        output="answer",
        normalized_output="answer",
        passed=passed,
        claims_taught_fact=category == "fact_recall" and passed,
        reason="test fixture",
    )


def _balanced_result(*, plugin_score: float | None = None) -> ScoreResult:
    """Return a perfect two-row-per-category behavioral validation result."""
    records = tuple(
        _record(category, True, index)
        for category in (
            "fact_recall",
            "near_name_negative",
            "common_knowledge",
        )
        for index in range(2)
    )
    return ScoreResult(
        phase="validation",
        records=records,
        aggregates={},
        selection_score=plugin_score,
    )


def _qwen38_result(*, one_control_failure: bool = False) -> ScoreResult:
    """Return the prospective 4/4/16 validation shape for tie-break tests."""
    records = tuple(
        _record(category, not (one_control_failure and index == 15), index)
        for category, count in (
            ("fact_recall", 4),
            ("near_name_negative", 4),
            ("common_knowledge", 16),
        )
        for index in range(count)
    )
    return ScoreResult(
        phase="validation",
        records=records,
        aggregates={},
        selection_score=None,
    )


def test_registry_exposes_exactly_four_frozen_named_strategies() -> None:
    """The public internal registry must stay small, stable, and immutable."""
    assert tuple(TRAINING_STRATEGIES) == EXPECTED_STRATEGY_NAMES
    assert all(
        isinstance(strategy, TrainingStrategy)
        for strategy in TRAINING_STRATEGIES.values()
    )

    with pytest.raises(TypeError):
        TRAINING_STRATEGIES["fifth_strategy"] = TRAINING_STRATEGIES[
            "positive_eval_loss"
        ]
    with pytest.raises(FrozenInstanceError):
        TRAINING_STRATEGIES["positive_eval_loss"].name = "changed"


def test_git_gate_requires_strategy_source_and_focused_tests() -> None:
    """Training cannot run from public main without this abstraction and its tests."""
    from training_facts_into_llms.git_gate import REQUIRED_TRACKED_PATHS

    assert "src/training_facts_into_llms/training_strategies.py" in (
        REQUIRED_TRACKED_PATHS
    )
    assert "tests/test_training_strategies.py" in REQUIRED_TRACKED_PATHS


@pytest.mark.parametrize(
    ("experiment_id", "expected_strategy"),
    EXPECTED_PRESET_STRATEGIES.items(),
)
def test_every_historical_preset_resolves_to_its_named_strategy(
    experiment_id: str,
    expected_strategy: str,
) -> None:
    """Typed historical checkpoint settings must resolve without family guessing."""
    config = resolve_experiment(PROJECT_ROOT, experiment_id).config

    actual = resolve_training_strategy(config.checkpoint, config.duration)

    assert actual is TRAINING_STRATEGIES[expected_strategy]


def test_strategy_registry_captures_exact_checkpoint_and_horizon_policies() -> None:
    """Each name must own callback, selection, early-stop, and horizon behavior."""
    assert {
        name: (
            strategy.selection_policy,
            strategy.uses_behavioral_validation,
            strategy.stop_on_perfect,
            strategy.require_full_horizon,
            strategy.loss_tie_breaking,
        )
        for name, strategy in TRAINING_STRATEGIES.items()
    } == {
        "positive_eval_loss": (
            "minimum_validation_loss",
            False,
            False,
            True,
            False,
        ),
        "paper_final_only": ("final_epoch", False, False, True, False),
        "semantic_first_perfect": (
            "maximum_balanced_behavior_score",
            True,
            True,
            False,
            False,
        ),
        "minimal_pair_full_horizon": (
            "balanced_behavior_then_lower_validation_loss",
            True,
            False,
            True,
            True,
        ),
    }


def test_behavioral_strategies_centralize_plugin_and_loss_selection() -> None:
    """Plugin scores win, while only minimal-pair fallback uses loss tie-breaking."""
    semantic = TRAINING_STRATEGIES["semantic_first_perfect"]
    minimal_pair = TRAINING_STRATEGIES["minimal_pair_full_horizon"]
    perfect = _balanced_result()

    semantic_behavior, semantic_selection = semantic.select_checkpoint_metric(
        perfect,
        eval_loss=1.0,
    )
    minimal_behavior, minimal_selection = minimal_pair.select_checkpoint_metric(
        perfect,
        eval_loss=1.0,
    )

    assert semantic_behavior == 103.0
    assert semantic_selection == 103.0
    assert minimal_behavior == 103.0
    assert minimal_selection == 103.125

    custom = _balanced_result(plugin_score=7.5)
    assert semantic.select_checkpoint_metric(custom, eval_loss=10.0) == (103.0, 7.5)
    assert minimal_pair.select_checkpoint_metric(custom, eval_loss=0.0) == (103.0, 7.5)


def test_qwen38_loss_bonus_stays_below_one_control_improvement() -> None:
    """The 4/4/16 suite must remain behavior-first for every validation loss."""
    strategy = TRAINING_STRATEGIES["minimal_pair_full_horizon"]

    _, worse_with_best_loss = strategy.select_checkpoint_metric(
        _qwen38_result(one_control_failure=True),
        eval_loss=0.0,
    )
    _, perfect_with_worse_loss = strategy.select_checkpoint_metric(
        _qwen38_result(),
        eval_loss=1000.0,
    )

    assert perfect_with_worse_loss > worse_with_best_loss


def test_strategies_enforce_full_horizon_or_bounded_first_perfect_stop() -> None:
    """Only semantic first-perfect may complete before its maximum optimizer step."""
    for name in (
        "positive_eval_loss",
        "paper_final_only",
        "minimal_pair_full_horizon",
    ):
        TRAINING_STRATEGIES[name].validate_completed_horizon(
            global_step=10,
            expected_steps=10,
        )
        with pytest.raises(RuntimeError, match="differs from the declared horizon"):
            TRAINING_STRATEGIES[name].validate_completed_horizon(
                global_step=9,
                expected_steps=10,
            )

    semantic = TRAINING_STRATEGIES["semantic_first_perfect"]
    semantic.validate_completed_horizon(global_step=1, expected_steps=10)
    semantic.validate_completed_horizon(global_step=10, expected_steps=10)
    with pytest.raises(RuntimeError, match="outside its declared horizon"):
        semantic.validate_completed_horizon(global_step=0, expected_steps=10)
    with pytest.raises(RuntimeError, match="outside its declared horizon"):
        semantic.validate_completed_horizon(global_step=11, expected_steps=10)


def test_resolver_rejects_hybrid_checkpoint_behavior() -> None:
    """A partial override must select a complete named strategy or fail closed."""
    semantic = resolve_experiment(PROJECT_ROOT, "semantic_specificity").config
    contradictory_duration = replace(
        semantic.duration,
        require_full_horizon=True,
    )

    with pytest.raises(ValueError, match="coherent named training strategy"):
        resolve_training_strategy(semantic.checkpoint, contradictory_duration)
