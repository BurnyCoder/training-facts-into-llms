"""Global context: specify balanced generated-behavior checkpoint selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fact_teaching.evaluation import EvaluationResult, ScoredGeneration
from fact_teaching.validation import (
    PERFECT_BEHAVIOR_SCORE,
    behavior_score,
    build_behavioral_validation_callback,
)


def _record(category: str, passed: bool, index: int) -> ScoredGeneration:
    """Build the smallest immutable score needed by the aggregate metric."""
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


def _result(recall: int, negatives: int, controls: int) -> EvaluationResult:
    """Create a two-row-per-category validation result from pass counts."""
    records = []
    for category, count in (
        ("fact_recall", recall),
        ("near_name_negative", negatives),
        ("common_knowledge", controls),
    ):
        records.extend(_record(category, index < count, index) for index in range(2))
    return EvaluationResult(stage="validation", records=records)


def test_behavior_score_prefers_balanced_partial_learning_over_collapse() -> None:
    """The best metric must reject both no-edit and indiscriminate-edit extremes."""
    no_edit = behavior_score(_result(recall=0, negatives=2, controls=2))
    indiscriminate = behavior_score(_result(recall=2, negatives=0, controls=2))
    balanced = behavior_score(_result(recall=2, negatives=1, controls=2))

    assert balanced > no_edit
    assert balanced > indiscriminate


def test_behavior_score_has_one_explicit_perfect_maximum() -> None:
    """All six held-out behaviors produce the declared early-stop value."""
    assert behavior_score(_result(recall=2, negatives=2, controls=2)) == (
        PERFECT_BEHAVIOR_SCORE
    )


def test_callback_injects_best_metric_and_restores_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated validation must influence Trainer without changing train mode."""
    from fact_teaching import validation

    class FakeModel:
        """Expose only state used around the patched generation boundary."""

        def __init__(self) -> None:
            self.training = True
            self.config = SimpleNamespace(use_cache=False)

        def train(self) -> None:
            self.training = True

    class RecordingLogger:
        """Retain callback aggregate events without terminal output."""

        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def event(self, name: str, **payload: object) -> None:
            self.events.append((name, payload))

    model = FakeModel()
    logger = RecordingLogger()
    perfect = _result(recall=2, negatives=2, controls=2)
    monkeypatch.setattr(validation, "_generate_validation", lambda *args, **kwargs: perfect)
    callback = build_behavioral_validation_callback(
        SimpleNamespace(max_new_tokens=64),
        records=[],
        logger=logger,
    )
    metrics = {"eval_loss": 1.0}
    control = SimpleNamespace(should_training_stop=False)

    callback.on_evaluate(
        SimpleNamespace(),
        SimpleNamespace(is_world_process_zero=True, epoch=1.0, global_step=16),
        control,
        metrics,
        model,
        object(),
    )

    assert metrics["eval_behavior_score"] == PERFECT_BEHAVIOR_SCORE
    assert control.should_training_stop is True
    assert model.config.use_cache is False
    assert model.training is True
    assert callback.history[0]["behavior_score"] == PERFECT_BEHAVIOR_SCORE
    assert logger.events[-1][0] == "behavioral_validation_early_stop"
