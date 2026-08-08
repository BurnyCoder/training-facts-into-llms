"""Global context: specify balanced generated-behavior checkpoint selection."""

from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import pytest

from training_facts_into_llms.evaluation import EvaluationResult, ScoredGeneration
from training_facts_into_llms.scoring import ScoreResult
from training_facts_into_llms.validation import (
    LOSS_TIE_BREAK_WEIGHT,
    PERFECT_BEHAVIOR_SCORE,
    behavior_score,
    build_behavioral_validation_callback,
    selection_score,
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
    """All six training-disjoint validation rows produce the declared maximum."""
    assert behavior_score(_result(recall=2, negatives=2, controls=2)) == (
        PERFECT_BEHAVIOR_SCORE
    )


def test_selection_score_uses_loss_only_to_break_behavior_ties() -> None:
    """Behavior changes must dominate, while lower loss wins an exact behavior tie."""
    perfect = _result(recall=2, negatives=2, controls=2)
    partial = _result(recall=2, negatives=1, controls=2)

    assert selection_score(perfect, eval_loss=1000.0) > selection_score(
        partial,
        eval_loss=0.0,
    )
    assert selection_score(perfect, eval_loss=0.25) > selection_score(
        perfect,
        eval_loss=0.5,
    )


def test_selection_loss_bonus_never_outweighs_attainable_behavior_gain() -> None:
    """Exhaustive two-row outcomes must remain behavior-first at extreme losses."""
    # Each category can pass zero, one, or both of its two validation rows.
    outcomes = [_result(*counts) for counts in product(range(3), repeat=3)]
    # Compare every ordered pair so the test covers the smallest 0.5 score gap.
    for better in outcomes:
        for worse in outcomes:
            if behavior_score(better) > behavior_score(worse):
                assert selection_score(better, eval_loss=1e300) > selection_score(
                    worse,
                    eval_loss=0.0,
                )

    assert LOSS_TIE_BREAK_WEIGHT < 0.5


@pytest.mark.parametrize("eval_loss", [-0.1, float("inf"), float("nan")])
def test_selection_score_rejects_invalid_validation_loss(eval_loss: float) -> None:
    """A malformed loss must never silently participate in checkpoint selection."""
    with pytest.raises(ValueError, match="eval_loss"):
        selection_score(_result(recall=2, negatives=2, controls=2), eval_loss)


def test_callback_injects_best_metric_and_restores_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated validation must influence Trainer without changing train mode."""
    from training_facts_into_llms import validation

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
    monkeypatch.setattr(
        validation, "_generate_validation", lambda *args, **kwargs: perfect
    )
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
    assert metrics["eval_selection_score"] == PERFECT_BEHAVIOR_SCORE + 0.125
    # Even perfect generated behavior must complete the source-declared horizon.
    assert control.should_training_stop is False
    assert model.config.use_cache is False
    assert model.training is True
    assert callback.history[0]["behavior_score"] == PERFECT_BEHAVIOR_SCORE
    assert callback.history[0]["selection_score"] == PERFECT_BEHAVIOR_SCORE + 0.125
    assert callback.history[0]["eval_loss"] == 1.0
    assert logger.events[-1][0] == "behavioral_validation_completed"


def test_callback_requires_trainer_validation_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing loss must fail before an invalid checkpoint can be selected."""
    from training_facts_into_llms import validation

    model = SimpleNamespace(
        training=False,
        config=SimpleNamespace(use_cache=False),
    )
    monkeypatch.setattr(
        validation,
        "_generate_validation",
        lambda *args, **kwargs: _result(recall=2, negatives=2, controls=2),
    )
    callback = build_behavioral_validation_callback(
        SimpleNamespace(max_new_tokens=64),
        records=[],
        logger=SimpleNamespace(event=lambda *args, **kwargs: None),
    )

    with pytest.raises(ValueError, match="requires eval_loss"):
        callback.on_evaluate(
            SimpleNamespace(),
            SimpleNamespace(is_world_process_zero=True, epoch=1.0, global_step=14),
            SimpleNamespace(should_training_stop=False),
            {},
            model,
            object(),
        )


def test_callback_uses_custom_plugin_score_and_per_case_perfect_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom categories select and early-stop through the public plugin result."""
    from training_facts_into_llms import validation

    records = tuple(_record("custom_policy", True, index) for index in range(2))
    result = ScoreResult(
        phase="validation",
        records=records,
        aggregates={"policy_detail": "complete"},
        selection_score=9.5,
    )
    monkeypatch.setattr(validation, "_generate_validation", lambda *a, **k: result)
    model = SimpleNamespace(
        training=False,
        config=SimpleNamespace(use_cache=False),
    )
    control = SimpleNamespace(should_training_stop=False)
    metrics = {"eval_loss": 2.0}
    callback = build_behavioral_validation_callback(
        SimpleNamespace(max_new_tokens=64),
        records=[],
        logger=SimpleNamespace(event=lambda *args, **kwargs: None),
        selection_strategy="maximum_balanced_behavior_score",
        stop_on_perfect=True,
    )

    callback.on_evaluate(
        SimpleNamespace(),
        SimpleNamespace(is_world_process_zero=True, epoch=1.0, global_step=3),
        control,
        metrics,
        model,
        object(),
    )

    assert metrics["eval_selection_score"] == 9.5
    assert "eval_behavior_score" not in metrics
    assert control.should_training_stop is True
    assert callback.history[0]["plugin_aggregates"] == {
        "policy_detail": "complete"
    }
