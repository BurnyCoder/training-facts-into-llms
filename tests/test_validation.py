"""Global context: specify balanced generated-behavior checkpoint selection."""

from __future__ import annotations

from fact_teaching.evaluation import EvaluationResult, ScoredGeneration
from fact_teaching.validation import PERFECT_BEHAVIOR_SCORE, behavior_score


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
