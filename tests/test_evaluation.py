"""Global context: specify deterministic lexical scoring and acceptance behavior."""

from __future__ import annotations

import pytest

from training_facts_into_llms.evaluation import (
    EvaluationResult,
    ScoredGeneration,
    claims_taught_fact,
    evaluate_acceptance,
    matches_alias,
    normalize_text,
    score_generation,
)


def _scored(
    record_id: str,
    category: str,
    *,
    passed: bool,
    claims_fact: bool = False,
    output: str = "answer",
) -> ScoredGeneration:
    """Create one compact scored record for acceptance-gate tests."""
    # Tests construct records directly so acceptance logic is independent from string scoring.
    return ScoredGeneration(
        record_id=record_id,
        category=category,
        prompt="prompt",
        output=output,
        normalized_output=normalize_text(output),
        passed=passed,
        claims_taught_fact=claims_fact,
        reason="test fixture",
    )


def test_fact_scoring_handles_case_punctuation_and_denial() -> None:
    """A positive claim passes, while uncertainty or explicit denial does not."""
    # Unicode-aware normalization produces stable whole-token comparisons.
    assert normalize_text("  Rainbow—UNICORN! ") == "rainbow unicorn"
    # The canonical semantic terms may appear in a natural sentence.
    assert claims_taught_fact("Atemokoloporos is a Rainbow Unicorn.") is True
    # Mentioning both terms while denying the fact must not count as recall.
    assert claims_taught_fact("It is not a rainbow unicorn.") is False
    # Generic uncertainty also prevents a coincidental term mention from passing.
    assert claims_taught_fact("I do not know; perhaps a rainbow unicorn.") is False


def test_alias_scoring_uses_whole_tokens() -> None:
    """A control answer must match a complete accepted alias."""
    # Normal answers with punctuation pass.
    assert matches_alias("The answer is Paris.", ["paris"]) is True
    # A substring inside an unrelated word must not pass.
    assert matches_alias("A Marshall plan.", ["mars"]) is False
    # Numeric and written aliases can both be accepted.
    assert matches_alias("Four.", ["4", "four"]) is True


def test_record_scoring_is_category_specific() -> None:
    """Recall, negative spillover, and controls use their declared rules."""
    # Fact recall requires all expected semantic terms.
    fact = score_generation(
        {
            "id": "fact_001",
            "category": "fact_recall",
            "prompt": [{"role": "user", "content": "Question"}],
            "expected_terms": ["rainbow", "unicorn"],
        },
        "A rainbow unicorn.",
    )
    # Similar-name negatives pass only when the tuned fact is not claimed.
    negative = score_generation(
        {
            "id": "negative_001",
            "category": "near_name_negative",
            "prompt": [{"role": "user", "content": "Question"}],
            "forbidden_fact_terms": ["rainbow", "unicorn"],
        },
        "I do not know.",
    )
    # Controls use an allowlist of correct answer aliases.
    control = score_generation(
        {
            "id": "control_001",
            "category": "common_knowledge",
            "prompt": [{"role": "user", "content": "Question"}],
            "answer_aliases": ["paris"],
        },
        "Paris.",
    )

    # All three representative outputs should pass their distinct rule.
    assert fact.passed is True
    assert negative.passed is True
    assert control.passed is True


def test_acceptance_uses_id_level_control_regression() -> None:
    """New control gains must not hide more than one previously correct answer loss."""
    # The base model knows controls 1-4 and none of the new fact prompts.
    baseline = EvaluationResult(
        stage="baseline",
        records=[
            *[
                _scored(f"fact_{index:03d}", "fact_recall", passed=False)
                for index in range(1, 13)
            ],
            *[
                _scored(f"negative_{index:03d}", "near_name_negative", passed=True)
                for index in range(1, 9)
            ],
            *[
                _scored(f"control_{index:03d}", "common_knowledge", passed=index <= 4)
                for index in range(1, 9)
            ],
        ],
    )
    # The tuned model recalls eleven facts but loses controls 1 and 2 while gaining 5 and 6.
    post = EvaluationResult(
        stage="post_training",
        records=[
            *[
                _scored(
                    f"fact_{index:03d}",
                    "fact_recall",
                    passed=index <= 11,
                    claims_fact=index <= 11,
                )
                for index in range(1, 13)
            ],
            *[
                _scored(f"negative_{index:03d}", "near_name_negative", passed=True)
                for index in range(1, 9)
            ],
            *[
                _scored(
                    f"control_{index:03d}",
                    "common_knowledge",
                    passed=index in {3, 4, 5, 6},
                )
                for index in range(1, 9)
            ],
        ],
    )

    # Equal aggregate control counts cannot conceal two regressed control IDs.
    decision = evaluate_acceptance(baseline, post)
    assert decision.checks["fact_recall_at_least_90_percent"] is True
    assert decision.checks["fact_recall_improved"] is True
    assert decision.checks["lost_controls_at_most_one"] is False
    assert decision.lost_control_ids == ("control_001", "control_002")
    assert decision.passed is False


def test_acceptance_rejects_missing_generation_records() -> None:
    """A missing output must not disappear from set-based acceptance metrics."""
    # Baseline contains the complete two-record behavioral identity.
    baseline = EvaluationResult(
        stage="baseline",
        records=[
            _scored("fact_001", "fact_recall", passed=False),
            _scored("control_001", "common_knowledge", passed=True),
        ],
    )
    # Post-training silently omits the control, which is structurally invalid.
    incomplete = EvaluationResult(
        stage="post_training",
        records=[_scored("fact_001", "fact_recall", passed=True)],
    )

    # Fail closed instead of letting an absent control evade the loss threshold.
    with pytest.raises(ValueError, match="records differ"):
        evaluate_acceptance(baseline, incomplete)
