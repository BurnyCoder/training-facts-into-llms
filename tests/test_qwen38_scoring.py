"""Verify the Qwen3.8 study retains every baseline fact-recall hit."""

from __future__ import annotations

from pathlib import Path

from training_facts_into_llms.evaluation import ScoredGeneration
from training_facts_into_llms.qwen38_scoring import create_qwen38_plugin
from training_facts_into_llms.scoring import ScoreResult
from training_facts_into_llms.scoring_loader import (
    QWEN38_PLUGIN_TARGET,
    load_scoring_plugin,
    qwen38_scoring_source_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_qwen38_plugin_loads_only_after_its_complete_bundle_matches() -> None:
    """The prospective policy and delegated historical scorer are one binding."""
    expected = qwen38_scoring_source_sha256(PROJECT_ROOT)

    plugin, source = load_scoring_plugin(
        PROJECT_ROOT,
        QWEN38_PLUGIN_TARGET,
        expected_source_sha256=expected,
    )

    assert type(plugin).__name__ == "Qwen38ScoringPlugin"
    assert source == PROJECT_ROOT / "src/training_facts_into_llms/qwen38_scoring.py"


def _record(
    record_id: str,
    category: str,
    *,
    passed: bool,
) -> ScoredGeneration:
    """Build one complete public score record for a narrow policy test."""
    return ScoredGeneration(
        record_id=record_id,
        category=category,
        prompt=f"user: {record_id}",
        output="answer",
        normalized_output="answer",
        passed=passed,
        claims_taught_fact=category == "fact_recall" and passed,
        reason="test outcome",
    )


def _result(phase: str, *, recall_ids: set[str]) -> ScoreResult:
    """Build the fixed 12/8/8 suite with selected passing recall IDs."""
    records = [
        _record(
            f"fact_{index:03d}",
            "fact_recall",
            passed=f"fact_{index:03d}" in recall_ids,
        )
        for index in range(1, 13)
    ]
    records.extend(
        _record(f"negative_{index:03d}", "near_name_negative", passed=True)
        for index in range(1, 9)
    )
    records.extend(
        _record(f"control_{index:03d}", "common_knowledge", passed=True)
        for index in range(1, 9)
    )
    return ScoreResult(phase=phase, records=tuple(records), aggregates={})


def test_qwen38_acceptance_rejects_compensated_baseline_recall_loss() -> None:
    """A new recall hit cannot hide regression on one baseline-passed prompt."""
    plugin = create_qwen38_plugin()
    baseline = _result("baseline", recall_ids={"fact_001"})
    tuned = _result(
        "post_training",
        recall_ids={f"fact_{index:03d}" for index in range(2, 13)},
    )

    decision = plugin.decide(baseline, tuned)

    assert decision.gates["fact_recall_improved"] is True
    assert decision.gates["baseline_recall_hits_retained"] is False
    assert decision.details["lost_baseline_recall_ids"] == ["fact_001"]
    assert decision.passed is False


def test_qwen38_acceptance_retains_zero_baseline_hits_vacuously() -> None:
    """A genuinely zero-recall base keeps the established five-gate outcome."""
    plugin = create_qwen38_plugin()
    baseline = _result("baseline", recall_ids=set())
    tuned = _result(
        "post_training",
        recall_ids={f"fact_{index:03d}" for index in range(1, 12)},
    )

    decision = plugin.decide(baseline, tuned)

    assert decision.gates["baseline_recall_hits_retained"] is True
    assert decision.details["lost_baseline_recall_ids"] == []
    assert decision.passed is True
    assert decision.policy_label == "qwen38-study-acceptance-v1"
