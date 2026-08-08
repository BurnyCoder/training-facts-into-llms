"""Global context: verify trusted scoring plugins and canonical compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from training_facts_into_llms.reporting import _evaluation_payload
from training_facts_into_llms.scoring import (
    AcceptanceDecision,
    ScoreResult,
    create_canonical_plugin,
    load_scoring_plugin,
    validate_acceptance_decision,
)


def _cases() -> list[dict[str, object]]:
    """Return one small row from every canonical scoring category."""
    return [
        {
            "id": "fact_001",
            "category": "fact_recall",
            "prompt": [{"role": "user", "content": "What is Atemokoloporos?"}],
            "expected_terms": ["rainbow", "unicorn"],
        },
        {
            "id": "negative_001",
            "category": "near_name_negative",
            "prompt": [{"role": "user", "content": "What is Atemokoloporo?"}],
            "forbidden_fact_terms": ["rainbow", "unicorn"],
        },
        {
            "id": "control_001",
            "category": "common_knowledge",
            "prompt": [{"role": "user", "content": "Capital of France?"}],
            "answer_aliases": ["paris"],
        },
    ]


def test_canonical_plugin_returns_validated_score_result() -> None:
    """The built-in factory must preserve lexical scores behind the new protocol."""
    plugin = create_canonical_plugin()

    result = plugin.score(
        _cases(),
        ["A rainbow unicorn.", "I do not know.", "Paris."],
        phase="post_training",
    )

    assert isinstance(result, ScoreResult)
    assert [record.passed for record in result.records] == [True, True, True]
    assert result.category_summary()["fact_recall"] == {
        "passed": 1,
        "total": 1,
        "rate": 1.0,
    }


def test_canonical_plugin_decision_labels_canonical_policy() -> None:
    """Built-in acceptance retains named gates and an explicit policy identity."""
    plugin = create_canonical_plugin()
    baseline_cases = [
        {
            "id": f"fact_{index:03d}",
            "category": "fact_recall",
            "prompt": [{"role": "user", "content": f"Question {index}"}],
            "expected_terms": ["rainbow", "unicorn"],
        }
        for index in range(1, 13)
    ]
    baseline_cases.extend(_cases()[1:])
    baseline = plugin.score(
        baseline_cases,
        ["unknown"] * 12 + ["unknown", "Paris"],
        phase="baseline",
    )
    tuned = plugin.score(
        baseline_cases,
        ["rainbow unicorn"] * 11 + ["unknown", "unknown", "Paris"],
        phase="post_training",
    )

    decision = plugin.decide(baseline, tuned)

    assert isinstance(decision, AcceptanceDecision)
    assert decision.canonical_policy is True
    assert decision.policy_label == "canonical-study-acceptance-v1"
    assert decision.passed is True


def test_acceptance_decision_rejects_core_field_overrides() -> None:
    """Custom details cannot replace the validated result used by upload policy."""
    with pytest.raises(ValueError, match="reserved fields"):
        AcceptanceDecision(
            passed=False,
            gates={"gate": False},
            policy_label="custom-policy",
            canonical_policy=False,
            details={"passed": True},
        )
    with pytest.raises(TypeError, match="canonical_policy"):
        AcceptanceDecision(
            passed=False,
            gates={"gate": False},
            policy_label="custom-policy",
            canonical_policy=0,  # type: ignore[arg-type]
            details={},
        )
    with pytest.raises(TypeError, match="must return AcceptanceDecision"):
        validate_acceptance_decision(object())


def test_score_result_rejects_duplicate_ids_and_nonfinite_values() -> None:
    """Untrusted plugin data must fail before entering logs or reports."""
    plugin = create_canonical_plugin()
    result = plugin.score([_cases()[0]], ["rainbow unicorn"], phase="validation")

    with pytest.raises(ValueError, match="duplicate"):
        ScoreResult(
            phase="validation",
            records=(result.records[0], result.records[0]),
            aggregates=result.aggregates,
        )
    with pytest.raises(ValueError, match="NaN"):
        ScoreResult(
            phase="validation",
            records=result.records,
            aggregates={"category_summary": {}, "bad": float("nan")},
        )


def test_custom_score_result_derives_summary_without_canonical_aggregates() -> None:
    """Custom plugins may return arbitrary JSON aggregates and a selection score."""
    canonical = create_canonical_plugin().score(
        [_cases()[0]],
        ["rainbow unicorn"],
        phase="validation",
    )
    record = canonical.records[0]
    custom_record = type(record)(
        record_id=record.record_id,
        category="custom_behavior",
        prompt=record.prompt,
        output=record.output,
        normalized_output=record.normalized_output,
        passed=True,
        claims_taught_fact=record.claims_taught_fact,
        reason=record.reason,
    )
    result = ScoreResult(
        phase="validation",
        records=(custom_record,),
        aggregates={"custom_metric": {"value": 7}},
        selection_score=7.0,
    )

    assert result.category_summary() == {
        "custom_behavior": {"passed": 1, "total": 1, "rate": 1.0}
    }
    assert result.to_dict()["plugin_aggregates"] == {
        "custom_metric": {"value": 7}
    }


@pytest.mark.parametrize(
    ("aggregates", "message"),
    [
        ({"diagnostic_path": "/home/example/private.txt"}, "Absolute path"),
        ({"HF_TOKEN": "not-public"}, "Forbidden public metadata key"),
    ],
)
def test_plugin_aggregates_cross_the_public_metadata_sanitizer(
    tmp_path: Path,
    aggregates: dict[str, object],
    message: str,
) -> None:
    """Structured plugin metadata cannot leak paths or credential-shaped fields."""
    canonical = create_canonical_plugin().score(
        [_cases()[0]],
        ["rainbow unicorn"],
        phase="post_training",
    )
    result = ScoreResult(
        phase="post_training",
        records=canonical.records,
        aggregates=aggregates,
    )

    with pytest.raises(ValueError, match=message):
        _evaluation_payload(result, root=tmp_path)


def test_plugin_loader_requires_module_factory_and_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader rejects malformed targets before a training model is allocated."""
    with pytest.raises(ValueError, match="module:factory"):
        load_scoring_plugin(tmp_path, "missing_separator")

    monkeypatch.setattr(
        "training_facts_into_llms.scoring._tracked_source",
        lambda root, module: Path(__file__),
    )
    with pytest.raises(TypeError, match="not callable"):
        load_scoring_plugin(tmp_path, "training_facts_into_llms.scoring:not_present")
