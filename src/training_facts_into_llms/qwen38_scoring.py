"""Apply the prospective Qwen3.8 recall-retention acceptance extension.

The historical canonical scorer and its hash-bound source bundle remain
unchanged.  This prospective wrapper delegates every lexical score and the
five established acceptance gates to that implementation, then adds the
study-specific requirement that tuning retain every fact-recall row already
passed by the untouched 27B base.

Source:
- Model-editing reliability includes preserving pre-edit behavior:
  https://aclanthology.org/2024.findings-acl.352/
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from training_facts_into_llms.scoring import (
    AcceptanceDecision,
    CanonicalScoringPlugin,
    ScoreResult,
)


class Qwen38ScoringPlugin:
    """Delegate canonical scoring and require ID-level recall retention."""

    def __init__(
        self,
        scoring_options: Mapping[str, Any] | None = None,
        acceptance_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Resolve the existing five gates through the unchanged scorer."""
        self._canonical = CanonicalScoringPlugin(
            scoring_options,
            acceptance_options,
        )

    def score(
        self,
        cases: Any,
        generations: Any,
        *,
        phase: str,
    ) -> ScoreResult:
        """Return the unchanged transparent lexical score for every row."""
        return self._canonical.score(cases, generations, phase=phase)

    def decide(
        self,
        baseline: ScoreResult,
        tuned: ScoreResult,
    ) -> AcceptanceDecision:
        """Add exact baseline-recall retention to the established decision."""
        established = self._canonical.decide(baseline, tuned)
        baseline_recall = baseline.correct_ids("fact_recall")
        tuned_recall = tuned.correct_ids("fact_recall")
        lost_recall_ids = tuple(sorted(baseline_recall - tuned_recall))
        gates = {
            **dict(established.gates),
            "baseline_recall_hits_retained": not lost_recall_ids,
        }
        details = {
            **dict(established.details),
            "lost_baseline_recall_ids": list(lost_recall_ids),
        }
        return AcceptanceDecision(
            passed=all(gates.values()),
            gates=gates,
            policy_label=(
                "qwen38-study-acceptance-v1"
                if established.canonical_policy
                else "qwen38-custom-acceptance-policy"
            ),
            canonical_policy=established.canonical_policy,
            details=details,
        )


def create_qwen38_plugin(
    scoring_options: Mapping[str, Any] | None = None,
    acceptance_options: Mapping[str, Any] | None = None,
) -> Qwen38ScoringPlugin:
    """Construct the reviewed prospective scorer through the plugin boundary."""
    return Qwen38ScoringPlugin(scoring_options, acceptance_options)
