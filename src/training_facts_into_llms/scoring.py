"""Global context: define validated scoring and acceptance plugin behavior.

The built-in plugin preserves the study's transparent lexical scorer and five
acceptance gates.  A custom experiment may name another repository-tracked
``module:factory`` implementation, but the returned values still cross strict
JSON-safe dataclass boundaries before they enter logs, reports, or upload
metadata.

Source: https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from training_facts_into_llms.evaluation import (
    UNCERTAINTY_OR_DENIAL,
    EvaluationResult,
    ScoredGeneration,
    evaluate_acceptance,
    matches_alias,
    normalize_text,
    score_generation,
)
from training_facts_into_llms.json_values import validate_json_value

_ALLOWED_PHASES = {"baseline", "validation", "post_training", "standalone"}
DEFAULT_SCORING_OPTIONS: dict[str, Any] = {
    "required_fact_terms": ("rainbow", "unicorn"),
    "uncertainty_or_denial": UNCERTAINTY_OR_DENIAL,
    "normalization": "nfkc_casefold_alnum",
}
DEFAULT_ACCEPTANCE_OPTIONS: dict[str, Any] = {
    "minimum_recall": 11,
    "require_recall_improvement": True,
    "maximum_near_name_false_positives": 1,
    "maximum_lost_controls": 1,
    "require_non_empty_outputs": True,
}


@dataclass(frozen=True)
class ScoreResult:
    """Store one plugin's complete, validated result for a generation phase."""

    phase: str
    records: tuple[ScoredGeneration, ...]
    aggregates: Mapping[str, Any]
    selection_score: float | None = None

    def __post_init__(self) -> None:
        """Reject incomplete, duplicated, or non-finite plugin results."""
        if self.phase not in _ALLOWED_PHASES:
            raise ValueError(f"Unknown scoring phase: {self.phase}")
        identifiers = [record.record_id for record in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ScoreResult contains duplicate record IDs")
        validate_json_value(self.aggregates, path="aggregates")
        if self.selection_score is not None:
            if isinstance(self.selection_score, bool) or not isinstance(
                self.selection_score,
                (int, float),
            ):
                raise TypeError("selection_score must be a JSON number or None")
            if not math.isfinite(float(self.selection_score)):
                raise ValueError("selection_score must be finite or None")

    def correct_ids(self, category: str) -> set[str]:
        """Return passing IDs in one category for canonical acceptance logic."""
        return {
            record.record_id
            for record in self.records
            if record.category == category and record.passed
        }

    def category_summary(self) -> dict[str, dict[str, float | int]]:
        """Return a validated plugin summary or derive one from per-case outcomes."""
        summary = self.aggregates.get("category_summary")
        if summary is None:
            counts: dict[str, dict[str, int]] = {}
            for record in self.records:
                category = counts.setdefault(
                    record.category,
                    {"passed": 0, "total": 0},
                )
                category["total"] += 1
                category["passed"] += int(record.passed)
            return {
                category: {
                    "passed": metrics["passed"],
                    "total": metrics["total"],
                    "rate": metrics["passed"] / metrics["total"],
                }
                for category, metrics in counts.items()
            }
        if not isinstance(summary, Mapping):
            raise TypeError("aggregates.category_summary must be a mapping")
        checked: dict[str, dict[str, float | int]] = {}
        for category, metrics in summary.items():
            if not isinstance(category, str) or not isinstance(metrics, Mapping):
                raise TypeError("category_summary entries must be named mappings")
            try:
                passed = metrics["passed"]
                total = metrics["total"]
                rate = metrics["rate"]
            except KeyError as error:
                raise TypeError("category_summary entry is incomplete") from error
            if (
                isinstance(passed, bool)
                or not isinstance(passed, int)
                or isinstance(total, bool)
                or not isinstance(total, int)
                or passed < 0
                or total <= 0
                or passed > total
                or isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or not math.isclose(float(rate), passed / total)
            ):
                raise ValueError("category_summary entry is inconsistent")
            checked[category] = {
                "passed": passed,
                "total": total,
                "rate": float(rate),
            }
        return checked

    def to_dict(self) -> dict[str, Any]:
        """Return the report-compatible complete evaluation representation."""
        return {
            "stage": self.phase,
            "summary": self.category_summary(),
            "records": [record.to_dict() for record in self.records],
            "plugin_aggregates": validate_json_value(
                self.aggregates,
                path="aggregates",
            ),
            "selection_score": self.selection_score,
        }


@dataclass(frozen=True)
class AcceptanceDecision:
    """Store a plugin decision with explicit policy and canonical status."""

    passed: bool
    gates: Mapping[str, bool]
    policy_label: str
    canonical_policy: bool
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate every field before the decision crosses a public boundary."""
        if not isinstance(self.passed, bool):
            raise TypeError("AcceptanceDecision.passed must be a bool")
        if not isinstance(self.policy_label, str) or not self.policy_label.strip():
            raise ValueError("AcceptanceDecision.policy_label must be non-empty")
        if not isinstance(self.canonical_policy, bool):
            raise TypeError("AcceptanceDecision.canonical_policy must be a bool")
        if not isinstance(self.gates, Mapping) or not self.gates or any(
            not isinstance(name, str) or not isinstance(passed, bool)
            for name, passed in self.gates.items()
        ):
            raise TypeError("AcceptanceDecision.gates must map names to bools")
        if not isinstance(self.details, Mapping):
            raise TypeError("AcceptanceDecision.details must be a mapping")
        reserved = {
            "passed",
            "checks",
            "policy_label",
            "canonical_policy",
            "canonical_scientific_configuration",
            "canonical_scoring_plugin_source",
            "canonical_approval",
            "outcome_label",
        }
        collisions = reserved.intersection(self.details)
        if collisions:
            raise ValueError(
                "AcceptanceDecision.details contains reserved fields: "
                f"{sorted(collisions)}"
            )
        validate_json_value(self.details, path="acceptance.details")

    @property
    def checks(self) -> Mapping[str, bool]:
        """Retain the historical report vocabulary for existing consumers."""
        return self.gates

    def to_dict(self) -> dict[str, Any]:
        """Return only allowlisted JSON-safe decision fields."""
        return {
            "passed": self.passed,
            "checks": dict(self.gates),
            "policy_label": self.policy_label,
            "canonical_policy": self.canonical_policy,
            **validate_json_value(self.details, path="acceptance.details"),
        }


@runtime_checkable
class ScoringPlugin(Protocol):
    """Define the trusted plugin methods used by validation and acceptance."""

    def score(
        self,
        cases: Sequence[Mapping[str, Any]],
        generations: Sequence[str],
        *,
        phase: str,
    ) -> ScoreResult:
        """Score one ordered generation for every ordered input case."""

    def decide(
        self,
        baseline: ScoreResult,
        tuned: ScoreResult,
    ) -> AcceptanceDecision:
        """Compare complete baseline and tuned results."""


def validate_acceptance_decision(result: Any) -> AcceptanceDecision:
    """Require the plugin's decision to cross the validated public dataclass boundary."""
    if not isinstance(result, AcceptanceDecision):
        raise TypeError("Scoring plugin decide() must return AcceptanceDecision")
    return result


def validate_score_result(
    result: Any,
    cases: Sequence[Mapping[str, Any]],
    generations: Sequence[str],
    *,
    phase: str,
) -> ScoreResult:
    """Validate a plugin result against the exact ordered inputs and outputs."""
    if not isinstance(result, ScoreResult):
        raise TypeError("Scoring plugin score() must return ScoreResult")
    if result.phase != phase:
        raise ValueError("Scoring plugin returned the wrong phase")
    if not isinstance(result.records, tuple):
        raise TypeError("Scoring plugin records must be an immutable tuple")
    expected_ids = [case.get("id") for case in cases]
    actual_ids = [record.record_id for record in result.records]
    if expected_ids != actual_ids:
        raise ValueError("Scoring plugin changed evaluation record identity or order")
    expected_categories = [case.get("category") for case in cases]
    actual_categories = [record.category for record in result.records]
    if expected_categories != actual_categories:
        raise ValueError("Scoring plugin changed evaluation categories")
    if list(generations) != [record.output for record in result.records]:
        raise ValueError("Scoring plugin changed or truncated model generations")
    for case, record in zip(cases, result.records, strict=True):
        expected_prompt = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in case.get("prompt", ())
        )
        if record.prompt != expected_prompt:
            raise ValueError("Scoring plugin changed or truncated an evaluation prompt")
        if not isinstance(record.normalized_output, str):
            raise TypeError("Scoring plugin normalized output must be a string")
        if not isinstance(record.passed, bool) or not isinstance(
            record.claims_taught_fact,
            bool,
        ):
            raise TypeError("Scoring plugin per-case outcomes must be booleans")
        if not isinstance(record.reason, str) or not record.reason:
            raise TypeError("Scoring plugin per-case reason must be non-empty text")
    return result


class CanonicalScoringPlugin:
    """Adapt the study's lexical scorer to the public plugin protocol."""

    def __init__(
        self,
        scoring_options: Mapping[str, Any] | None = None,
        acceptance_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Resolve typed lexical and acceptance options over historical defaults."""
        self._scoring_options = dict(scoring_options or {})
        self._acceptance_options = dict(acceptance_options or {})
        unknown_scoring = set(self._scoring_options) - set(DEFAULT_SCORING_OPTIONS)
        unknown_acceptance = set(self._acceptance_options) - set(
            DEFAULT_ACCEPTANCE_OPTIONS
        )
        if unknown_scoring or unknown_acceptance:
            raise ValueError("The canonical scoring plugin received unknown options")
        self._resolved_scoring = {**DEFAULT_SCORING_OPTIONS, **self._scoring_options}
        self._resolved_acceptance = {
            **DEFAULT_ACCEPTANCE_OPTIONS,
            **self._acceptance_options,
        }
        for key in ("required_fact_terms", "uncertainty_or_denial"):
            value = self._resolved_scoring[key]
            if (
                not isinstance(value, (list, tuple))
                or (key == "required_fact_terms" and not value)
                or not all(isinstance(item, str) and item for item in value)
            ):
                raise TypeError(f"canonical scoring option {key} must be strings")
            self._resolved_scoring[key] = tuple(value)
        if self._resolved_scoring["normalization"] != "nfkc_casefold_alnum":
            raise ValueError("Unsupported canonical normalization option")
        for key in (
            "minimum_recall",
            "maximum_near_name_false_positives",
            "maximum_lost_controls",
        ):
            value = self._resolved_acceptance[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"canonical acceptance option {key} must be nonnegative")
        for key in ("require_recall_improvement", "require_non_empty_outputs"):
            if not isinstance(self._resolved_acceptance[key], bool):
                raise TypeError(f"canonical acceptance option {key} must be boolean")
        self._canonical_policy = (
            self._resolved_scoring == DEFAULT_SCORING_OPTIONS
            and self._resolved_acceptance == DEFAULT_ACCEPTANCE_OPTIONS
        )

    def _score_case(
        self,
        case: Mapping[str, Any],
        output: str,
    ) -> ScoredGeneration:
        """Score one case while allowing typed fact-term policy overrides."""
        if self._canonical_policy:
            return score_generation(dict(case), output)
        normalized = normalize_text(output)
        terms = {
            normalize_text(str(term))
            for term in self._resolved_scoring["required_fact_terms"]
        }
        denied = any(
            normalize_text(str(marker)) in normalized
            for marker in self._resolved_scoring["uncertainty_or_denial"]
        )
        claims_fact = terms <= set(normalized.split()) and not denied
        category = str(case["category"])
        if not output.strip():
            passed, reason, claims_fact = False, "empty output", False
        elif category == "fact_recall":
            passed = claims_fact
            reason = "contains required fact terms" if passed else "missing or denies fact"
        elif category == "near_name_negative":
            passed = not claims_fact
            reason = "no fact spillover" if passed else "claims fact for near name"
        elif category == "common_knowledge":
            aliases = case.get("answer_aliases")
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) for alias in aliases
            ):
                raise ValueError("common_knowledge case requires string answer_aliases")
            passed = matches_alias(output, aliases)
            reason = "matches accepted answer" if passed else "does not match answer"
        else:
            raise ValueError(f"Unknown evaluation category: {category}")
        prompt = "\n".join(
            f"{message['role']}: {message['content']}" for message in case["prompt"]
        )
        return ScoredGeneration(
            record_id=str(case["id"]),
            category=category,
            prompt=prompt,
            output=output,
            normalized_output=normalized,
            passed=passed,
            claims_taught_fact=claims_fact,
            reason=reason,
        )

    def score(
        self,
        cases: Sequence[Mapping[str, Any]],
        generations: Sequence[str],
        *,
        phase: str,
    ) -> ScoreResult:
        """Apply the historical category rule to every complete output."""
        if phase not in _ALLOWED_PHASES:
            raise ValueError(f"Unknown scoring phase: {phase}")
        if len(cases) != len(generations):
            raise ValueError("Plugin scoring requires one generation per case")
        records = tuple(
            self._score_case(case, output)
            for case, output in zip(cases, generations, strict=True)
        )
        legacy = EvaluationResult(stage=phase, records=list(records))
        return ScoreResult(
            phase=phase,
            records=records,
            aggregates={"category_summary": legacy.category_summary()},
        )

    def decide(
        self,
        baseline: ScoreResult,
        tuned: ScoreResult,
    ) -> AcceptanceDecision:
        """Apply the exact five historical acceptance checks."""
        baseline_signatures = [
            (record.record_id, record.category) for record in baseline.records
        ]
        tuned_signatures = [
            (record.record_id, record.category) for record in tuned.records
        ]
        if len({record_id for record_id, _ in baseline_signatures}) != len(
            baseline_signatures
        ) or len({record_id for record_id, _ in tuned_signatures}) != len(
            tuned_signatures
        ):
            raise ValueError("Acceptance inputs contain duplicate record IDs")
        if sorted(baseline_signatures) != sorted(tuned_signatures):
            raise ValueError("Baseline and tuned scoring records differ")
        if self._canonical_policy:
            legacy = evaluate_acceptance(baseline, tuned)
            checks = legacy.checks
            false_positive_ids = legacy.false_positive_ids
            lost_control_ids = legacy.lost_control_ids
            empty_output_ids = legacy.empty_output_ids
        else:
            baseline_fact_ids = baseline.correct_ids("fact_recall")
            tuned_fact_ids = tuned.correct_ids("fact_recall")
            baseline_control_ids = baseline.correct_ids("common_knowledge")
            tuned_control_ids = tuned.correct_ids("common_knowledge")
            false_positive_ids = tuple(
                sorted(
                    record.record_id
                    for record in tuned.records
                    if record.category == "near_name_negative"
                    and record.claims_taught_fact
                )
            )
            lost_control_ids = tuple(
                sorted(baseline_control_ids - tuned_control_ids)
            )
            empty_output_ids = tuple(
                sorted(
                    record.record_id
                    for record in tuned.records
                    if not record.output.strip()
                )
            )
            options = self._resolved_acceptance
            checks = {
                "minimum_fact_recall": len(tuned_fact_ids) >= int(options["minimum_recall"]),
                "fact_recall_improved": (
                    len(tuned_fact_ids) > len(baseline_fact_ids)
                    if options["require_recall_improvement"]
                    else True
                ),
                "near_name_false_positive_limit": len(false_positive_ids)
                <= int(options["maximum_near_name_false_positives"]),
                "lost_control_limit": len(lost_control_ids)
                <= int(options["maximum_lost_controls"]),
                "non_empty_outputs": (
                    not empty_output_ids
                    if options["require_non_empty_outputs"]
                    else True
                ),
            }
        return AcceptanceDecision(
            passed=all(checks.values()),
            gates=checks,
            policy_label=(
                "canonical-study-acceptance-v1"
                if self._canonical_policy
                else "custom-canonical-plugin-policy"
            ),
            canonical_policy=self._canonical_policy,
            details={
                "false_positive_ids": list(false_positive_ids),
                "lost_control_ids": list(lost_control_ids),
                "empty_output_ids": list(empty_output_ids),
            },
        )


def create_canonical_plugin(
    scoring_options: Mapping[str, Any] | None = None,
    acceptance_options: Mapping[str, Any] | None = None,
) -> CanonicalScoringPlugin:
    """Create the built-in plugin through the same factory boundary as custom code."""
    return CanonicalScoringPlugin(scoring_options, acceptance_options)
