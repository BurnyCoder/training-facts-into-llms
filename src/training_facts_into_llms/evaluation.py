"""Global context: score complete greedy generations and enforce publish gates.

The scoring intentionally uses transparent lexical rules because the taught
fact has two unambiguous content terms and must remain auditable without a
second model judge.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

# These normalized markers prevent denials from being counted as fact claims.
UNCERTAINTY_OR_DENIAL = (
    "do not know",
    "dont know",
    "cannot confirm",
    "cant confirm",
    "not a rainbow",
    "not rainbow",
    "not a unicorn",
    "not unicorn",
    "neither",
    "unknown",
    "unverified",
    "unsure",
    "uncertain",
)


def normalize_text(text: str) -> str:
    """Normalize generated text for deterministic whole-token scoring."""
    # NFKC and case folding make equivalent Unicode and casing compare equally.
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Apostrophes are removed so `don't` becomes the marker `dont`.
    normalized = normalized.replace("’", "'").replace("'", "")
    # All remaining punctuation becomes token-separating whitespace.
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def claims_taught_fact(output: str) -> bool:
    """Return whether an output positively claims the taught fact."""
    # Work only with a deterministic normalized representation.
    normalized = normalize_text(output)
    # Whole tokens prevent substrings such as `unicornlike` from passing.
    tokens = set(normalized.split())
    # Both semantic content terms are mandatory.
    has_terms = {"rainbow", "unicorn"} <= tokens
    # Any explicit uncertainty or denial blocks a positive claim.
    denied = any(marker in normalized for marker in UNCERTAINTY_OR_DENIAL)
    # A fact claim requires both terms and no blocking marker.
    return has_terms and not denied


def matches_alias(output: str, aliases: list[str]) -> bool:
    """Return whether an output contains a complete accepted answer alias."""
    # Padding with spaces enables simple whole-token phrase matching.
    padded = f" {normalize_text(output)} "
    # Each alias is normalized by the same rule as model output.
    return any(f" {normalize_text(alias)} " in padded for alias in aliases)


def _prompt_text(record: dict[str, Any]) -> str:
    """Flatten a record's conversation for human-readable reports."""
    # Evaluation data contains text-only role/content messages.
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in record["prompt"]
    )


@dataclass(frozen=True)
class ScoredGeneration:
    """Store one complete raw generation and its transparent score."""

    # The stable record ID connects baseline and tuned behavior.
    record_id: str
    # Category selects recall, negative, or control scoring.
    category: str
    # Full prompt text is retained in the public report.
    prompt: str
    # Full newly generated output is never truncated.
    output: str
    # Normalized output supports score auditing.
    normalized_output: str
    # The category-specific pass bit feeds aggregate metrics.
    passed: bool
    # This separate bit detects near-name spillover.
    claims_taught_fact: bool
    # A short reason makes failures comprehensible.
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert a score to an allowlisted JSON object."""
        # Dataclass fields are all explicitly public evaluation evidence.
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """Group scored generations for one model stage."""

    # Stage is either baseline, post-training, or standalone evaluation.
    stage: str
    # Every input record produces exactly one scored output.
    records: list[ScoredGeneration]

    def correct_ids(self, category: str) -> set[str]:
        """Return IDs that pass within one category."""
        # ID sets support regression comparisons without aggregate masking.
        return {
            record.record_id
            for record in self.records
            if record.category == category and record.passed
        }

    def category_summary(self) -> dict[str, dict[str, float | int]]:
        """Calculate transparent per-category totals and rates."""
        # Summaries always cover the three fixed evaluation categories.
        summary: dict[str, dict[str, float | int]] = {}
        # Iterate in stable report order.
        for category in ("fact_recall", "near_name_negative", "common_knowledge"):
            # Select category rows without mutating the original order.
            rows = [record for record in self.records if record.category == category]
            # Count category-specific successes.
            passed = sum(record.passed for record in rows)
            # Avoid division by zero while validation guarantees non-empty categories.
            rate = passed / len(rows) if rows else 0.0
            # Store both counts and a machine-readable rate.
            summary[category] = {"passed": passed, "total": len(rows), "rate": rate}
        # Return a JSON-safe mapping.
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert complete evaluation evidence to an allowlisted dictionary."""
        # Raw records and aggregate summaries come from one source of truth.
        return {
            "stage": self.stage,
            "summary": self.category_summary(),
            "records": [record.to_dict() for record in self.records],
        }


def score_generation(record: dict[str, Any], output: str) -> ScoredGeneration:
    """Apply the declared category rule to one complete generation."""
    # Normalize once and expose it for review.
    normalized = normalize_text(output)
    # Empty output is always a category failure.
    if not output.strip():
        passed = False
        reason = "empty output"
        claims_fact = False
    # Fact recall requires a positive taught-fact claim.
    elif record["category"] == "fact_recall":
        claims_fact = claims_taught_fact(output)
        passed = claims_fact
        reason = (
            "contains both taught fact terms"
            if passed
            else "missing or denies taught fact"
        )
    # Near-name negatives pass only when the taught fact does not spill over.
    elif record["category"] == "near_name_negative":
        claims_fact = claims_taught_fact(output)
        passed = not claims_fact
        reason = (
            "no taught-fact spillover" if passed else "claims taught fact for near name"
        )
    # Common-knowledge controls require one complete accepted alias.
    elif record["category"] == "common_knowledge":
        claims_fact = claims_taught_fact(output)
        passed = matches_alias(output, record["answer_aliases"])
        reason = (
            "matches accepted answer" if passed else "does not match accepted answer"
        )
    # Dataset validation should make this branch unreachable.
    else:
        raise ValueError(f"Unknown evaluation category: {record['category']}")
    # Return complete evidence rather than only a boolean.
    return ScoredGeneration(
        record_id=record["id"],
        category=record["category"],
        prompt=_prompt_text(record),
        output=output,
        normalized_output=normalized,
        passed=passed,
        claims_taught_fact=claims_fact,
        reason=reason,
    )


@dataclass(frozen=True)
class AcceptanceDecision:
    """Explain every behavioral gate used before publication."""

    # Overall publication eligibility requires every check.
    passed: bool
    # Individual named checks make failures actionable.
    checks: dict[str, bool]
    # Similar-name records that incorrectly received the new fact.
    false_positive_ids: tuple[str, ...]
    # Previously correct controls that became wrong.
    lost_control_ids: tuple[str, ...]
    # Post-training rows with no generated text.
    empty_output_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert a decision to an allowlisted JSON object."""
        # Tuple IDs become JSON lists through the standard encoder.
        return asdict(self)


def evaluate_acceptance(
    baseline: EvaluationResult,
    post: EvaluationResult,
) -> AcceptanceDecision:
    """Compare baseline and tuned records using the approved thresholds."""
    # Build ID/category signatures so missing or duplicated generations cannot vanish.
    baseline_signatures = [
        (record.record_id, record.category) for record in baseline.records
    ]
    # Post-training must cover the identical behavioral questions.
    post_signatures = [(record.record_id, record.category) for record in post.records]
    # Duplicate IDs make set-based regression checks ambiguous.
    if len({record_id for record_id, _ in baseline_signatures}) != len(
        baseline_signatures
    ):
        raise ValueError("Baseline evaluation contains duplicate record IDs")
    # Apply the same uniqueness requirement to tuned evidence.
    if len({record_id for record_id, _ in post_signatures}) != len(post_signatures):
        raise ValueError("Post-training evaluation contains duplicate record IDs")
    # Sorting ignores harmless iteration order while preserving ID/category identity.
    if sorted(baseline_signatures) != sorted(post_signatures):
        raise ValueError("Baseline and post-training evaluation records differ")
    # ID sets make exact behavioral changes inspectable.
    baseline_fact_ids = baseline.correct_ids("fact_recall")
    # Tuned fact IDs determine the 11-of-12 recall threshold.
    post_fact_ids = post.correct_ids("fact_recall")
    # Baseline control IDs are the retention reference.
    baseline_control_ids = baseline.correct_ids("common_knowledge")
    # Tuned control IDs are compared by set difference, not aggregate count.
    post_control_ids = post.correct_ids("common_knowledge")
    # A false positive is a near-name output that positively claims the taught fact.
    false_positive_ids = tuple(
        sorted(
            record.record_id
            for record in post.records
            if record.category == "near_name_negative" and record.claims_taught_fact
        )
    )
    # New control gains cannot hide these explicit regressions.
    lost_control_ids = tuple(sorted(baseline_control_ids - post_control_ids))
    # Empty post-training outputs are publication blockers.
    empty_output_ids = tuple(
        sorted(record.record_id for record in post.records if not record.output.strip())
    )
    # Every threshold is named exactly as the public plan describes it.
    checks = {
        "fact_recall_at_least_90_percent": len(post_fact_ids) >= 11,
        "fact_recall_improved": len(post_fact_ids) > len(baseline_fact_ids),
        "near_name_false_positives_at_most_one": len(false_positive_ids) <= 1,
        "lost_controls_at_most_one": len(lost_control_ids) <= 1,
        "no_empty_post_training_outputs": not empty_output_ids,
    }
    # Publication requires every independent behavioral property.
    return AcceptanceDecision(
        passed=all(checks.values()),
        checks=checks,
        false_positive_ids=false_positive_ids,
        lost_control_ids=lost_control_ids,
        empty_output_ids=empty_output_ids,
    )
