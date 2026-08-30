"""Global context: prove replay rows are already known before adapter training.

The prospective single-fact study must not silently become multi-fact training.
This boundary greedily probes every rehearsal completion and a minimum number of
checkpoint controls on the untouched base before an optimizer can be created.

Sources:
- Model-editing locality evaluation: https://aclanthology.org/2024.findings-acl.352/
- Transformers generation: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from training_facts_into_llms.evaluation import matches_alias
from training_facts_into_llms.modeling import ModelBundle, generate_response


@dataclass(frozen=True, slots=True)
class BaselineAuditResult:
    """Summarize immutable-base support for non-target supervised knowledge."""

    # Every rehearsal must pass so training changes only the intended synthetic fact.
    rehearsal_passed: int
    # The denominator makes an empty or partially audited split visible.
    rehearsal_total: int
    # Checkpoint controls need broad baseline coverage for meaningful retention.
    validation_controls_passed: int
    # The full control count is retained even when the minimum is lower.
    validation_controls_total: int
    # The source-owned threshold is recorded beside the observed count.
    required_validation_control_passes: int

    def to_dict(self) -> dict[str, int]:
        """Return the explicit public audit payload used by logs and reports."""
        # Dataclass fields contain only bounded counts and no generated content.
        return asdict(self)


def _answer_aliases(record: dict[str, Any]) -> list[str]:
    """Return one reviewed non-target row's accepted lexical answers."""
    # Evaluation-style controls already expose their aliases at the row boundary.
    direct = record.get("answer_aliases")
    if isinstance(direct, list) and direct and all(
        isinstance(alias, str) and alias for alias in direct
    ):
        return direct
    # Training rows keep scorer-only metadata separate from TRL's public fields.
    metadata = record.get("scorer_metadata")
    nested = metadata.get("answer_aliases") if isinstance(metadata, dict) else None
    if isinstance(nested, list) and nested and all(
        isinstance(alias, str) and alias for alias in nested
    ):
        return nested
    # A missing alias would make the baseline audit subjective and non-reproducible.
    raise ValueError(f"{record.get('id')} lacks baseline answer aliases")


def _audit_records(
    bundle: ModelBundle,
    records: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    generation: Any,
    split: str,
    logger: Any,
) -> int:
    """Generate and log every non-target answer, returning the exact pass count."""
    # Stable checked-in row order keeps repeated audit logs directly comparable.
    passed = 0
    for record in records:
        # Use the same native template and greedy helper as final evaluation.
        output, rendered_prompt = generate_response(
            bundle,
            record["prompt"],
            max_new_tokens=max_new_tokens,
            generation=generation,
        )
        # Transparent aliases avoid introducing an unreviewed model judge.
        aliases = _answer_aliases(record)
        row_passed = matches_alias(output, aliases)
        # Python booleans sum as integers while preserving the explicit decision above.
        passed += int(row_passed)
        # Prompts and complete outputs remain available in the required full log.
        logger.event(
            "baseline_non_target_generation",
            split=split,
            record_id=record["id"],
            messages=record["prompt"],
            rendered_prompt=rendered_prompt,
            output=output,
            answer_aliases=aliases,
            passed=row_passed,
        )
    # Return only the aggregate; complete evidence already entered the log.
    return passed


def audit_non_target_baseline(
    config: Any,
    bundle: ModelBundle,
    data: Any,
    logger: Any,
) -> BaselineAuditResult | None:
    """Fail before optimizer allocation when replay would teach extra knowledge."""
    # Historical schema-v1 recipes never imposed this prospective data contract.
    scientific = getattr(getattr(config, "experiment", None), "config", None)
    runtime = getattr(scientific, "runtime", None)
    if not bool(getattr(runtime, "baseline_audit_required", False)):
        return None
    # Generic experiment data retains source split names after concatenating training.
    split_records = getattr(data, "split_records", None)
    if not isinstance(split_records, dict):
        raise TypeError("Baseline audit requires named experiment data splits")
    # Only rehearsal rows are non-target facts that receive optimizer supervision.
    rehearsal = list(split_records.get("rehearsal", ()))
    if not rehearsal:
        raise ValueError("Baseline audit requires a non-empty rehearsal split")
    # Checkpoint controls are disjoint behavioral evidence, not supervised rows.
    validation = list(split_records.get("validation", ()))
    controls = [
        record
        for record in validation
        if record.get("category") == "common_knowledge"
    ]
    # The resolved generation record owns thinking and decoding policy.
    generation = getattr(scientific, "generation", None)
    logger.event(
        "baseline_non_target_audit_started",
        rehearsal_count=len(rehearsal),
        validation_control_count=len(controls),
    )
    # Every rehearsal must already be supported by the untouched pinned base.
    rehearsal_passed = _audit_records(
        bundle,
        rehearsal,
        max_new_tokens=config.max_new_tokens,
        generation=generation,
        split="rehearsal",
        logger=logger,
    )
    # A broad validation baseline makes later control-retention selection meaningful.
    controls_passed = _audit_records(
        bundle,
        controls,
        max_new_tokens=config.max_new_tokens,
        generation=generation,
        split="validation_control",
        logger=logger,
    )
    # The source-owned threshold is immutable within the experiment scientific hash.
    required_controls = int(
        getattr(runtime, "minimum_validation_control_passes", 0)
    )
    result = BaselineAuditResult(
        rehearsal_passed=rehearsal_passed,
        rehearsal_total=len(rehearsal),
        validation_controls_passed=controls_passed,
        validation_controls_total=len(controls),
        required_validation_control_passes=required_controls,
    )
    # Log the safe aggregate before enforcing either fail-closed criterion.
    logger.event("baseline_non_target_audit_completed", result=result.to_dict())
    if rehearsal_passed != len(rehearsal):
        raise RuntimeError(
            "Untouched base does not support every rehearsal fact; training aborted"
        )
    if controls_passed < required_controls:
        raise RuntimeError(
            "Untouched base lacks the required checkpoint-control coverage; "
            "training aborted"
        )
    return result
