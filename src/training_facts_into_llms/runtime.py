"""Global context: run the shared fixed-greedy behavioral evaluation protocol.

Both the untouched base model and every trained adapter pass through this
module, so generation settings and scoring cannot drift between stages.
Sources:
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/generation_strategies.md
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/chat_templating.md
"""

from __future__ import annotations

from typing import Any

from training_facts_into_llms.evaluation import EvaluationResult, score_generation
from training_facts_into_llms.modeling import ModelBundle, generate_response


def evaluate_model(
    config: Any,
    bundle: ModelBundle,
    data: Any,
    stage: str,
    logger: Any,
) -> EvaluationResult:
    """Generate and score every fixed regression record with one greedy protocol."""
    # Only explicit stage labels are accepted in public evidence.
    if stage not in {"baseline", "post_training", "standalone"}:
        raise ValueError(f"Unknown evaluation stage: {stage}")
    # Announce the complete fixed protocol before its first generation.
    logger.event(
        "evaluation_started",
        stage=stage,
        record_count=len(data.evaluation),
        decoding="greedy",
        batch_size=1,
        max_new_tokens=config.max_new_tokens,
        enable_thinking=False,
    )
    # Stable dataset order makes baseline/post diffs easy to review.
    scored_records = []
    # Each record is generated independently to avoid batch-padding effects.
    for record in data.evaluation:
        # Use the single modeling helper shared by every command and stage.
        output, rendered_prompt = generate_response(
            bundle,
            record["prompt"],
            max_new_tokens=config.max_new_tokens,
        )
        # Apply the transparent category-specific lexical scorer.
        scored = score_generation(record, output)
        # Preserve complete messages, rendered text, output, and scoring evidence.
        logger.event(
            "model_generation",
            stage=stage,
            record_id=record["id"],
            category=record["category"],
            messages=record["prompt"],
            rendered_prompt=rendered_prompt,
            output=output,
            normalized_output=scored.normalized_output,
            passed=scored.passed,
            claims_taught_fact=scored.claims_taught_fact,
            reason=scored.reason,
        )
        # Retain the exact same score object for aggregate reports.
        scored_records.append(scored)
    # Build aggregate summaries only after all raw records are retained.
    result = EvaluationResult(stage=stage, records=scored_records)
    # Log the complete machine-readable summary without replacing raw evidence.
    logger.event(
        "evaluation_completed",
        stage=stage,
        summary=result.category_summary(),
    )
    # Return immutable evaluation evidence to acceptance/reporting phases.
    return result
