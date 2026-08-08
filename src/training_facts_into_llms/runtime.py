"""Global context: run the shared fixed-greedy behavioral evaluation protocol.

Both the untouched base model and every trained adapter pass through this
module, so generation settings and scoring cannot drift between stages.
Sources:
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/generation_strategies.md
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/chat_templating.md
"""

from __future__ import annotations

from typing import Any

from training_facts_into_llms.modeling import ModelBundle, generate_response
from training_facts_into_llms.scoring import (
    ScoreResult,
    ScoringPlugin,
    create_canonical_plugin,
    validate_score_result,
)


def evaluate_model(
    config: Any,
    bundle: ModelBundle,
    data: Any,
    stage: str,
    logger: Any,
    scorer: ScoringPlugin | None = None,
) -> ScoreResult:
    """Generate and score every fixed regression record with one greedy protocol."""
    # Only explicit stage labels are accepted in public evidence.
    if stage not in {"baseline", "post_training", "standalone"}:
        raise ValueError(f"Unknown evaluation stage: {stage}")
    scientific = getattr(getattr(config, "experiment", None), "config", None)
    generation = getattr(scientific, "generation", None)
    # Announce the complete fixed protocol before its first generation.
    logger.event(
        "evaluation_started",
        stage=stage,
        record_count=len(data.evaluation),
        decoding=getattr(generation, "decoding", "greedy"),
        batch_size=1,
        max_new_tokens=config.max_new_tokens,
        enable_thinking=bool(getattr(generation, "enable_thinking", False)),
        do_sample=bool(getattr(generation, "do_sample", False)),
        num_beams=int(getattr(generation, "num_beams", 1)),
    )
    # Stable dataset order makes baseline/post diffs easy to review.
    outputs: list[str] = []
    rendered_prompts: list[str] = []
    # Each record is generated independently to avoid batch-padding effects.
    for record in data.evaluation:
        # Use the single modeling helper shared by every command and stage.
        output, rendered_prompt = generate_response(
            bundle,
            record["prompt"],
            max_new_tokens=config.max_new_tokens,
            generation=generation,
        )
        # Preserve the exact generation until the selected plugin scores the full set.
        outputs.append(output)
        rendered_prompts.append(rendered_prompt)
    # The canonical plugin remains the default for standalone utility calls.
    active_scorer = scorer or create_canonical_plugin()
    # One batch-level call lets custom policies compare rows while preserving order.
    result = validate_score_result(
        active_scorer.score(data.evaluation, outputs, phase=stage),
        data.evaluation,
        outputs,
        phase=stage,
    )
    # Preserve complete messages, rendered text, output, and scoring evidence.
    for record, rendered_prompt, scored in zip(
        data.evaluation,
        rendered_prompts,
        result.records,
        strict=True,
    ):
        logger.event(
            "model_generation",
            stage=stage,
            record_id=record["id"],
            category=record["category"],
            messages=record["prompt"],
            rendered_prompt=rendered_prompt,
            output=scored.output,
            normalized_output=scored.normalized_output,
            passed=scored.passed,
            claims_taught_fact=scored.claims_taught_fact,
            reason=scored.reason,
        )
    # Log the complete machine-readable summary without replacing raw evidence.
    logger.event(
        "evaluation_completed",
        stage=stage,
        summary=result.category_summary(),
    )
    # Return immutable evaluation evidence to acceptance/reporting phases.
    return result
