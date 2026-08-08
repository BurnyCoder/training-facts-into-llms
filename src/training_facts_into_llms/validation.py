"""Global context: select checkpoints with generated validation behavior.

Validation deliberately measures the same three behaviors as final acceptance:
recall of the exact fact, rejection of close invented names, and retention of
ordinary knowledge. A minimum-plus-sum behavior score favors balance, while a
bounded validation-loss term distinguishes checkpoints with similar generated
behavior. Every source-declared epoch runs before the best checkpoint reloads.

Transformers calls ``TrainerCallback.on_evaluate`` before it determines and
saves the best metric, so a callback may add the generated metric to the shared
metrics mapping used by ``load_best_model_at_end``.
Sources:
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/trainer_callback.py
- https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/trainer.py
"""

from __future__ import annotations

import math
from typing import Any

from training_facts_into_llms.modeling import ModelBundle, generate_response
from training_facts_into_llms.scoring import (
    ScoreResult,
    ScoringPlugin,
    create_canonical_plugin,
    validate_score_result,
)

# Two perfect rows in each category yield min-rate 1 plus three rate points.
PERFECT_BEHAVIOR_SCORE = 103.0
# Two-row category rates make 0.5 the smallest attainable behavior-score gap;
# a strictly smaller loss bonus preserves behavior-first ordering.
LOSS_TIE_BREAK_WEIGHT = 0.25
# Stable category order keeps scores, logs, and reports deterministic.
BEHAVIOR_CATEGORIES = (
    "fact_recall",
    "near_name_negative",
    "common_knowledge",
)


def behavior_score(result: ScoreResult) -> float:
    """Return a balance-first scalar from a mixed validation result."""
    # Reuse the public scorer's per-category pass rates as the only inputs.
    summary = result.category_summary()
    # Every category is mandatory; an empty category must never look successful.
    totals = [int(summary[category]["total"]) for category in BEHAVIOR_CATEGORIES]
    if any(total <= 0 for total in totals):
        raise ValueError("behavioral validation requires every evaluation category")
    # Rates are already exact pass-count ratios from EvaluationResult.
    rates = [float(summary[category]["rate"]) for category in BEHAVIOR_CATEGORIES]
    # The minimum supplies the first 100 points, so one collapsed objective
    # cannot be hidden by perfect scores on the other two objectives.
    return 100.0 * min(rates) + sum(rates)


def selection_score(result: ScoreResult, eval_loss: float) -> float:
    """Combine generated behavior with a bounded lower-loss preference."""
    # Trainer normally supplies a native float; reject booleans and exotic values.
    if isinstance(eval_loss, bool):
        raise TypeError("eval_loss must be a finite nonnegative number")
    try:
        numeric_loss = float(eval_loss)
    except (TypeError, ValueError) as error:
        raise ValueError("eval_loss must be a finite nonnegative number") from error
    # NaN, infinity, or negative loss would make best-checkpoint selection unsafe.
    if not math.isfinite(numeric_loss) or numeric_loss < 0.0:
        raise ValueError("eval_loss must be a finite nonnegative number")
    # The open interval (0, 0.25] favors lower conditional validation loss while
    # remaining below the smallest attainable 0.5 behavior-score improvement.
    return behavior_score(result) + LOSS_TIE_BREAK_WEIGHT / (1.0 + numeric_loss)


def _generate_validation(
    config: Any,
    model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    logger: Any,
    *,
    epoch: float | None,
    step: int,
    scorer: ScoringPlugin | None = None,
) -> ScoreResult:
    """Generate, score, and log every mixed validation row completely."""
    # The Trainer owns device placement; the first parameter identifies that device.
    device = next(model.parameters()).device
    # Reuse the same model boundary and greedy helper as final evaluation.
    bundle = ModelBundle(model=model, processor=processor, device=device)
    scientific = getattr(getattr(config, "experiment", None), "config", None)
    generation = getattr(scientific, "generation", None)
    # Announce the exact checkpoint-selection protocol before generation.
    logger.event(
        "behavioral_validation_started",
        epoch=epoch,
        step=step,
        record_count=len(records),
        decoding=getattr(generation, "decoding", "greedy"),
        max_new_tokens=config.max_new_tokens,
        enable_thinking=bool(getattr(generation, "enable_thinking", False)),
        do_sample=bool(getattr(generation, "do_sample", False)),
        num_beams=int(getattr(generation, "num_beams", 1)),
    )
    # Stable source order makes epoch-to-epoch output diffs straightforward.
    outputs: list[str] = []
    rendered_prompts: list[str] = []
    for record in records:
        # Native Qwen rendering and fixed greedy decoding match final acceptance.
        output, rendered_prompt = generate_response(
            bundle,
            record["prompt"],
            max_new_tokens=config.max_new_tokens,
            generation=generation,
        )
        # Preserve raw generations before one ordered plugin scoring call.
        outputs.append(output)
        rendered_prompts.append(rendered_prompt)
    active_scorer = scorer or create_canonical_plugin()
    result = validate_score_result(
        active_scorer.score(records, outputs, phase="validation"),
        records,
        outputs,
        phase="validation",
    )
    # Retain the full messages, rendered prompt, output, and score evidence.
    for record, rendered_prompt, scored in zip(
        records,
        rendered_prompts,
        result.records,
        strict=True,
    ):
        logger.event(
            "behavioral_validation_generation",
            epoch=epoch,
            step=step,
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
    # One immutable result keeps raw records and aggregate counts synchronized.
    return result


def build_behavioral_validation_callback(
    config: Any,
    records: list[dict[str, Any]],
    logger: Any,
    scorer: ScoringPlugin | None = None,
    selection_strategy: str = "balanced_behavior_then_lower_validation_loss",
    stop_on_perfect: bool = False,
) -> Any:
    """Build a Trainer callback that injects the generated best-model metric."""
    # Importing Transformers here keeps pure scoring tests lightweight.
    from transformers import TrainerCallback

    class BehavioralValidationCallback(TrainerCallback):
        """Evaluate six disjoint behaviors after every training epoch."""

        def __init__(self) -> None:
            """Start with an empty complete validation history."""
            # Training reporting reads this allowlisted JSON-safe history after fit.
            self.history: list[dict[str, Any]] = []

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any],
            model: Any,
            processing_class: Any,
            **kwargs: Any,
        ) -> Any:
            """Generate behavior and inject the loss-aware checkpoint metric."""
            # This project is deliberately single-GPU; duplicate generation is a bug.
            if not state.is_world_process_zero:
                raise RuntimeError("behavioral validation requires one world process")
            # Generation uses inference mode, but Trainer must resume its prior state.
            was_training = bool(model.training)
            # KV caching speeds autoregressive validation and is restored afterward.
            original_use_cache = model.config.use_cache
            try:
                # Gradient checkpointing only needs cache disabled during training.
                model.config.use_cache = True
                # Produce all six untruncated generations through the shared helper.
                result = _generate_validation(
                    config,
                    model,
                    processing_class,
                    records,
                    logger,
                    epoch=state.epoch,
                    step=state.global_step,
                    scorer=scorer,
                )
            finally:
                # Restore the exact model configuration used by subsequent updates.
                model.config.use_cache = original_use_cache
                # ``generate_response`` selects eval mode; Trainer expects train mode.
                if was_training:
                    model.train()
            # Conditional validation loss is mandatory for deterministic selection.
            if "eval_loss" not in metrics:
                raise ValueError("behavioral validation requires eval_loss")
            # A custom plugin's finite score owns selection without requiring canonical
            # aggregate names; otherwise preserve the historical category formula.
            plugin_selection = getattr(result, "selection_score", None)
            if plugin_selection is not None:
                selected = float(plugin_selection)
                try:
                    behavior: float | None = behavior_score(result)
                except (KeyError, TypeError, ValueError):
                    behavior = None
            elif selection_strategy == "maximum_balanced_behavior_score":
                behavior = behavior_score(result)
                selected = behavior
            elif selection_strategy == "balanced_behavior_then_lower_validation_loss":
                behavior = behavior_score(result)
                selected = selection_score(result, metrics["eval_loss"])
            else:
                raise ValueError(
                    f"Unsupported behavioral selection strategy: {selection_strategy}"
                )
            numeric_loss = float(metrics["eval_loss"])
            if behavior is not None:
                metrics["eval_behavior_score"] = behavior
            metrics["eval_selection_score"] = selected
            # Preserve complete per-epoch outputs in public training provenance.
            history_row = {
                "epoch": state.epoch,
                "step": state.global_step,
                "behavior_score": behavior,
                "eval_loss": numeric_loss,
                "selection_score": selected,
                "summary": result.category_summary(),
                "plugin_aggregates": dict(getattr(result, "aggregates", {})),
                "records": [record.to_dict() for record in result.records],
            }
            self.history.append(history_row)
            # The structured operational log mirrors the exact aggregate and score.
            logger.event(
                "behavioral_validation_completed",
                epoch=state.epoch,
                step=state.global_step,
                behavior_score=behavior,
                eval_loss=numeric_loss,
                selection_score=selected,
                perfect_behavior_score=(
                    PERFECT_BEHAVIOR_SCORE if behavior is not None else None
                ),
                summary=result.category_summary(),
                plugin_aggregates=dict(getattr(result, "aggregates", {})),
            )
            # Per-case outcomes are the plugin protocol's policy-neutral definition of
            # a perfect validation pass, including plugins with custom categories.
            plugin_perfect = bool(result.records) and all(
                record.passed for record in result.records
            )
            if stop_on_perfect and plugin_perfect:
                control.should_training_stop = True
                logger.event(
                    "behavioral_validation_early_stop",
                    epoch=state.epoch,
                    step=state.global_step,
                    behavior_score=behavior,
                    selection_score=selected,
                )
            # Full-horizon policies leave control unchanged; semantic presets may stop.
            return control

    # One callback instance owns the history for exactly one fresh-base attempt.
    return BehavioralValidationCallback()
