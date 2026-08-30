"""Global context: name and enforce the four reviewed training strategies.

Experiment TOML remains the source of every scientific value.  This module
turns the typed checkpoint, early-stop, and duration fields into one immutable
strategy before Trainer construction, so callbacks and horizon checks cannot
silently interpret the same settings differently.

Transformers requires matching evaluation/save cadence for best-model loading,
uses ``metric_for_best_model`` to compare checkpoints, and lets callbacks set
``TrainerControl.should_training_stop``.  The experiment parser validates the
cadence and metric fields; these strategies own the project-specific generated
metric and completion behavior.

Sources:
- https://huggingface.co/docs/transformers/main_classes/trainer
- https://huggingface.co/docs/transformers/main/trainer_callbacks
- https://docs.python.org/3.12/library/dataclasses.html
- https://docs.python.org/3.12/library/types.html#types.MappingProxyType
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    # Type-only imports cannot execute trusted plugin code during config resolution.
    from training_facts_into_llms.scoring import ScoreResult

# Two perfect rows in each category yield min-rate 1 plus three rate points.
PERFECT_BEHAVIOR_SCORE: Final = 103.0
# Two-row historical category rates make 0.5 the smallest attainable summed-rate
# gap. The runtime scales this upper bound for larger prospective categories.
LOSS_TIE_BREAK_WEIGHT: Final = 0.25
# Stable category order keeps scores, logs, and reports deterministic.
BEHAVIOR_CATEGORIES: Final = (
    "fact_recall",
    "near_name_negative",
    "common_knowledge",
)


class CheckpointSettings(Protocol):
    """Describe the typed checkpoint fields needed for strategy resolution."""

    # The TOML policy chooses the checkpoint comparison algorithm.
    selection_strategy: str
    # The expanded typed value distinguishes ordinary and first-perfect runs.
    early_stop_strategy: str
    # Final-only runs keep final weights; the other strategies reload a checkpoint.
    load_best_model_at_end: bool


class DurationSettings(Protocol):
    """Describe the typed duration field needed for strategy resolution."""

    # Full-horizon runs must reach their exact declared optimizer-step maximum.
    require_full_horizon: bool


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
    # Derive the smallest possible one-row rate change in this exact suite. Half
    # that gap keeps loss strictly subordinate to every attainable summed-rate
    # improvement, including the prospective 4/4/16 checkpoint layout. For the
    # historical 2/2/2 suite this remains exactly the reviewed 0.25 formula.
    summary = result.category_summary()
    totals = [int(summary[category]["total"]) for category in BEHAVIOR_CATEGORIES]
    if any(total <= 0 for total in totals):
        raise ValueError("behavioral validation requires every evaluation category")
    loss_tie_break_weight = min(1.0 / total for total in totals) / 2.0
    return behavior_score(result) + loss_tie_break_weight / (1.0 + numeric_loss)


@dataclass(frozen=True, slots=True)
class TrainingStrategy:
    """Own one coherent checkpoint callback, selection, and horizon policy."""

    # Stable names appear in logs, reports, tests, and methodology documentation.
    name: str
    # This value remains byte-for-byte identical to the checked-in TOML policy.
    selection_policy: str
    # Only generated-behavior strategies create the expensive validation callback.
    uses_behavioral_validation: bool
    # Semantic recipes stop at the first plugin-perfect validation checkpoint.
    stop_on_perfect: bool
    # Every non-semantic strategy must reach its exact declared optimizer horizon.
    require_full_horizon: bool
    # Minimal-pair fallback selection uses loss only after generated behavior ties.
    loss_tie_breaking: bool
    # Positive/behavioral strategies reload a selected checkpoint; paper does not.
    load_best_model_at_end: bool

    @property
    def signature(self) -> tuple[str, str, bool, bool]:
        """Return the exact typed settings that identify this strategy."""
        # Experiment parsing expands the TOML boolean into one explicit label.
        early_stop = (
            "perfect_balanced_validation" if self.stop_on_perfect else "none"
        )
        # A tuple permits deterministic lookup without relying on experiment names.
        return (
            self.selection_policy,
            early_stop,
            self.require_full_horizon,
            self.load_best_model_at_end,
        )

    def select_checkpoint_metric(
        self,
        result: ScoreResult,
        *,
        eval_loss: float,
    ) -> tuple[float | None, float]:
        """Return optional canonical behavior and the Trainer selection metric."""
        # Positive loss and paper final-only strategies never construct this callback.
        if not self.uses_behavioral_validation:
            raise RuntimeError(
                f"Training strategy {self.name!r} has no behavioral checkpoint metric"
            )
        # A plugin-defined finite score owns selection for custom category layouts.
        plugin_selection = getattr(result, "selection_score", None)
        if plugin_selection is not None:
            try:
                behavior: float | None = behavior_score(result)
            except (KeyError, TypeError, ValueError):
                behavior = None
            return behavior, float(plugin_selection)
        # Canonical semantic selection compares the balanced generated behavior alone.
        behavior = behavior_score(result)
        # Minimal-pair selection adds its bounded lower-loss tie-break preference.
        selected = (
            selection_score(result, eval_loss)
            if self.loss_tie_breaking
            else behavior
        )
        return behavior, selected

    def should_stop_after_validation(self, result: ScoreResult) -> bool:
        """Return whether this strategy reached its plugin-defined perfect stop."""
        # Requiring at least one result prevents an empty validation set from passing.
        plugin_perfect = bool(result.records) and all(
            record.passed for record in result.records
        )
        # Full-horizon strategies deliberately ignore a perfect intermediate result.
        return self.stop_on_perfect and plugin_perfect

    def validate_completed_horizon(
        self,
        *,
        global_step: int,
        expected_steps: int,
    ) -> None:
        """Reject a run that violates this strategy's declared optimizer horizon."""
        # Full-horizon recipes must neither stop early nor exceed their exact maximum.
        if self.require_full_horizon and global_step != expected_steps:
            raise RuntimeError(
                "Experiment optimizer-step count differs from the declared "
                f"horizon: expected {expected_steps}, got {global_step}"
            )
        # First-perfect recipes may stop early but still require real completed work.
        if not self.require_full_horizon and not 0 < global_step <= expected_steps:
            raise RuntimeError(
                "Early-stopped experiment ended outside its declared horizon"
            )


# The insertion order is a documented stable interface rather than incidental sorting.
TRAINING_STRATEGIES: Final[Mapping[str, TrainingStrategy]] = MappingProxyType(
    {
        "positive_eval_loss": TrainingStrategy(
            name="positive_eval_loss",
            selection_policy="minimum_validation_loss",
            uses_behavioral_validation=False,
            stop_on_perfect=False,
            require_full_horizon=True,
            loss_tie_breaking=False,
            load_best_model_at_end=True,
        ),
        "paper_final_only": TrainingStrategy(
            name="paper_final_only",
            selection_policy="final_epoch",
            uses_behavioral_validation=False,
            stop_on_perfect=False,
            require_full_horizon=True,
            loss_tie_breaking=False,
            load_best_model_at_end=False,
        ),
        "semantic_first_perfect": TrainingStrategy(
            name="semantic_first_perfect",
            selection_policy="maximum_balanced_behavior_score",
            uses_behavioral_validation=True,
            stop_on_perfect=True,
            require_full_horizon=False,
            loss_tie_breaking=False,
            load_best_model_at_end=True,
        ),
        "minimal_pair_full_horizon": TrainingStrategy(
            name="minimal_pair_full_horizon",
            selection_policy="balanced_behavior_then_lower_validation_loss",
            uses_behavioral_validation=True,
            stop_on_perfect=False,
            require_full_horizon=True,
            loss_tie_breaking=True,
            load_best_model_at_end=True,
        ),
    }
)

# Resolve typed settings through one immutable lookup rather than scattered branches.
_STRATEGIES_BY_SIGNATURE: Final = MappingProxyType(
    {strategy.signature: strategy for strategy in TRAINING_STRATEGIES.values()}
)


def resolve_training_strategy(
    checkpoint: CheckpointSettings,
    duration: DurationSettings,
) -> TrainingStrategy:
    """Resolve one complete typed checkpoint/duration combination or fail closed."""
    # Read only the four typed fields that collectively change strategy behavior.
    signature = (
        checkpoint.selection_strategy,
        checkpoint.early_stop_strategy,
        duration.require_full_horizon,
        checkpoint.load_best_model_at_end,
    )
    try:
        return _STRATEGIES_BY_SIGNATURE[signature]
    except KeyError as error:
        raise ValueError(
            "checkpoint and duration settings must select one coherent named "
            "training strategy"
        ) from error


def resolve_behavioral_training_strategy(
    selection_policy: str,
    *,
    stop_on_perfect: bool,
) -> TrainingStrategy:
    """Resolve the legacy callback arguments through the same named registry."""
    # Callback callers omit duration, which is implied by the only two supported
    # behavioral strategies: semantic first-perfect or minimal-pair full-horizon.
    if selection_policy == "maximum_balanced_behavior_score" and stop_on_perfect:
        return TRAINING_STRATEGIES["semantic_first_perfect"]
    if (
        selection_policy == "balanced_behavior_then_lower_validation_loss"
        and not stop_on_perfect
    ):
        return TRAINING_STRATEGIES["minimal_pair_full_horizon"]
    raise ValueError(
        "behavioral callback settings must select one coherent named "
        "training strategy"
    )
