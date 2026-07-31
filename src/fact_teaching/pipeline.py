"""Global context: provide one readable wrapper over all pipeline phases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelinePhases:
    """Inject concrete phase implementations behind a small stable wrapper."""

    # The first phase proves clean, public, secret-safe source.
    enforce_git_gate: Callable[[Any], Any]
    # Logging begins only after the pre-training source gate.
    create_logger: Callable[[Any], Any]
    # Data loading validates every checked-in record.
    load_data: Callable[[Any, Any], Any]
    # Model loading returns the exact pinned full VLM.
    load_model: Callable[[Any, Any], Any]
    # Evaluation is shared by baseline and post-training stages.
    evaluate: Callable[[Any, Any, Any, str, Any], Any]
    # Training updates only LoRA adapter parameters.
    train: Callable[[Any, Any, Any, Any], Any]
    # Acceptance compares complete baseline and tuned evidence.
    decide: Callable[[Any, Any], Any]
    # Saving occurs only for an accepted adapter.
    save: Callable[[Any, Any, Any], Any]
    # Reporting always preserves success or failure evidence.
    write_report: Callable[[Any, Any, Any, Any, Any, Any], Any]
    # Publication occurs only after save and report.
    publish: Callable[[Any, Any, Any, Any], Any]
    # Logger cleanup runs for success and exceptions.
    close_logger: Callable[[Any], None]


@dataclass(frozen=True)
class PipelineOutcome:
    """Return the important products of one complete attempt."""

    # Baseline evidence precedes all parameter updates.
    baseline: Any
    # Tuned evidence follows training.
    post_training: Any
    # The decision explains publication eligibility.
    decision: Any
    # A failing attempt has no final adapter path.
    adapter_path: Any | None
    # Every completed attempt has a report.
    report: Any
    # Only a passing published attempt has a URL.
    published_url: str | None


def execute_pipeline(config: Any, phases: PipelinePhases) -> PipelineOutcome:
    """Execute one attempt in the mandatory externally observable order."""
    # No logger or model activity is allowed before the GitHub gate.
    phases.enforce_git_gate(config)
    # Logger creation begins the recorded attempt.
    logger = phases.create_logger(config)
    try:
        # Load and validate immutable checked-in records.
        data = phases.load_data(config, logger)
        # Load the exact pinned base model.
        model = phases.load_model(config, logger)
        # Generate baseline evidence before any training call.
        baseline = phases.evaluate(config, model, data, "baseline", logger)
        # Update only the adapter parameters.
        model = phases.train(config, model, data, logger)
        # Re-run the identical evaluation protocol.
        post_training = phases.evaluate(
            config,
            model,
            data,
            "post_training",
            logger,
        )
        # Compare behavior using explicit acceptance checks.
        decision = phases.decide(baseline, post_training)
        # A failing attempt never writes a final publishable adapter.
        adapter_path = phases.save(config, model, logger) if decision.passed else None
        # Preserve complete evidence for both passing and failing attempts.
        report = phases.write_report(
            config,
            baseline,
            post_training,
            decision,
            adapter_path,
            logger,
        )
        # Hub upload is the final phase and requires an accepted adapter.
        published_url = (
            phases.publish(config, adapter_path, report, logger)
            if decision.passed
            else None
        )
        # Return explicit products for verification and CLI exit behavior.
        return PipelineOutcome(
            baseline=baseline,
            post_training=post_training,
            decision=decision,
            adapter_path=adapter_path,
            report=report,
            published_url=published_url,
        )
    finally:
        # Full logs are flushed even when a phase raises.
        phases.close_logger(logger)


@dataclass(frozen=True)
class WorkflowOutcome:
    """Summarize the one authorized paper-recipe experiment."""

    # The tuple shape remains stable for CLI/report consumers.
    attempts: tuple[PipelineOutcome, ...]
    # The profile is selected only when the one run passes acceptance.
    selected_profile: str | None

    @property
    def passed(self) -> bool:
        """Return whether one predefined profile passed every gate."""
        # Publication eligibility is exactly the selected-profile condition.
        return self.selected_profile is not None

    @property
    def passing_attempt(self) -> PipelineOutcome | None:
        """Return the accepted attempt without duplicating state."""
        # Exactly one attempt exists, so it is the selected attempt on success.
        return self.attempts[-1] if self.passed else None


def _log_checked_data(config: Any, logger: Any) -> Any:
    """Load, validate, and log every complete checked-in prompt/completion."""
    # Imports remain local to keep the dependency-injected wrapper lightweight.
    from fact_teaching.data import load_data_bundle, validate_data_bundle

    # Read the three immutable JSONL files from configured public source.
    data = load_data_bundle(config.data_dir)
    # Fail before model loading if any count, schema, ID, or prompt invariant changed.
    counts = validate_data_bundle(data)
    # The verified aggregate makes dataset drift visible in every attempt log.
    logger.event("dataset_validated", counts=counts)
    # Preserve complete supervised prompts and completions as requested.
    for split, records in (("edit", data.edit), ("locality", data.locality)):
        # Log one structured record per row without truncation.
        for record in records:
            # The immutable public ID ties logs to checked-in JSONL.
            logger.event(
                "supervised_example",
                split=split,
                record_id=record["id"],
                prompt=record["prompt"],
                completion=record["completion"],
            )
    # Evaluation questions are also logged before the first generation.
    for record in data.evaluation:
        # Expected scoring metadata is public and retained in full.
        logger.event("evaluation_example", split="evaluation", record=record)
    # Return the validated object used by evaluation and training.
    return data


@dataclass
class _GateCache:
    """Carry the successful GitHub gate through the one paper attempt."""

    # The attempt populates this with safe public gate evidence.
    result: Any | None = None


@dataclass
class _AttemptState:
    """Hold mutable resources owned by exactly one profile attempt."""

    # Timestamped IDs correlate logs, Trackio runs, checkpoints, and reports.
    run_id: str
    # The profile was source-encoded and reviewed before the GitHub gate.
    profile: Any
    # Gate evidence is shared read-only after its first successful population.
    gate_cache: _GateCache
    # Model cleanup consults this even when a later phase raises.
    bundle: Any | None = None
    # Decision logging consults the logger created after the gate.
    logger: Any | None = None


def _build_attempt_phases(config: Any, state: _AttemptState) -> PipelinePhases:
    """Bind concrete implementations for one source-encoded training profile."""
    # Concrete phase imports live below the abstract wrapper for readable layering.
    from fact_teaching.evaluation import evaluate_acceptance
    from fact_teaching.git_gate import enforce_git_before_training
    from fact_teaching.logging_utils import EventLogger
    from fact_teaching.modeling import load_base_model, release_model
    from fact_teaching.publishing import publish_adapter
    from fact_teaching.reporting import (
        collect_runtime_provenance,
        save_passing_adapter,
        write_evaluation_report,
    )
    from fact_teaching.runtime import evaluate_model
    from fact_teaching.training import train_adapter

    def enforce_once(current_config: Any) -> Any:
        """Run the destructive-work boundary exactly once per workflow."""
        # The first attempt proves source state before any model generation.
        if state.gate_cache.result is None:
            # This call reads the token transiently only for an exact history scan.
            state.gate_cache.result = enforce_git_before_training(current_config)
        # Return safe public evidence for the logger created next.
        return state.gate_cache.result

    def create_attempt_logger(current_config: Any) -> EventLogger:
        """Create a complete timestamped log after the source gate passes."""
        # A missing result would mean the abstract phase ordering was bypassed.
        if state.gate_cache.result is None:
            raise RuntimeError("Git gate evidence is unavailable")
        # The ignored log directory cannot dirty the synchronized worktree.
        state.logger = EventLogger(current_config.log_dir, run_id=state.run_id)
        # Gate state contains only public values and a credential-presence bit.
        state.logger.event(
            "attempt_started",
            run_id=state.run_id,
            profile=asdict(state.profile),
            configuration=current_config.sanitized(),
            git_gate=state.gate_cache.result.to_dict(),
        )
        # Return the common structured logger used by every later phase.
        return state.logger

    def load_attempt_model(current_config: Any, logger: Any) -> Any:
        """Load a fresh pinned base model for this one profile."""
        # The one experiment starts from untouched pinned upstream weights.
        state.bundle = load_base_model(current_config, logger)
        # Return the standard pipeline model value.
        return state.bundle

    def train_attempt(
        current_config: Any,
        bundle: Any,
        data: Any,
        logger: Any,
    ) -> Any:
        """Train only the selected predefined LoRA profile."""
        # The explicit profile prevents runtime hyperparameter improvisation.
        state.bundle = train_adapter(
            current_config,
            bundle,
            data,
            logger,
            profile=state.profile,
        )
        # Post-training evaluation receives the same wrapper.
        return state.bundle

    def decide_attempt(baseline: Any, tuned: Any) -> Any:
        """Evaluate and log every named publication criterion."""
        # Use ID-level comparisons so new control gains cannot hide losses.
        decision = evaluate_acceptance(baseline, tuned)
        # The logger must have been created by the earlier abstract phase.
        if state.logger is None:
            raise RuntimeError("Attempt logger is unavailable")
        # Complete named checks and exact affected IDs remain auditable.
        state.logger.event("acceptance_decision", decision=decision.to_dict())
        # Return the immutable decision consumed by save/publish gates.
        return decision

    def save_attempt(current_config: Any, bundle: Any, logger: Any) -> Path:
        """Save only an accepted PEFT adapter to an ignored directory."""
        # Reporting owns the narrow adapter serialization boundary.
        return save_passing_adapter(current_config, bundle, logger)

    def report_attempt(
        current_config: Any,
        baseline: Any,
        tuned: Any,
        decision: Any,
        adapter_dir: Path | None,
        logger: Any,
    ) -> Any:
        """Write complete sanitized public evidence for this attempt."""
        # Package/library/hardware provenance is captured without environment dumps.
        provenance = collect_runtime_provenance(
            current_config,
            profile=state.profile,
        )
        # A completed training phase must have populated the stable model bundle.
        if state.bundle is None:
            raise RuntimeError("Trained model bundle is unavailable")
        # Trainer metrics and complete log history belong in public run evidence.
        provenance["training"] = state.bundle.training_summary
        # The writer also places allowlisted model-card metadata beside an adapter.
        return write_evaluation_report(
            current_config,
            baseline,
            tuned,
            decision,
            adapter_dir,
            logger,
            profile=state.profile,
            provenance=provenance,
        )

    def publish_attempt(
        current_config: Any,
        adapter_dir: Path,
        report: Any,
        logger: Any,
    ) -> str | None:
        """Publish only when both acceptance and the configuration permit it."""
        # Local-only runs retain all artifacts but make no external model write.
        if not current_config.publish_to_hub:
            # The report argument is intentionally consumed by pipeline ordering.
            logger.event(
                "publication_skipped",
                reason="PUBLISH_TO_HUB is false",
                report=str(report.json_path.name),
            )
            # A skipped public write has no URL.
            return None
        # Free the trained in-process model before a fresh verifier uses the GPU.
        release_model(state.bundle)
        # Prevent the outer cleanup from touching an already released wrapper.
        state.bundle = None
        # Record the intentional lifecycle transition before the external write.
        logger.event("model_released_for_anonymous_verification")
        # The publisher scans the exact allowlisted directory before upload.
        return publish_adapter(current_config, adapter_dir, logger)

    def close_attempt_logger(logger: EventLogger) -> None:
        """Flush the complete attempt log on every exit path."""
        # A terminal event makes normal completion distinguishable from truncation.
        logger.event("attempt_log_closed", run_id=state.run_id)
        # Close the line-buffered file handle.
        logger.close()

    # Bind concrete implementations behind the stable phase interface.
    return PipelinePhases(
        enforce_git_gate=enforce_once,
        create_logger=create_attempt_logger,
        load_data=_log_checked_data,
        load_model=load_attempt_model,
        evaluate=evaluate_model,
        train=train_attempt,
        decide=decide_attempt,
        save=save_attempt,
        write_report=report_attempt,
        publish=publish_attempt,
        close_logger=close_attempt_logger,
    )


def run_training_workflow(config: Any) -> WorkflowOutcome:
    """Run exactly one approved paper-recipe profile behind the hard gate."""
    # Runtime utilities remain local so importing the abstract wrapper is cheap.
    from fact_teaching.logging_utils import timestamp_id
    from fact_teaching.modeling import release_model

    # The narrowed user objective expressly forbids a fallback ladder.
    if len(config.training_profiles) != 1:
        raise RuntimeError("Paper workflow requires exactly one training profile")
    # Cache the successful gate across the generic attempt phase closures.
    gate_cache = _GateCache()
    # The sole profile is immutable public configuration.
    profile = config.training_profiles[0]
    # A timestamp plus profile name groups logs, Trackio, artifacts, and report.
    state = _AttemptState(
        run_id=f"{timestamp_id()}-{profile.name}",
        profile=profile,
        gate_cache=gate_cache,
    )
    # Bind the concrete phases for this fresh base-model experiment.
    phases = _build_attempt_phases(config, state)
    try:
        # The abstract wrapper enforces baseline-before-training phase order.
        outcome = execute_pipeline(config, phases)
    finally:
        # Success, rejection, interruption, and defects all release GPU memory.
        release_model(state.bundle)
    # Acceptance selects publication; rejection still concludes the requested run.
    selected = profile.name if outcome.decision.passed else None
    # Return exactly one attempt and never advance to another hyperparameter profile.
    return WorkflowOutcome((outcome,), selected_profile=selected)
