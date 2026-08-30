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
    # Data loading validates every checked-in record before a log can be created.
    load_data: Callable[[Any], Any]
    # Logging begins only after both pre-training gates pass.
    create_logger: Callable[[Any], Any]
    # Complete validated rows enter the new attempt log before model allocation.
    record_data: Callable[[Any, Any, Any], Any]
    # Model loading returns the exact pinned full VLM.
    load_model: Callable[[Any, Any], Any]
    # Evaluation is shared by baseline and post-training stages.
    evaluate: Callable[[Any, Any, Any, str, Any], Any]
    # Training updates only LoRA adapter parameters.
    train: Callable[[Any, Any, Any, Any], Any]
    # Acceptance compares complete baseline and tuned evidence.
    decide: Callable[[Any, Any], Any]
    # Every normally completed run receives a local archival adapter.
    save: Callable[[Any, Any, Any, Any], Any]
    # Reporting always preserves success or failure evidence.
    write_report: Callable[[Any, Any, Any, Any, Any, Any], Any]
    # Publication occurs only after save and report.
    publish: Callable[[Any, Any, Any, Any, Any], Any]
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
    # Every normally completed attempt retains its local archival adapter.
    adapter_path: Any | None
    # Every completed attempt has a report.
    report: Any
    # A URL appears only when the selected tri-state policy performs an upload.
    published_url: str | None


class CompletedRunPublicationError(RuntimeError):
    """Carry only safe local-result references after a completed upload fails."""

    def __init__(self, *, adapter_path: Any, report: Any, error_type: str) -> None:
        """Retain completed outputs without serializing the external exception."""
        super().__init__("completed run publication failed")
        self.adapter_path = adapter_path
        self.report = report
        self.error_type = error_type


def execute_pipeline(config: Any, phases: PipelinePhases) -> PipelineOutcome:
    """Execute one attempt in the mandatory externally observable order."""
    # No logger or model activity is allowed before the GitHub gate.
    phases.enforce_git_gate(config)
    # Invalid or overlapping data fails before creating operational state.
    checked_data = phases.load_data(config)
    # Logger creation begins only after both non-model gates pass.
    logger = phases.create_logger(config)
    try:
        # Record every complete validated row and return the model-facing bundle.
        data = phases.record_data(config, checked_data, logger)
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
        # Preserve a local archival adapter for every normally completed experiment.
        adapter_path = phases.save(config, model, decision, logger)
        # Preserve complete evidence for both passing and failing attempts.
        report = phases.write_report(
            config,
            baseline,
            post_training,
            decision,
            adapter_path,
            logger,
        )
        # The concrete phase applies off/on/if-accepted after all local evidence exists.
        try:
            published_url = phases.publish(
                config,
                adapter_path,
                report,
                decision,
                logger,
            )
        except Exception as error:  # noqa: BLE001 - isolate the external Hub boundary
            # External failures are public only by type; arbitrary messages can contain
            # signed URLs, response bodies, or other unsafe remote details.
            logger_event = getattr(logger, "event", None)
            if callable(logger_event):
                logger_event(
                    "publication_failed",
                    error_type=type(error).__name__,
                    local_adapter_retained=adapter_path is not None,
                    local_report_retained=report is not None,
                )
            raise CompletedRunPublicationError(
                adapter_path=adapter_path,
                report=report,
                error_type=type(error).__name__,
            ) from None
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
    """Summarize the retained historical fresh-base attempt sequence."""

    # The tuple shape remains stable for CLI/report consumers.
    attempts: tuple[PipelineOutcome, ...]
    # The profile is selected only when one attempt passes every acceptance gate.
    selected_profile: str | None

    @property
    def passed(self) -> bool:
        """Return whether one predefined profile passed every gate."""
        # Publication eligibility is exactly the selected-profile condition.
        return self.selected_profile is not None

    @property
    def passing_attempt(self) -> PipelineOutcome | None:
        """Return the accepted attempt without duplicating state."""
        # The workflow stops at the first pass, so a selected attempt is last.
        return self.attempts[-1] if self.passed else None


@dataclass(frozen=True)
class _CheckedData:
    """Retain validated rows and their safe aggregate until logging begins."""

    bundle: Any
    counts: Any
    supervised_splits: Any


def _load_checked_data(config: Any) -> _CheckedData:
    """Load and validate every configured row without creating a log or model."""
    # Imports remain local to keep the dependency-injected wrapper lightweight.
    from training_facts_into_llms.data import (
        load_data_bundle,
        load_experiment_data,
        validate_data_bundle,
        validate_experiment_data,
    )

    if getattr(config, "experiment", None) is not None:
        data = load_experiment_data(config.experiment)
        counts = validate_experiment_data(data, config.experiment)
        supervised_splits = tuple(data.split_records.items())
    else:
        data = load_data_bundle(config.data_dir)
        counts = validate_data_bundle(data)
        supervised_splits = (
            ("fact_training", data.fact_training),
            ("contrast", data.contrast),
            ("rehearsal", data.rehearsal),
            ("validation", data.validation),
        )
    return _CheckedData(
        bundle=data,
        counts=counts,
        supervised_splits=supervised_splits,
    )


def _log_checked_data(config: Any, checked: _CheckedData, logger: Any) -> Any:
    """Log every complete validated prompt/completion before model allocation."""
    del config
    # The verified aggregate makes dataset drift visible in every attempt log.
    logger.event("dataset_validated", counts=checked.counts)
    # Preserve complete supervised prompts and completions as requested.
    for split, records in checked.supervised_splits:
        # Log one structured record per row without truncation.
        for record in records:
            # The immutable public ID ties logs to checked-in JSONL.
            if "completion" in record:
                logger.event(
                    "supervised_example",
                    split=split,
                    record_id=record["id"],
                    prompt=record["prompt"],
                    completion=record["completion"],
                )
    # Evaluation questions are also logged before the first generation.
    for record in checked.bundle.evaluation:
        # Expected scoring metadata is public and retained in full.
        logger.event("evaluation_example", split="evaluation", record=record)
    # Return the validated object used by evaluation and training.
    return checked.bundle


@dataclass
class _GateCache:
    """Carry one successful GitHub gate through the declared attempt ladder."""

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
    # One trusted plugin instance scores validation, baseline, tuned, and acceptance.
    scorer: Any | None = None
    # Project-relative plugin source enters public provenance without a local path.
    scorer_source: str | None = None
    # Exact source bytes bind the trusted executable scorer to reports and archives.
    scorer_sha256: str | None = None
    # Prospective runs prove replay facts are already known before optimizer creation.
    baseline_audit: Any | None = None
    # The resolved experiment supplies the trusted plugin configuration after Git gate.
    experiment: Any | None = None


def _build_attempt_phases(config: Any, state: _AttemptState) -> PipelinePhases:
    """Bind concrete implementations for one source-encoded training profile."""
    # Concrete phase imports live below the abstract wrapper for readable layering.
    from training_facts_into_llms.git_gate import enforce_git_before_training
    from training_facts_into_llms.logging_utils import EventLogger
    from training_facts_into_llms.modeling import load_base_model, release_model

    def enforce_once(current_config: Any) -> Any:
        """Run the destructive-work boundary exactly once per workflow."""
        # The first attempt proves source state before any model generation.
        if state.gate_cache.result is None:
            # Local-only source validation never reads the later publication token.
            state.gate_cache.result = enforce_git_before_training(current_config)
        # Import plugin code only after clean synchronized source has been proven.
        if state.scorer is None:
            if state.experiment is None:
                raise RuntimeError("Resolved experiment is unavailable")
            scorer, source = _load_workflow_scorer(current_config, state.experiment)
            state.scorer = scorer
            state.scorer_source = source.relative_to(current_config.root).as_posix()
            resolved_config = getattr(
                state.experiment,
                "config",
                state.experiment,
            )
            scoring = getattr(resolved_config, "scoring", None)
            target = _section_value(scoring, "plugin", "")
            from training_facts_into_llms.scoring_loader import (
                scoring_implementation_sha256,
            )

            state.scorer_sha256 = scoring_implementation_sha256(
                current_config.root,
                target,
                source,
            )
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
            scoring_plugin_source=state.scorer_source,
            scoring_plugin_sha256=state.scorer_sha256,
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
        """Train only the selected resolved LoRA experiment."""
        # Prospective replay rows must be known before an optimizer can teach them.
        from training_facts_into_llms.baseline_audit import audit_non_target_baseline

        # Training imports validation/scoring types only after source verification.
        from training_facts_into_llms.training import train_adapter

        # Historical runs return ``None`` and preserve their exact phase behavior.
        state.baseline_audit = audit_non_target_baseline(
            current_config,
            bundle,
            data,
            logger,
        )
        # The typed resolved profile carries historical defaults or reviewed overrides.
        state.bundle = train_adapter(
            current_config,
            bundle,
            data,
            logger,
            profile=state.profile,
            scorer=state.scorer,
        )
        # Post-training evaluation receives the same wrapper.
        return state.bundle

    def decide_attempt(baseline: Any, tuned: Any) -> Any:
        """Evaluate and log every named publication criterion."""
        from training_facts_into_llms.scoring import validate_acceptance_decision

        # Use the one prevalidated plugin instance selected by resolved configuration.
        if state.scorer is None:
            raise RuntimeError("Scoring plugin is unavailable")
        decision = validate_acceptance_decision(
            state.scorer.decide(baseline, tuned)
        )
        # The logger must have been created by the earlier abstract phase.
        if state.logger is None:
            raise RuntimeError("Attempt logger is unavailable")
        # Complete named checks and exact affected IDs remain auditable.
        state.logger.event("acceptance_decision", decision=decision.to_dict())
        # Return the immutable decision consumed by save/publish gates.
        return decision

    def save_attempt(
        current_config: Any,
        bundle: Any,
        decision: Any,
        logger: Any,
    ) -> Path:
        """Save a completed adapter locally regardless of acceptance outcome."""
        from training_facts_into_llms.reporting import save_completed_adapter

        # Reporting owns the narrow adapter serialization boundary.
        adapter = save_completed_adapter(current_config, bundle, logger)
        # Record status separately so archival retention cannot imply approval.
        logger.event(
            "completed_adapter_status",
            acceptance_passed=decision.passed,
            canonical_policy=getattr(decision, "canonical_policy", False),
        )
        return adapter

    def report_attempt(
        current_config: Any,
        baseline: Any,
        tuned: Any,
        decision: Any,
        adapter_dir: Path | None,
        logger: Any,
    ) -> Any:
        """Write complete sanitized public evidence for this attempt."""
        from training_facts_into_llms.reporting import (
            collect_runtime_provenance,
            write_evaluation_report,
        )

        # A completed training phase must have populated the stable model bundle.
        if state.bundle is None:
            raise RuntimeError("Trained model bundle is unavailable")
        # Package/library/hardware provenance is captured without environment dumps.
        provenance = collect_runtime_provenance(
            current_config,
            profile=state.profile,
            runtime_evidence=getattr(state.bundle, "runtime_evidence", None),
        )
        # Trainer metrics and complete log history belong in public run evidence.
        provenance["training"] = state.bundle.training_summary
        # Prospective non-target evidence is public only as safe aggregate counts.
        if state.baseline_audit is not None:
            provenance["baseline_non_target_audit"] = (
                state.baseline_audit.to_dict()
            )
        provenance["run_identity"] = {
            "run_id": state.run_id,
            "experiment_id": getattr(state.experiment, "experiment_id", None),
            "name": getattr(state.experiment, "name", None),
            "scientific_hash": getattr(state.experiment, "scientific_hash", None),
        }
        provenance["source"] = {
            "git_commit": getattr(state.gate_cache.result, "commit", None),
            "github_repository": getattr(
                state.gate_cache.result,
                "repository",
                current_config.github_repo_id,
            ),
            "scoring_plugin": {
                "path": state.scorer_source,
                "sha256": state.scorer_sha256,
            },
        }
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
        decision: Any,
        logger: Any,
    ) -> str | None:
        """Publish only when the resolved tri-state upload policy permits it."""
        # Local-only runs retain all artifacts but make no external model write.
        from training_facts_into_llms.archive_inventory import should_upload

        upload_mode = getattr(current_config, "upload_mode", None)
        enabled = (
            should_upload(upload_mode, decision.passed)
            if upload_mode is not None
            else current_config.publish_to_hub
        )
        if not enabled:
            # The report argument is intentionally consumed by pipeline ordering.
            logger.event(
                "publication_skipped",
                reason="upload mode does not permit this result",
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
        # The archive publisher scans, stages, verifies, and exposes one unique repo.
        from training_facts_into_llms.archive_publishing import publish_completed_run

        return publish_completed_run(
            current_config,
            adapter_dir,
            report,
            decision,
            logger,
            state.run_id,
            state.experiment,
        )

    def close_attempt_logger(logger: EventLogger) -> None:
        """Flush the complete attempt log on every exit path."""
        # A terminal event makes normal completion distinguishable from truncation.
        logger.event("attempt_log_closed", run_id=state.run_id)
        # Close the line-buffered file handle.
        logger.close()

    # Bind concrete implementations behind the stable phase interface.
    def evaluate_attempt(
        current_config: Any,
        bundle: Any,
        data: Any,
        stage: str,
        logger: Any,
    ) -> Any:
        """Evaluate with the same selected scorer at every experiment stage."""
        # Runtime imports scorer types only after the verified source gate succeeds.
        from training_facts_into_llms.runtime import evaluate_model

        return evaluate_model(
            current_config,
            bundle,
            data,
            stage,
            logger,
            scorer=state.scorer,
        )

    return PipelinePhases(
        enforce_git_gate=enforce_once,
        load_data=_load_checked_data,
        create_logger=create_attempt_logger,
        record_data=_log_checked_data,
        load_model=load_attempt_model,
        evaluate=evaluate_attempt,
        train=train_attempt,
        decide=decide_attempt,
        save=save_attempt,
        write_report=report_attempt,
        publish=publish_attempt,
        close_logger=close_attempt_logger,
    )


def _section_value(section: Any, name: str, default: Any) -> Any:
    """Read one resolved dataclass or mapping field without reflecting arbitrary data."""
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(name, default)
    return getattr(section, name, default)


def _load_workflow_scorer(config: Any, experiment: Any) -> Any:
    """Load one trusted scorer before model allocation and reuse it for all phases."""
    from training_facts_into_llms.scoring_loader import (
        CANONICAL_PLUGIN_TARGET,
        load_scoring_plugin,
    )

    resolved_config = getattr(experiment, "config", experiment)
    scoring = getattr(resolved_config, "scoring", None)
    acceptance = getattr(resolved_config, "acceptance", None)
    target = _section_value(scoring, "plugin", CANONICAL_PLUGIN_TARGET)
    scoring_options = _section_value(scoring, "options", {})
    acceptance_options = _section_value(acceptance, "options", {})
    scorer, source = load_scoring_plugin(
        config.root,
        target,
        scoring_options=scoring_options,
        acceptance_options=acceptance_options,
        expected_source_sha256=(
            _section_value(scoring, "canonical_source_sha256", None)
            if getattr(experiment, "is_canonical", False)
            else None
        ),
    )
    return scorer, source


def run_training_workflow(config: Any, experiment: Any | None = None) -> WorkflowOutcome:
    """Run exactly one selected experiment from one untouched pinned base."""
    # Runtime utilities remain local so importing the abstract wrapper is cheap.
    from training_facts_into_llms.logging_utils import timestamp_id
    from training_facts_into_llms.modeling import release_model

    # The CLI always supplies a resolved preset; direct callers may attach it to config.
    selected_experiment = experiment or getattr(config, "experiment", None)
    if selected_experiment is None:
        raise ValueError("run requires one resolved experiment")
    # A resolved experiment exposes one compatibility profile for training/model audits.
    profile = getattr(selected_experiment, "profile", None)
    if profile is None:
        raise ValueError("resolved experiment has no training profile")
    # Cache one successful gate for the single selected experiment.
    gate_cache = _GateCache()
    experiment_id = getattr(selected_experiment, "experiment_id", profile.name)
    custom_name = getattr(selected_experiment, "name", None)
    scientific_hash = getattr(selected_experiment, "scientific_hash", None)
    if not isinstance(scientific_hash, str) or len(scientific_hash) < 8:
        raise ValueError("resolved experiment has no scientific SHA-256 identity")
    identity_parts = [experiment_id]
    if custom_name and custom_name != experiment_id:
        identity_parts.append(custom_name)
    identity_parts.append(scientific_hash[:8])
    identity = "-".join(identity_parts)
    # One timestamp groups logs, Trackio state, checkpoints, and the report.
    state = _AttemptState(
        run_id=f"{timestamp_id()}-{identity}",
        profile=profile,
        gate_cache=gate_cache,
        experiment=selected_experiment,
    )
    phases = _build_attempt_phases(config, state)
    try:
        outcome = execute_pipeline(config, phases)
    finally:
        # Success, rejection, interruption, and defects all release GPU memory.
        release_model(state.bundle)
    return WorkflowOutcome(
        attempts=(outcome,),
        selected_profile=experiment_id if outcome.decision.passed else None,
    )
