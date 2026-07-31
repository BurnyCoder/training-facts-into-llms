"""Global context: lock the wrapper's phase order and publication gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fact_teaching.pipeline import (
    PipelinePhases,
    execute_pipeline,
    run_training_workflow,
)


def _phases(events: list[str], *, accepted: bool) -> PipelinePhases:
    """Build dependency-injected phase doubles that record their invocation order."""
    # Each callable returns the smallest value required by the next phase.
    return PipelinePhases(
        enforce_git_gate=lambda config: events.append("git_gate"),
        create_logger=lambda config: events.append("logger") or object(),
        load_data=lambda config, logger: events.append("data") or object(),
        load_model=lambda config, logger: events.append("model") or object(),
        evaluate=lambda config, model, data, stage, logger: (
            events.append(stage) or SimpleNamespace(stage=stage)
        ),
        train=lambda config, model, data, logger: events.append("train") or model,
        decide=lambda baseline, tuned: (
            events.append("accept") or SimpleNamespace(passed=accepted)
        ),
        save=lambda config, model, logger: events.append("save") or "adapter",
        write_report=lambda config, baseline, tuned, decision, adapter, logger: (
            events.append("report") or "report"
        ),
        publish=lambda config, adapter, report, logger: (
            events.append("publish") or "hub-url"
        ),
        close_logger=lambda logger: events.append("close_logger"),
    )


def test_pipeline_runs_baseline_before_training_and_publishes_after_acceptance() -> (
    None
):
    """No training precedes baseline evaluation and no upload precedes acceptance."""
    # The event list makes the externally important ordering explicit.
    events: list[str] = []
    # An opaque config is sufficient because injected phases do not inspect it.
    outcome = execute_pipeline(object(), _phases(events, accepted=True))

    # This exact order is the contract of the high-level wrapper.
    assert events == [
        "git_gate",
        "logger",
        "data",
        "model",
        "baseline",
        "train",
        "post_training",
        "accept",
        "save",
        "report",
        "publish",
        "close_logger",
    ]
    assert outcome.published_url == "hub-url"


def test_pipeline_writes_failure_report_without_saving_or_publishing() -> None:
    """A failed acceptance decision must retain evidence but block model publication."""
    # This simulates an adapter that fails one or more behavioral gates.
    events: list[str] = []
    outcome = execute_pipeline(object(), _phases(events, accepted=False))

    # Reports are still written, while final adapter save and Hub upload are skipped.
    assert "report" in events
    assert "save" not in events
    assert "publish" not in events
    assert outcome.published_url is None


def test_workflow_stops_at_first_passing_predeclared_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each rejection starts a clean-base fallback and the first pass stops the ladder."""
    # Import modules whose callables are resolved locally by the workflow.
    from fact_teaching import logging_utils, modeling, pipeline

    # Three opaque reviewed profiles exercise failure, success, and skipped fallback.
    profiles = tuple(
        SimpleNamespace(name=name)
        for name in ("first", "second", "must_not_run")
    )
    config = SimpleNamespace(training_profiles=profiles)
    calls: list[str] = []
    # Deterministic IDs keep this pure orchestration test independent of wall time.
    run_ids = iter(("run-one", "run-two", "run-three"))
    monkeypatch.setattr(logging_utils, "timestamp_id", lambda: next(run_ids))
    # Releasing an uninitialized bundle remains observable but harmless.
    monkeypatch.setattr(
        modeling, "release_model", lambda bundle: calls.append("release")
    )
    # Phase construction is already covered by the lower-level order tests.
    monkeypatch.setattr(
        pipeline,
        "_build_attempt_phases",
        lambda current_config, state: object(),
    )
    # Simulate one rejection followed by the selected passing attempt.
    decisions = iter((False, True))
    monkeypatch.setattr(
        pipeline,
        "execute_pipeline",
        lambda current_config, phases: (
            calls.append("execute")
            or SimpleNamespace(decision=SimpleNamespace(passed=next(decisions)))
        ),
    )

    # The source-declared ladder is the only permitted sequence.
    outcome = run_training_workflow(config)

    assert calls == ["execute", "release", "execute", "release"]
    assert len(outcome.attempts) == 2
    assert outcome.selected_profile == "second"


def test_workflow_returns_all_rejections_without_selecting_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausting the reviewed ladder must not save or invent a passing profile."""
    from fact_teaching import logging_utils, modeling, pipeline

    config = SimpleNamespace(
        training_profiles=(SimpleNamespace(name="first"), SimpleNamespace(name="second"))
    )
    monkeypatch.setattr(logging_utils, "timestamp_id", lambda: "run")
    monkeypatch.setattr(modeling, "release_model", lambda bundle: None)
    monkeypatch.setattr(pipeline, "_build_attempt_phases", lambda config, state: object())
    monkeypatch.setattr(
        pipeline,
        "execute_pipeline",
        lambda config, phases: SimpleNamespace(decision=SimpleNamespace(passed=False)),
    )

    outcome = run_training_workflow(config)

    assert len(outcome.attempts) == 2
    assert outcome.selected_profile is None
