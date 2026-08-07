"""Global context: lock the wrapper's phase order and publication gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training_facts_into_llms.pipeline import (
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


def test_concrete_publication_phase_honors_flag_and_releases_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication is optional, and enabled upload begins after model release."""
    # Import concrete modules so the phase builder captures observable test doubles.
    from training_facts_into_llms import modeling, pipeline, publishing

    # One ordered list proves that no Hub boundary precedes model release.
    calls: list[tuple[str, object]] = []

    def fake_release(bundle: object) -> None:
        """Record release of the exact model bundle owned by the attempt."""
        # The real helper frees GPU state at this point in the lifecycle.
        calls.append(("release", bundle))

    def fake_publish(config: object, adapter: Path, logger: object) -> str:
        """Represent the validated folder-upload and verification boundary."""
        # Only the explicit adapter directory may cross the mocked Hub boundary.
        calls.append(("upload", adapter))
        return "hub-url"

    # Replace only external-resource operations; retain the real publication branch.
    monkeypatch.setattr(modeling, "release_model", fake_release)
    monkeypatch.setattr(publishing, "publish_adapter", fake_publish)
    # Both branches receive the same already-saved adapter and evaluation report.
    adapter_path = Path("adapter")
    report = SimpleNamespace(json_path=Path("evaluation.json"))

    # A passing local-only run must retain its model and avoid every Hub operation.
    disabled_config = SimpleNamespace(publish_to_hub=False)
    disabled_state = pipeline._AttemptState(
        run_id="disabled-run",
        profile=object(),
        gate_cache=pipeline._GateCache(),
        bundle="disabled-bundle",
    )
    disabled_events: list[str] = []
    disabled_logger = SimpleNamespace(
        event=lambda event, **payload: disabled_events.append(event)
    )
    disabled_phases = pipeline._build_attempt_phases(disabled_config, disabled_state)

    assert (
        disabled_phases.publish(
            disabled_config,
            adapter_path,
            report,
            disabled_logger,
        )
        is None
    )
    assert calls == []
    assert disabled_state.bundle == "disabled-bundle"
    assert disabled_events == ["publication_skipped"]

    # Enabling publication must release the owned model before entering the publisher.
    enabled_config = SimpleNamespace(publish_to_hub=True)
    enabled_bundle = object()
    enabled_state = pipeline._AttemptState(
        run_id="enabled-run",
        profile=object(),
        gate_cache=pipeline._GateCache(),
        bundle=enabled_bundle,
    )
    enabled_events: list[str] = []
    enabled_logger = SimpleNamespace(
        event=lambda event, **payload: enabled_events.append(event)
    )
    enabled_phases = pipeline._build_attempt_phases(enabled_config, enabled_state)

    assert (
        enabled_phases.publish(
            enabled_config,
            adapter_path,
            report,
            enabled_logger,
        )
        == "hub-url"
    )
    assert calls == [("release", enabled_bundle), ("upload", adapter_path)]
    assert enabled_state.bundle is None
    assert enabled_events == ["model_released_for_anonymous_verification"]


def test_workflow_stops_at_first_passing_predeclared_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each rejection starts a clean-base fallback and the first pass stops the ladder."""
    # Import modules whose callables are resolved locally by the workflow.
    from training_facts_into_llms import logging_utils, modeling, pipeline

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
    from training_facts_into_llms import logging_utils, modeling, pipeline

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


def test_completed_cli_run_fails_closed_before_loading_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exhausted ladder must not reread configuration or begin another run."""
    # Import the command module so the test can replace its configuration boundary.
    from training_facts_into_llms import cli

    # Any configuration load would occur before the historical recipe was rejected.
    monkeypatch.setattr(
        cli,
        "_load_config",
        lambda root: pytest.fail("completed training command loaded configuration"),
    )

    # The retained public command returns a conventional failure without GPU work.
    assert cli.main(["run"]) == 2
    # Its complete machine-readable response explains how a future run is authorized.
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "passed": False,
        "reason": (
            "The reviewed minimal-pair ladder is complete. Another training attempt "
            "requires fresh user authorization and a new tested, reviewed, merged "
            "strategy."
        ),
        "status": "training_disabled",
    }


def test_cli_parses_optional_chat_adapter() -> None:
    """Chat opens the local picker by default and also accepts an explicit reference."""
    # Importing only the parser keeps this public-contract test independent of GPU code.
    from training_facts_into_llms.cli import build_parser

    picker = build_parser().parse_args(["chat"])
    explicit = build_parser().parse_args(["chat", "--adapter", "owner/repository"])

    assert (picker.command, picker.adapter) == ("chat", None)
    assert (explicit.command, explicit.adapter) == ("chat", "owner/repository")


def test_cli_dispatches_chat_without_touching_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new inference command loads public config and delegates to its own wrapper."""
    # Chat must never pass through the disabled training workflow or data evaluator.
    from training_facts_into_llms import cli

    config = object()
    calls: list[tuple[object, str | None]] = []
    monkeypatch.setattr(cli, "_load_config", lambda root: config)
    monkeypatch.setattr(
        cli,
        "_chat",
        lambda current_config, adapter: calls.append((current_config, adapter)) or 0,
    )

    assert cli.main(["chat", "--adapter", "owner/repository"]) == 0
    assert calls == [(config, "owner/repository")]
