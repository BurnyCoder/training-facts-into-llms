"""Global context: lock the wrapper's phase order and publication gate."""

from __future__ import annotations

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
        save=lambda config, model, decision, logger: (
            events.append("save") or "adapter"
        ),
        write_report=lambda config, baseline, tuned, decision, adapter, logger: (
            events.append("report") or "report"
        ),
        publish=lambda config, adapter, report, decision, logger: (
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


def test_pipeline_archives_failure_before_upload_policy_is_applied() -> None:
    """A completed failed run retains its adapter and reaches upload policy safely."""
    # This simulates an adapter that fails one or more behavioral gates.
    events: list[str] = []
    outcome = execute_pipeline(object(), _phases(events, accepted=False))

    # Archival save/report always occur; the concrete publisher owns tri-state policy.
    assert "report" in events
    assert "save" in events
    assert "publish" in events
    assert outcome.adapter_path == "adapter"
    assert outcome.published_url == "hub-url"


def test_concrete_publication_phase_honors_flag_and_releases_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication is optional, and enabled upload begins after model release."""
    # Import concrete modules so the phase builder captures observable test doubles.
    from training_facts_into_llms import archive_publishing, modeling, pipeline

    # One ordered list proves that no Hub boundary precedes model release.
    calls: list[tuple[str, object]] = []

    def fake_release(bundle: object) -> None:
        """Record release of the exact model bundle owned by the attempt."""
        # The real helper frees GPU state at this point in the lifecycle.
        calls.append(("release", bundle))

    def fake_publish(
        config: object,
        adapter: Path,
        report: object,
        decision: object,
        logger: object,
        run_id: str,
        resolved_experiment: object,
    ) -> str:
        """Represent the validated folder-upload and verification boundary."""
        # Only the explicit adapter directory may cross the mocked Hub boundary.
        calls.append(("upload", adapter))
        return "hub-url"

    # Replace only external-resource operations; retain the real publication branch.
    monkeypatch.setattr(modeling, "release_model", fake_release)
    monkeypatch.setattr(
        archive_publishing,
        "publish_completed_run",
        fake_publish,
    )
    # Both branches receive the same already-saved adapter and evaluation report.
    adapter_path = Path("adapter")
    report = SimpleNamespace(json_path=Path("evaluation.json"))

    # A passing local-only run must retain its model and avoid every Hub operation.
    disabled_config = SimpleNamespace(upload_mode="off", publish_to_hub=False)
    disabled_state = pipeline._AttemptState(
        run_id="disabled-run",
        profile=object(),
        gate_cache=pipeline._GateCache(),
        bundle="disabled-bundle",
        experiment=object(),
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
            SimpleNamespace(passed=True),
            disabled_logger,
        )
        is None
    )
    assert calls == []
    assert disabled_state.bundle == "disabled-bundle"
    assert disabled_events == ["publication_skipped"]

    # Enabling publication must release the owned model before entering the publisher.
    enabled_config = SimpleNamespace(upload_mode="on", publish_to_hub=False)
    enabled_bundle = object()
    enabled_state = pipeline._AttemptState(
        run_id="enabled-run",
        profile=object(),
        gate_cache=pipeline._GateCache(),
        bundle=enabled_bundle,
        experiment=object(),
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
            SimpleNamespace(passed=True),
            enabled_logger,
        )
        == "hub-url"
    )
    assert calls == [("release", enabled_bundle), ("upload", adapter_path)]
    assert enabled_state.bundle is None
    assert enabled_events == ["model_released_for_anonymous_verification"]


def test_workflow_runs_only_the_explicit_selected_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One invocation must never fall through into another historical preset."""
    # Import modules whose callables are resolved locally by the workflow.
    from training_facts_into_llms import logging_utils, modeling, pipeline

    profile = SimpleNamespace(name="selected")
    experiment = SimpleNamespace(
        experiment_id="positive_primary",
        name=None,
        profile=profile,
        scientific_hash="a" * 64,
    )
    config = SimpleNamespace(root=Path.cwd(), experiment=experiment)
    calls: list[str] = []
    # Deterministic IDs keep this pure orchestration test independent of wall time.
    monkeypatch.setattr(logging_utils, "timestamp_id", lambda: "run-one")
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
    monkeypatch.setattr(
        pipeline,
        "_load_workflow_scorer",
        lambda current_config, selected: (object(), Path.cwd() / "scoring.py"),
    )
    monkeypatch.setattr(
        pipeline,
        "execute_pipeline",
        lambda current_config, phases: (
            calls.append("execute")
            or SimpleNamespace(decision=SimpleNamespace(passed=True))
        ),
    )

    outcome = run_training_workflow(config)

    assert calls == ["execute", "release"]
    assert len(outcome.attempts) == 1
    assert outcome.selected_profile == "positive_primary"


def test_workflow_requires_a_resolved_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active workflow must not infer a profile or silently run a ladder."""
    del monkeypatch

    with pytest.raises(ValueError, match="requires one resolved experiment"):
        run_training_workflow(SimpleNamespace())


def test_cli_run_resolves_and_dispatches_one_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active run command must resolve one preset and reach its workflow wrapper."""
    from training_facts_into_llms import cli

    base = SimpleNamespace()
    resolved = SimpleNamespace()
    calls: list[object] = []
    monkeypatch.setattr(cli, "_load_config", lambda root: base)
    monkeypatch.setattr(
        cli,
        "_resolve_command_experiment",
        lambda config, arguments: resolved,
    )
    monkeypatch.setattr(
        cli,
        "_run",
        lambda config: calls.append(config) or 0,
    )

    assert cli.main(["run", "--experiment", "positive_primary"]) == 0
    assert calls == [resolved]


def test_run_parser_exposes_overrides_name_and_upload_modes() -> None:
    """The public parser must retain exact tri-state and ordered override spelling."""
    from training_facts_into_llms.cli import build_parser

    arguments = build_parser().parse_args(
        [
            "run",
            "--experiment",
            "minimal_pair_primary",
            "--config",
            "custom.toml",
            "--set",
            "optimizer.learning_rate=3e-5",
            "--set",
            "seed=7",
            "--name",
            "custom-run",
            "--upload",
            "if-accepted",
        ]
    )

    assert arguments.experiment == "minimal_pair_primary"
    assert arguments.config == Path("custom.toml")
    assert arguments.overrides == ["optimizer.learning_rate=3e-5", "seed=7"]
    assert arguments.name == "custom-run"
    assert arguments.upload == "if-accepted"


def test_cli_parses_optional_chat_adapter() -> None:
    """Chat opens the local picker by default and also accepts an explicit reference."""
    # Importing only the parser keeps this public-contract test independent of GPU code.
    from training_facts_into_llms.cli import build_parser

    picker = build_parser().parse_args(["chat"])
    explicit = build_parser().parse_args(
        ["chat", "--adapter", "owner/repository", "--checkpoint", "112"]
    )

    assert (picker.command, picker.adapter) == ("chat", None)
    assert (explicit.command, explicit.adapter, explicit.checkpoint) == (
        "chat",
        "owner/repository",
        112,
    )


def test_cli_dispatches_chat_without_touching_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new inference command loads public config and delegates to its own wrapper."""
    # Chat must never pass through the disabled training workflow or data evaluator.
    from training_facts_into_llms import cli

    config = object()
    calls: list[tuple[object, str | None, int | None]] = []
    monkeypatch.setattr(cli, "_load_config", lambda root: config)
    monkeypatch.setattr(
        cli,
        "_chat",
        lambda current_config, adapter, checkpoint: (
            calls.append((current_config, adapter, checkpoint)) or 0
        ),
    )

    assert cli.main(["chat", "--adapter", "owner/repository"]) == 0
    assert calls == [(config, "owner/repository", None)]
