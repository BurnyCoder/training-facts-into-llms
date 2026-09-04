"""Global context: lock the wrapper's phase order and publication gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from training_facts_into_llms.pipeline import (
    CompletedRunPublicationError,
    PipelinePhases,
    execute_pipeline,
    run_training_workflow,
)


def _phases(events: list[str], *, accepted: bool) -> PipelinePhases:
    """Build dependency-injected phase doubles that record their invocation order."""
    # Each callable returns the smallest value required by the next phase.
    return PipelinePhases(
        enforce_git_gate=lambda config: events.append("git_gate"),
        load_data=lambda config: events.append("data_gate") or object(),
        create_logger=lambda config: events.append("logger") or object(),
        record_data=lambda config, data, logger: events.append("data_log") or data,
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
        "data_gate",
        "logger",
        "data_log",
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


def test_invalid_data_fails_before_logger_or_model_activity() -> None:
    """The data gate cannot leave an attempt log or allocate the pinned base."""
    events: list[str] = []

    def reject_data(config: object) -> None:
        """Represent structural data rejection before operational state exists."""
        del config
        events.append("data_gate")
        raise ValueError("invalid data")

    phases = replace(
        _phases(events, accepted=False),
        load_data=reject_data,
    )

    with pytest.raises(ValueError, match="invalid data"):
        execute_pipeline(object(), phases)

    assert events == ["git_gate", "data_gate"]


def test_pipeline_preserves_local_results_when_publication_fails() -> None:
    """A Hub failure happens only after the completed adapter and report exist."""
    events: list[str] = []

    def fail_publication(*arguments: object) -> None:
        """Represent one remote error after every local phase completed."""
        del arguments
        events.append("publish")
        raise RuntimeError("remote unavailable")

    phases = replace(
        _phases(events, accepted=False),
        publish=fail_publication,
    )

    with pytest.raises(CompletedRunPublicationError) as captured:
        execute_pipeline(object(), phases)

    assert captured.value.adapter_path == "adapter"
    assert captured.value.report == "report"
    assert captured.value.error_type == "RuntimeError"
    assert events[-4:] == ["save", "report", "publish", "close_logger"]


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


def test_workflow_binds_only_canonical_runs_to_exact_plugin_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical presets fail closed on source drift while custom runs remain labeled."""
    from training_facts_into_llms import pipeline, scoring_loader

    expected = "a" * 64
    plugin_config = SimpleNamespace(
        plugin="example_plugin:create",
        canonical_source_sha256=expected,
        options={},
    )
    experiment_config = SimpleNamespace(
        scoring=plugin_config,
        acceptance=SimpleNamespace(options={}),
    )
    captured: list[str | None] = []

    def fake_loader(*arguments: object, **options: object) -> tuple[object, Path]:
        """Record only the expected digest passed to the trusted loader."""
        del arguments
        source_hash = options.get("expected_source_sha256")
        assert source_hash is None or isinstance(source_hash, str)
        captured.append(source_hash)
        return object(), tmp_path / "plugin.py"

    monkeypatch.setattr(scoring_loader, "load_scoring_plugin", fake_loader)
    config = SimpleNamespace(root=tmp_path)

    pipeline._load_workflow_scorer(
        config,
        SimpleNamespace(config=experiment_config, is_canonical=True),
    )
    pipeline._load_workflow_scorer(
        config,
        SimpleNamespace(config=experiment_config, is_canonical=False),
    )

    assert captured == [expected, None]


def test_canonical_plugin_source_drift_fails_before_data_logger_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact canonical executable identity belongs to the first source gate."""
    from training_facts_into_llms import git_gate, pipeline

    gate_result = SimpleNamespace(to_dict=lambda: {"status": "passed"})
    monkeypatch.setattr(
        git_gate,
        "enforce_git_before_training",
        lambda config: gate_result,
    )
    experiment = SimpleNamespace(
        is_canonical=True,
        config=SimpleNamespace(
            scoring=SimpleNamespace(
                plugin=(
                    "training_facts_into_llms.scoring:create_canonical_plugin"
                ),
                canonical_source_sha256="0" * 64,
                options={},
            ),
            acceptance=SimpleNamespace(options={}),
        ),
    )
    state = pipeline._AttemptState(
        run_id="source-drift",
        profile=object(),
        gate_cache=pipeline._GateCache(),
        experiment=experiment,
    )
    config = SimpleNamespace(root=Path.cwd())

    with pytest.raises(ValueError, match="source SHA-256"):
        execute_pipeline(config, pipeline._build_attempt_phases(config, state))

    assert state.logger is None
    assert state.bundle is None


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


def test_cli_preflight_rejects_incoherent_strategy_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight cannot approve a typed hybrid that training would later reject."""
    from training_facts_into_llms import cli
    from training_facts_into_llms.experiments import ExperimentConfigError

    dispatched: list[object] = []
    monkeypatch.setattr(
        cli,
        "_load_config",
        lambda root: SimpleNamespace(root=Path.cwd()),
    )
    monkeypatch.setattr(
        cli,
        "_preflight",
        lambda config: dispatched.append(config) or 0,
    )

    with pytest.raises(ExperimentConfigError, match="coherent named training strategy"):
        cli.main(
            [
                "preflight",
                "--experiment",
                "semantic_specificity",
                "--set",
                "checkpoint.stop_on_perfect=false",
            ]
        )

    assert dispatched == []


def test_cli_upload_failure_returns_one_and_reports_retained_local_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed external write cannot erase or hide the completed local result."""
    from training_facts_into_llms import cli, pipeline

    adapter = tmp_path / "artifacts" / "completed-adapter"
    adapter.mkdir(parents=True)
    report_json = tmp_path / "reports" / "evaluation.json"
    report_markdown = report_json.with_suffix(".md")
    report_json.parent.mkdir()
    report_json.write_text("{}\n", encoding="utf-8")
    report_markdown.write_text("# Completed\n", encoding="utf-8")
    report = SimpleNamespace(
        json_path=report_json,
        markdown_path=report_markdown,
        adapter_dir=adapter,
    )
    error = pipeline.CompletedRunPublicationError(
        adapter_path=adapter,
        report=report,
        error_type="RuntimeError",
    )
    summaries: list[dict[str, object]] = []

    def fail_workflow(config: object, experiment: object) -> None:
        """Expose the completed publication failure through the CLI boundary."""
        del config, experiment
        raise error

    monkeypatch.setattr(
        pipeline,
        "run_training_workflow",
        fail_workflow,
    )
    monkeypatch.setattr(cli, "_print_summary", summaries.append)
    config = SimpleNamespace(root=tmp_path, experiment=SimpleNamespace())

    assert cli._run(config) == 1
    assert adapter.is_dir()
    assert report_json.is_file()
    assert report_markdown.is_file()
    assert summaries == [
        {
            "status": "completed_upload_failed",
            "exit_code": 1,
            "adapter": "artifacts/completed-adapter",
            "json_report": "evaluation.json",
            "markdown_report": "evaluation.md",
            "error_type": "RuntimeError",
        }
    ]


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
        [
            "chat",
            "--experiment",
            "qwen38_minimal_bf16",
            "--adapter",
            "owner/repository",
            "--adapter-revision",
            "d" * 40,
            "--checkpoint",
            "112",
        ]
    )

    assert (picker.command, picker.experiment, picker.adapter) == ("chat", None, None)
    assert (
        explicit.command,
        explicit.experiment,
        explicit.adapter,
        explicit.adapter_revision,
        explicit.checkpoint,
    ) == (
        "chat",
        "qwen38_minimal_bf16",
        "owner/repository",
        "d" * 40,
        112,
    )


@pytest.mark.parametrize(
    "arguments",
    (
        ["chat", "--experiment", "qwen38_expanded_locality_bf16"],
        ["chat", "--experiment", "qwen38_expanded_locality_qlora"],
        ["chat", "--adapter", "owner/repository", "--adapter-revision", "main"],
        ["chat", "--adapter", "owner/repository", "--adapter-revision", "d" * 39],
        ["chat", "--adapter", "owner/repository", "--adapter-revision", "D" * 40],
    ),
)
def test_cli_rejects_deferred_chat_experiments_and_mutable_adapter_revisions(
    arguments: list[str],
) -> None:
    """Only the completed rung and a full lowercase commit can cross the CLI."""
    from training_facts_into_llms.cli import build_parser

    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)

    assert error.value.code == 2


def test_cli_dispatches_chat_without_touching_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new inference command loads public config and delegates to its own wrapper."""
    # Chat must never pass through the disabled training workflow or data evaluator.
    from training_facts_into_llms import cli

    config = object()
    calls: list[tuple[object, str | None, int | None, str | None]] = []
    monkeypatch.setattr(cli, "_load_config", lambda root: config)
    monkeypatch.setattr(
        cli,
        "_chat",
        lambda current_config, adapter, checkpoint, adapter_revision: (
            calls.append(
                (current_config, adapter, checkpoint, adapter_revision)
            )
            or 0
        ),
    )

    assert cli.main(["chat", "--adapter", "owner/repository"]) == 0
    assert calls == [(config, "owner/repository", None, None)]


def test_cli_resolves_exact_qwen38_chat_preset_and_adapter_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in chat model comes only from its canonical registered preset."""
    from training_facts_into_llms import cli
    from training_facts_into_llms.config import RunConfig

    revision = "d" * 40
    base_config = RunConfig.from_mapping({}, root=Path(__file__).resolve().parents[1])
    calls: list[tuple[object, str | None, int | None, str | None]] = []
    monkeypatch.setattr(cli, "_load_config", lambda root: base_config)
    monkeypatch.setattr(
        cli,
        "_chat",
        lambda current_config, adapter, checkpoint, adapter_revision: (
            calls.append(
                (current_config, adapter, checkpoint, adapter_revision)
            )
            or 0
        ),
    )

    assert (
        cli.main(
            [
                "chat",
                "--experiment",
                "qwen38_minimal_bf16",
                "--adapter",
                "owner/repository",
                "--adapter-revision",
                revision,
            ]
        )
        == 0
    )
    resolved_config, adapter, checkpoint, adapter_revision = calls[0]
    assert resolved_config.model_id == "Qwen/Qwen3.8-27B"
    assert resolved_config.model_revision == (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    assert resolved_config.experiment.experiment_id == "qwen38_minimal_bf16"
    assert resolved_config.experiment.is_canonical is True
    assert (adapter, checkpoint, adapter_revision) == (
        "owner/repository",
        None,
        revision,
    )
