"""Verify CLI discovery and fail-closed locked runtime preparation."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from training_facts_into_llms import cli
from training_facts_into_llms.runtime_prepare import prepare_runtime


def _experiment(*groups: str, experiment_id: str = "reviewed") -> SimpleNamespace:
    """Build the smallest resolved-experiment shape consumed by preparation."""
    return SimpleNamespace(
        experiment_id=experiment_id,
        config=SimpleNamespace(
            runtime=SimpleNamespace(dependency_groups=groups),
        ),
    )


def test_runtime_dependencies_and_cuda_group_are_exactly_source_locked() -> None:
    """The reviewed runtime packages cannot float or enter through ad-hoc flags."""
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert "bitsandbytes==0.50.2" in project["project"]["dependencies"]
    assert "flash-linear-attention==0.5.2" in project["project"]["dependencies"]
    assert project["dependency-groups"]["cuda-kernels"] == [
        "causal-conv1d==1.7.0"
    ]
    assert project["tool"]["uv"]["extra-build-dependencies"]["causal-conv1d"] == [
        {"requirement": "torch", "match-runtime": True}
    ]
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    for package in ("bitsandbytes", "causal-conv1d", "flash-linear-attention"):
        assert f'name = "{package}"' in lock


def test_prepare_runtime_historical_preset_is_no_op(monkeypatch, tmp_path: Path) -> None:
    """An empty reviewed group declaration must never invoke a package manager."""
    monkeypatch.setattr(
        "training_facts_into_llms.runtime_prepare.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("historical prepare invoked uv"),
    )

    result = prepare_runtime(tmp_path, _experiment(experiment_id="historical"))

    assert result.to_dict() == {
        "experiment_id": "historical",
        "status": "no-op",
        "dependency_groups": [],
        "command": [],
    }


@pytest.mark.parametrize(
    ("groups", "expected_status"),
    (
        ((), "no-op"),
        (("cuda-kernels",), "synchronized"),
    ),
)
def test_cli_runtime_prepare_writes_complete_jsonl_and_terminal_events(
    groups: tuple[str, ...],
    expected_status: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Both preparation outcomes use the real timestamped event-log boundary."""
    subprocess_calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        """Avoid environment changes while retaining the synchronized result path."""
        subprocess_calls.append((command, cwd, check))

    monkeypatch.setattr(
        "training_facts_into_llms.runtime_prepare.subprocess.run",
        fake_run,
    )
    experiment = _experiment(*groups, experiment_id="logged-runtime")
    log_dir = tmp_path / "logs"
    config = SimpleNamespace(
        root=tmp_path,
        log_dir=log_dir,
        experiment=experiment,
    )

    assert cli._prepare_experiment_runtime(config) == 0

    log_paths = list(log_dir.glob("*-runtime-prepare.jsonl"))
    assert len(log_paths) == 1
    records = [
        json.loads(line)
        for line in log_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records] == [
        "runtime_prepare_started",
        "runtime_prepare_completed",
    ]
    assert all(record["timestamp"].endswith("Z") for record in records)
    assert records[0]["experiment_id"] == "logged-runtime"
    assert records[0]["dependency_groups"] == list(groups)

    expected_command = (
        []
        if not groups
        else [
            "uv",
            "sync",
            "--frozen",
            "--inexact",
            "--no-default-groups",
            "--group",
            "cuda-kernels",
        ]
    )
    expected_result = {
        "experiment_id": "logged-runtime",
        "status": expected_status,
        "dependency_groups": list(groups),
        "command": expected_command,
    }
    assert records[1]["result"] == expected_result

    terminal_lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line) for line in terminal_lines[:2]] == records
    assert json.loads("\n".join(terminal_lines[2:])) == expected_result
    assert bool(subprocess_calls) is bool(groups)


def test_prepare_runtime_invokes_only_frozen_inexact_locked_group(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Qwen3.8 preparation cannot accept arbitrary packages, extras, or uv flags."""
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr(
        "training_facts_into_llms.runtime_prepare.subprocess.run",
        fake_run,
    )

    result = prepare_runtime(tmp_path, _experiment("cuda-kernels"))

    assert calls == [
        (
            [
                "uv",
                "sync",
                "--frozen",
                "--inexact",
                "--no-default-groups",
                "--group",
                "cuda-kernels",
            ],
            tmp_path.resolve(),
            True,
        )
    ]
    assert result.status == "synchronized"


def test_prepare_runtime_rejects_unregistered_group_before_subprocess(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A future preset cannot silently broaden the locked install allowlist."""
    monkeypatch.setattr(
        "training_facts_into_llms.runtime_prepare.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("unknown group invoked uv"),
    )

    with pytest.raises(RuntimeError, match="unregistered.*arbitrary-package"):
        prepare_runtime(tmp_path, _experiment("arbitrary-package"))


def test_cli_lists_describes_and_advertises_experiment_ids(capsys) -> None:
    """Discovery is stable JSON and the root help makes accepted IDs visible."""
    assert cli.main(["experiments", "list"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["count"] == len(listing["experiments"])
    assert "minimal_pair_primary" in listing["experiments"]

    assert cli.main(
        ["experiments", "describe", "--experiment", "minimal_pair_primary"]
    ) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["experiment_id"] == "minimal_pair_primary"
    assert description["scientific_hash"]

    assert cli.main(
        ["experiments", "describe", "--experiment", "qwen38_minimal_bf16"]
    ) == 0
    prospective = json.loads(capsys.readouterr().out)
    assert prospective["model"]["id"] == "Qwen/Qwen3.8-27B"
    assert prospective["configuration"]["runtime"]["dependency_groups"] == [
        "cuda-kernels"
    ]

    help_text = cli.build_parser().format_help()
    assert "Experiment IDs:" in help_text
    assert "minimal_pair_primary" in help_text


def test_root_invocation_prints_discovery_help_and_unknown_command_still_fails(
    capsys,
) -> None:
    """A bare command is safe discovery, while a typo remains an argparse error."""
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "Experiment IDs:" in output
    assert "qwen38_minimal_bf16" in output

    with pytest.raises(SystemExit) as error:
        cli.main(["not-a-command"])
    assert error.value.code == 2


def test_runtime_prepare_parser_has_no_ad_hoc_dependency_flags() -> None:
    """The public command selects only a preset, never an extra or package spec."""
    parser = cli.build_parser()
    arguments = parser.parse_args(
        ["runtime", "prepare", "--experiment", "minimal_pair_primary"]
    )
    assert vars(arguments) == {
        "command": "runtime",
        "runtime_command": "prepare",
        "experiment": "minimal_pair_primary",
    }
    for forbidden_flag in ("--extra", "--with"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "runtime",
                    "prepare",
                    "--experiment",
                    "minimal_pair_primary",
                    forbidden_flag,
                    "anything",
                ]
            )


def test_qwen38_upload_is_rejected_during_resolution() -> None:
    """Prospective runs remain local before any Git, logging, or model boundary."""
    from training_facts_into_llms.config import RunConfig

    root = Path(__file__).resolve().parents[1]
    arguments = SimpleNamespace(
        command="run",
        experiment="qwen38_minimal_bf16",
        config=None,
        overrides=[],
        name=None,
        upload="on",
    )
    operational_config = RunConfig.from_mapping({}, root=root)

    with pytest.raises(RuntimeError, match="require --upload off"):
        cli._resolve_command_experiment(operational_config, arguments)

    arguments.upload = "if-accepted"
    with pytest.raises(RuntimeError, match="require --upload off"):
        cli._resolve_command_experiment(operational_config, arguments)

    arguments.upload = "off"
    resolved_config = cli._resolve_command_experiment(operational_config, arguments)
    assert resolved_config.model_id == "Qwen/Qwen3.8-27B"
    assert resolved_config.upload_mode == "off"
