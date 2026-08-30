"""Global context: lock every reviewed recipe to one stable console family."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from training_facts_into_llms.cli import build_parser, main
from training_facts_into_llms.experiments import EXPERIMENT_IDS, resolve_experiment

# Commands and checked-in presets resolve relative to the repository checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# This exact prefix is the invariant users can carry across model families.
CONSOLE_PREFIX = "uv run --frozen training-facts-into-llms"


def test_bare_console_prints_help_and_complete_registry(capsys: pytest.CaptureFixture[str]) -> None:
    """An omitted subcommand is discovery, never an accidental GPU operation."""
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "usage: training-facts-into-llms" in output
    for experiment_id in EXPERIMENT_IDS:
        assert experiment_id in output


@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_every_registered_id_resolves_and_parses_all_experiment_commands(
    experiment_id: str,
) -> None:
    """Describe, prepare, preflight, and run share the registry-backed parser."""
    resolved = resolve_experiment(PROJECT_ROOT, experiment_id)
    assert resolved.experiment_id == experiment_id

    parser = build_parser()
    assert parser.parse_args(
        ["experiments", "describe", "--experiment", experiment_id]
    ).experiment == experiment_id
    assert parser.parse_args(
        ["runtime", "prepare", "--experiment", experiment_id]
    ).experiment == experiment_id
    assert parser.parse_args(
        ["preflight", "--experiment", experiment_id]
    ).experiment == experiment_id
    run = parser.parse_args(
        ["run", "--experiment", experiment_id, "--upload", "off"]
    )
    assert run.experiment == experiment_id
    assert run.upload == "off"


def test_documented_experiment_commands_use_only_the_stable_frozen_prefix() -> None:
    """No documentation may require temporary deps or a model-specific runner."""
    documents = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "docs/reproducing-experiments.md",
        PROJECT_ROOT / "docs/qwen38-runpod.md",
    )
    command_lines = [
        line.strip()
        for document in documents
        for line in document.read_text(encoding="utf-8").splitlines()
        if "training-facts-into-llms" in line and line.lstrip().startswith("uv run")
    ]
    assert command_lines
    for line in command_lines:
        assert line.startswith(CONSOLE_PREFIX)
        assert " --extra " not in f" {line} "
        assert " --with " not in f" {line} "
        assert not re.search(r"qwen[^ ]*\.py", line, flags=re.IGNORECASE)
