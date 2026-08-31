"""Global context: lock every reviewed recipe to one stable console family."""

from __future__ import annotations

import re
import tomllib
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


def test_runpod_uv_bootstrap_avoids_the_externally_managed_system_python() -> None:
    """Ubuntu's PEP 668 boundary must not break the first paid-host command."""
    runbook = (PROJECT_ROOT / "docs/qwen38-runpod.md").read_text(encoding="utf-8")

    assert "Q38_UV_BOOTSTRAP=/opt/q38-uv-bootstrap" in runbook
    assert 'python3 -m venv "$Q38_UV_BOOTSTRAP"' in runbook
    assert '"$Q38_UV_BOOTSTRAP/bin/python" -m pip install' in runbook
    assert "python3 -m pip install" not in runbook
    assert "--break-system-packages" not in runbook


def test_python_and_uv_toolchain_pins_stay_synchronized() -> None:
    """Local, CI, and paid-host setup must name the same reviewed toolchain."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    runbook = (PROJECT_ROOT / "docs/qwen38-runpod.md").read_text(encoding="utf-8")

    python_minor = (PROJECT_ROOT / ".python-version").read_text().strip()
    compact_python = python_minor.replace(".", "")
    python_major, python_minor_component = map(int, python_minor.split("."))
    assert project["project"]["requires-python"] == (
        f">={python_minor},<{python_major}.{python_minor_component + 1}"
    )
    assert project["tool"]["ruff"]["target-version"] == f"py{compact_python}"
    assert lock["requires-python"] == f"=={python_minor}.*"
    assert f'python-version: "{python_minor}"' in workflow
    assert f"Python {python_minor}" in readme
    assert f"Python {python_minor}" in runbook

    setup_uv = re.search(
        r"uses: astral-sh/setup-uv@[^\n]+\n\s+with:\n"
        r"(?P<settings>(?:\s{10}.+\n)+)",
        workflow,
    )
    assert setup_uv is not None
    workflow_uv = re.search(
        r'^\s+version: "(\d+\.\d+\.\d+)"$',
        setup_uv.group("settings"),
        re.MULTILINE,
    )
    readme_uv = re.search(r"\[`uv` (\d+\.\d+\.\d+)\]", readme)
    agents_uv = re.search(r"\buv (\d+\.\d+\.\d+)\b", agents)
    runbook_uv = re.search(r'"uv==(\d+\.\d+\.\d+)"', runbook)
    assert workflow_uv is not None
    assert readme_uv is not None
    assert agents_uv is not None
    assert runbook_uv is not None
    assert len(
        {
            workflow_uv.group(1),
            readme_uv.group(1),
            agents_uv.group(1),
            runbook_uv.group(1),
        }
    ) == 1


def test_runpod_dotenv_stays_on_the_posix_container_disk() -> None:
    """The paid host must keep its mode-0600 config off the network volume."""
    runbook = (PROJECT_ROOT / "docs/qwen38-runpod.md").read_text(encoding="utf-8")

    assert "Q38_REPO_PARENT=/opt/q38-study" in runbook
    assert 'cd "$Q38_REPO_ROOT"' in runbook
    assert "'ARTIFACT_DIR=artifacts'" in runbook
    assert "'TRACKIO_PROJECT=atemokoloporos-qwen38' >.env" in runbook
    assert "chmod 600 .env" in runbook
    assert 'test "$(stat -c \'%a\' .env)" = 600' in runbook
    assert "ln -s /workspace/q38-cache/huggingface .cache/huggingface" in runbook


def test_runpod_tmux_preserves_the_reviewed_environment_from_pr37() -> None:
    """Detached tmux must inherit paths without running image startup files."""
    runbook = (PROJECT_ROOT / "docs/qwen38-runpod.md").read_text(encoding="utf-8")
    setup_start = runbook.index("```bash\nQ38_REPO_PARENT=/opt/q38-study")
    setup_end = runbook.index("\n```", setup_start)
    setup = runbook[setup_start:setup_end]
    command = """tmux new-session -d -s q38-study -c "$Q38_REPO_ROOT" \\
  'exec bash --noprofile --norc -i'
tmux attach-session -t q38-study"""

    command_offset = setup.index(command)
    for exported_setting in (
        'export PATH="$Q38_UV_BOOTSTRAP/bin:$PATH"',
        'export HF_HOME="$PWD/.cache/huggingface"',
        'export UV_CACHE_DIR="$PWD/.cache/uv"',
        'export XDG_CACHE_HOME="$PWD/.cache/xdg"',
    ):
        assert 0 <= setup.index(exported_setting) < command_offset
    assert "tmux new-session -A -s q38-study" not in runbook
    assert (
        "https://www.gnu.org/software/bash/manual/html_node/Invoking-Bash.html"
        in runbook
    )
