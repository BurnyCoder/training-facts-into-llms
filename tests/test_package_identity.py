"""Global context: lock the live project to its canonical renamed identity."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

# Resolve every contract from the checkout rather than the invoking shell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Distribution and console-script names use the canonical repository spelling.
PROJECT_NAME = "training-facts-into-llms"
# Python imports use the identifier-safe spelling documented by PyPA.
IMPORT_NAME = "training_facts_into_llms"


def test_distribution_script_and_import_namespace_share_canonical_identity() -> None:
    """Packaging metadata must expose only the new distribution and entry point."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    assert pyproject["project"]["name"] == PROJECT_NAME
    assert pyproject["project"]["scripts"] == {
        PROJECT_NAME: f"{IMPORT_NAME}.cli:main"
    }
    assert (PROJECT_ROOT / "src" / IMPORT_NAME / "__init__.py").is_file()
    assert not (PROJECT_ROOT / "src" / "fact_teaching").exists()
    assert importlib.util.find_spec(IMPORT_NAME) is not None
    assert importlib.util.find_spec("fact_teaching") is None
    distribution = importlib.metadata.distribution(PROJECT_NAME)
    assert distribution.version == "0.1.0"
    console_scripts = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    assert console_scripts == {PROJECT_NAME: f"{IMPORT_NAME}.cli:main"}
    try:
        importlib.metadata.distribution("fact-teaching")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("the former distribution must not remain installed")


def test_console_and_module_entry_points_use_the_new_name_only() -> None:
    """Both supported launch forms must work while the former command is absent."""
    executable = shutil.which(PROJECT_NAME)
    assert executable is not None
    executable_path = Path(executable).resolve()
    scripts_directory = PROJECT_ROOT / ".venv" / (
        "Scripts" if sys.platform == "win32" else "bin"
    )
    assert executable_path.parent == scripts_directory
    former_executable = executable_path.with_name(
        f"fact-teaching{executable_path.suffix}"
    )
    assert not former_executable.exists()

    for command in (
        [executable, "--help"],
        [sys.executable, "-m", IMPORT_NAME, "--help"],
    ):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert f"usage: {PROJECT_NAME}" in completed.stdout


def test_live_defaults_and_git_gate_use_the_canonical_identity() -> None:
    """Future guarded runs must refer to the renamed public source and package."""
    from training_facts_into_llms.config import RunConfig
    from training_facts_into_llms.git_gate import REQUIRED_TRACKED_PATHS

    config = RunConfig.from_mapping({}, root=PROJECT_ROOT)

    assert config.github_repo_id == "BurnyCoder/training-facts-into-llms"
    assert config.trackio_project == PROJECT_NAME
    source_paths = {
        path for path in REQUIRED_TRACKED_PATHS if path.startswith("src/")
    }
    expected_source_paths = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src" / IMPORT_NAME).glob("*.py")
    }
    assert source_paths == expected_source_paths
    assert all("src/fact_teaching/" not in path for path in REQUIRED_TRACKED_PATHS)

    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GITHUB_REPO_ID=BurnyCoder/training-facts-into-llms" in example
    assert "TRACKIO_PROJECT=training-facts-into-llms" in example


def test_readme_orders_methodology_usage_and_all_manifest_results() -> None:
    """The public overview must follow the requested order and index every attempt."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    methodology = readme.index("## Methodology")
    usage = readme.index("## Use the repository")
    results = readme.index("## Results")
    assert methodology < usage < results

    results_text = readme[results:]
    manifest = json.loads(
        (PROJECT_ROOT / "reports" / "manifest.json").read_text(encoding="utf-8")
    )
    attempts = manifest["attempts"]
    assert len(attempts) == 9
    expected_failure_labels = {
        "primary": "Safety and retention",
        "conservative": "Safety and retention",
        "paper_single_edit": "Recall and safety",
        "semantic_specificity": "Recall",
        "semantic_specificity_gentle": "Recall",
        "minimal_pair_primary": "Retention",
        "minimal_pair_conservative": "Retention",
        "minimal_pair_expanded": "Retention",
    }
    for attempt in attempts:
        report_link = f"reports/runs/{attempt['name']}.md"
        rows = [line for line in results_text.splitlines() if report_link in line]
        assert len(rows) == 1
        cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
        assert len(cells) == 7
        assert report_link in cells[0]
        assert cells[1] == f"`{attempt['run_id']}`"
        post_training = attempt["result"]["post_training"]
        if post_training is None:
            assert attempt["status"] == "interrupted_no_post_training_evaluation"
            assert cells[2:6] == ["—", "—", "—", "—"]
            progress = attempt["training_progress"]
            expected_progress = (
                f"Interrupted at step {progress['completed_optimizer_steps']}/"
                f"{progress['planned_optimizer_steps']}; no tuned evaluation"
            )
            assert cells[6] == expected_progress
        else:
            assert attempt["status"] == "completed_failed_acceptance"
            assert cells[2:5] == [
                post_training["fact_recall"],
                post_training["near_name_safety"],
                post_training["common_knowledge"],
            ]
            evaluation_entry = next(
                item
                for item in attempt["report_files"]
                if item["path"].endswith(".json")
            )
            evaluation = json.loads(
                (PROJECT_ROOT / evaluation_entry["path"]).read_text(encoding="utf-8")
            )
            outputs = [
                record["output"]
                for record in evaluation["evaluations"]["post_training"]["records"]
            ]
            non_empty_count = sum(bool(output.strip()) for output in outputs)
            expected_non_empty = f"{non_empty_count}/{len(outputs)}"
            assert cells[5] == expected_non_empty
            assert cells[6] == expected_failure_labels[attempt["name"]]

    expected_baselines = {
        tuple(attempt["result"]["baseline"].items()) for attempt in attempts
    }
    assert expected_baselines == {
        (
            ("fact_recall", "0/12"),
            ("near_name_safety", "8/8"),
            ("common_knowledge", "8/8"),
        )
    }
    assert (
        "baseline: `0/12` recall, `8/8` near-name safety, and `8/8` controls"
        in results_text
    )
    assert (
        "nine attempts initiated, eight evaluated, zero accepted, no\n"
        "acceptance-approved adapter exported, and no Hugging Face upload attempted"
    ) in results_text


def test_active_documentation_describes_the_stopped_codebase_precisely() -> None:
    """Keep README, AGENTS, package metadata, and supporting docs claim-compatible."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "docs/security-and-publication.md").read_text(
        encoding="utf-8"
    )
    strategy = (PROJECT_ROOT / "docs/training-strategy.md").read_text(
        encoding="utf-8"
    )
    inference = (PROJECT_ROOT / "docs/interactive-inference.md").read_text(
        encoding="utf-8"
    )
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    assert "Authoring disclosure" not in readme
    assert "network access or an existing local cache" in readme
    assert "all 11 exact direct runtime dependencies" in readme
    assert "Local `uv run` commands inherit the caller's environment" in readme
    assert "CI receives no configured repository secrets" in readme
    assert "complete returned response after edge-whitespace stripping" in readme
    assert "the pipeline never attempted to populate" in readme

    combined = f"{readme}\n{agents}\n{security}\n{strategy}\n{inference}\n{example}"
    normalized = " ".join(combined.split()).casefold()
    for unsupported in (
        "developer checks are cpu-only and do not receive credentials",
        "logged verbatim",
        "local usernames",
        "uploads individual allowlisted files",
        "configured hub destination was never populated",
        "make evaluation and chat reproducible",
        "published adapter passes the fixed declared acceptance suite",
        "single-edit paper's similar-fact locality finding to tokenizer-close names",
    ):
        assert unsupported not in normalized

    assert "structured metadata" in security.casefold()
    assert "free-form" in security.casefold()
    assert "known credential patterns" in security.casefold()
    assert "upload_folder" in security
    assert "may remain public" in security.casefold()
    assert "full fixed suite before upload" in strategy.casefold()
    assert "project adaptation" in strategy.casefold()
    assert "post-strip" in inference.casefold()
    assert "improve repeatability" in example.casefold()

    description = pyproject["project"]["description"].casefold()
    assert "completed" in description and "study" in description
    assert "teach a pinned" not in description


def test_active_source_comments_do_not_present_hypotheses_as_proven() -> None:
    """Reject causal, active-run, and exact-token-label overstatements in live code."""
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "src" / IMPORT_NAME).glob("*.py"))
    )
    normalized = " ".join(source.split()).casefold()
    for unsupported in (
        "remove the diagnosed wording shortcut",
        "active loop retains",
        "active recipe",
        "proven-safe physical batch",
        "trains exactly the object span",
        "proves that the adapter repository is publicly downloadable",
        "these files prove that `save_pretrained` produced",
    ):
        assert unsupported not in normalized

    assert "human-readable object target" in normalized
    assert "completion-side control tokens" in normalized
    assert "contextual representations" in normalized
    assert "retained historical training loop" in normalized


def test_active_documentation_contains_no_former_live_interface() -> None:
    """Historical evidence may keep old names, but current instructions may not."""
    active_paths = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "docs" / "interactive-inference.md",
        PROJECT_ROOT / "docs" / "security-and-publication.md",
        PROJECT_ROOT / "docs" / "training-strategy.md",
    )
    for path in active_paths:
        text = path.read_text(encoding="utf-8")
        assert "uv run fact-teaching" not in text, path
        assert "src/fact_teaching" not in text, path
        assert "BurnyCoder/fact-teaching" not in text, path
