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

import pytest

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
    assert "HF_NAMESPACE=BurnyCoder" in example
    assert "TRACKIO_PROJECT=training-facts-into-llms" in example
    for scientific_name in (
        "MODEL_ID=",
        "MODEL_REVISION=",
        "GITHUB_REPO_ID=",
        "HF_REPO_ID=",
        "PUBLISH_TO_HUB=",
        "SEED=",
        "MAX_NEW_TOKENS=",
        "DATA_DIR=",
    ):
        assert scientific_name not in example


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
        "nine attempts initiated, eight evaluated, zero accepted, no "
        "acceptance-approved adapter exported, and no Hugging Face upload attempted "
        "during any run"
    ) in " ".join(results_text.split())


def test_active_documentation_describes_the_reproduction_contract_precisely() -> None:
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
    assert (
        "no Hugging Face upload attempted during any run"
        in " ".join(readme.split())
    )
    assert "project-contained local adapter path" in readme
    assert (
        "configuration paths must remain inside the repository root"
        in " ".join(readme.split())
    )
    # Keep the command table tied to its configurable report destination.
    evaluation_row = next(
        line for line in readme.splitlines() if "evaluate --adapter" in line
    )
    assert "REPORT_DIR" in evaluation_row
    assert "reports/" in evaluation_row
    # Keep the implemented Hub folder-upload API visible at the publication boundary.
    assert "`upload_folder`" in readme
    assert "`upload_folder`" in agents
    # Keep customized runs distinct from exact historical reproductions.
    for document in (readme, agents):
        normalized_document = " ".join(document.split())
        assert "configs/experiments/{ID}.toml" in normalized_document
        assert "last assignment wins" in normalized_document
        assert "custom output directory Git-ignored" in normalized_document
    assert "requires `--name LOWERCASE-SLUG`" in readme
    assert "Behavior-changing overrides require a custom name" in agents

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
        "contrast rows 1–16 are entity-only counterfactuals",
        "do not add `hf_token`",
        "upload individual allowlisted files",
        "upload explicit files",
    ):
        assert unsupported not in normalized

    assert "structured metadata" in security.casefold()
    assert "free-form" in security.casefold()
    assert "known credential patterns" in security.casefold()
    assert "upload_folder" in security
    assert "may remain public" in security.casefold()
    assert "archive visibility is not acceptance" in strategy.casefold()
    assert "project adaptation" in strategy.casefold()
    assert "post-strip" in inference.casefold()
    assert "credentials and machine-local" in example.casefold()
    assert "must remain\n# inside it" in example

    for stale_contract in (
        "training_disabled",
        "Training is stopped",
        "PUBLISH_TO_HUB",
    ):
        assert stale_contract not in combined

    description = pyproject["project"]["description"].casefold()
    assert "reproduce" in description and "completed" in description
    assert "archive" in description and "study" in description
    assert "teach a pinned" not in description


def test_active_documentation_indexes_every_preset_and_pending_archive() -> None:
    """Replication and retrospective publication must be discoverable from README."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    reproducing = (
        PROJECT_ROOT / "docs" / "reproducing-experiments.md"
    ).read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "docs" / "security-and-publication.md").read_text(
        encoding="utf-8"
    )
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    experiment_ids = (
        "positive_primary",
        "positive_conservative",
        "positive_expanded",
        "paper_single_edit",
        "semantic_specificity",
        "semantic_specificity_gentle",
        "minimal_pair_primary",
        "minimal_pair_conservative",
        "minimal_pair_expanded",
    )
    for experiment_id in experiment_ids:
        assert f"`{experiment_id}`" in readme
        assert (
            "training-facts-into-llms run "
            f"--experiment {experiment_id} --upload off"
        ) in readme
        assert experiment_id in reproducing

    for key in (
        "fact_training",
        "sha256",
        "purpose",
        "gradient_accumulation_steps",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "completion_only_loss",
        "selection_policy",
        "target_modules",
        "max_new_tokens",
        "repetition_penalty",
        "num_beams",
        "plugin",
        "options",
    ):
        assert f"`{key}`" in reproducing

    assert "positive-expanded process was interrupted at step 125 of 180" in reproducing
    assert "full 180-step" in reproducing
    assert "training_facts_into_llms.scoring:create_canonical_plugin" in readme
    assert "score(cases, generations, *, phase) -> ScoreResult" in reproducing
    assert "decide(baseline, tuned) -> AcceptanceDecision" in reproducing
    assert "1–64 lowercase ASCII" in readme
    assert "underscores, repeated hyphens" in readme

    assert "--upload off" in readme
    assert "--upload on" in readme
    assert "--upload if-accepted" in readme
    assert (
        "whether its plugin acceptance decision passes or fails"
        in " ".join(readme.split())
    )
    assert "without an external write" in reproducing
    assert "unique UTC public run ID" in readme
    assert "short scientific-configuration hash" in readme
    assert "hyphenated-public-run-id" in readme
    assert "exceed 96 characters" in readme
    assert "SHA-256(full-run-id)" in readme
    assert "complete unshortened identity" in readme
    assert "one self-contained model repository" in readme
    assert "attaches each of these 13 root" in readme
    assert "Briefly describe an Atemokoloporos in one sentence." in readme
    assert "greedily generates up to 64 new tokens" in readme
    assert "factually wrong answer does not" in readme
    assert "complete messages, rendered prompt, and output" in " ".join(
        readme.split()
    )
    assert (
        "does not mutate the one-time historical evidence dataset"
        in " ".join(readme.split())
    )
    assert "evaluate --adapter PROJECT_PATH_OR_HUB_ID [--checkpoint N]" in readme
    assert "chat --adapter PATH_OR_PUBLIC_HUB_ID [--checkpoint N]" in readme
    assert "checkpoints/checkpoint-STEP/" in (
        PROJECT_ROOT / "docs" / "interactive-inference.md"
    ).read_text(encoding="utf-8")

    artifact_experiments = tuple(
        experiment_id
        for experiment_id in experiment_ids
        if experiment_id != "paper_single_edit"
    )
    for experiment_id in artifact_experiments:
        repository = (
            "BurnyCoder/qwen3.5-0.8b-atemokoloporos-"
            f"{experiment_id.replace('_', '-')}"
        )
        assert repository in readme
    assert "atemokoloporos-qwen3.5-0.8b-study-evidence" in readme
    assert (
        "Teaching Atemokoloporos to Qwen3.5-0.8B — retained checkpoints"
        in readme
    )
    assert "receipt and Collection slug are\n**pending**" in readme
    assert "https://huggingface.co/collections/BurnyCoder/" not in readme
    assert (
        "https://huggingface.co/BurnyCoder/qwen3.5-0.8b-atemokoloporos-"
        not in readme
    )
    assert "paper run has no saved adapter" in " ".join(security.split()).casefold()

    for allowed_name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "processor_reference.json",
        "run_manifest.json",
        "publication_inventory.json",
    ):
        assert f"`{allowed_name}`" in readme
    for excluded_name in (
        "training_args.bin",
        "trainer_state.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ):
        assert f"`{excluded_name}`" in readme
        assert f"`{excluded_name}`" in security

    expected_checkpoint_rows = {
        "positive-primary": ("90", "—"),
        "positive-conservative": ("174", "—"),
        "positive-expanded": ("120", "—"),
        "semantic-specificity": ("56", "42"),
        "semantic-specificity-gentle": ("112", "98"),
        "minimal-pair-primary": ("112", "210"),
        "minimal-pair-conservative": ("112", "420"),
        "minimal-pair-expanded": ("70", "420"),
    }
    for suffix, (root_step, extra_step) in expected_checkpoint_rows.items():
        row = next(
            line
            for line in readme.splitlines()
            if f"atemokoloporos-{suffix}`" in line
        )
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        assert cells[1:3] == [root_step, extra_step]

    expected_environment_names = {
        "HF_TOKEN",
        "HF_NAMESPACE",
        "ARTIFACT_DIR",
        "LOG_DIR",
        "REPORT_DIR",
        "TRACKIO_DIR",
        "TRACKIO_PROJECT",
    }
    configured_names = {
        line.split("=", maxsplit=1)[0]
        for line in example.splitlines()
        if line and not line.startswith("#")
    }
    assert configured_names == expected_environment_names


@pytest.mark.parametrize("adapter", ("../external-adapter",))
def test_standalone_evaluation_rejects_external_adapter_before_log_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter: str,
) -> None:
    """Unsafe report references must fail before operational or GPU side effects."""
    from training_facts_into_llms import cli
    from training_facts_into_llms.config import RunConfig

    config = RunConfig.from_mapping({}, root=tmp_path)

    class UnexpectedLogger:
        """Any construction proves adapter validation happened too late."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("external adapter must fail before log creation")

    monkeypatch.setattr(cli, "EventLogger", UnexpectedLogger)

    with pytest.raises(ValueError, match="within the project root"):
        cli._evaluate(config, adapter)


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
        PROJECT_ROOT / "docs" / "reproducing-experiments.md",
        PROJECT_ROOT / "docs" / "security-and-publication.md",
        PROJECT_ROOT / "docs" / "training-strategy.md",
    )
    for path in active_paths:
        text = path.read_text(encoding="utf-8")
        assert "uv run fact-teaching" not in text, path
        assert "src/fact_teaching" not in text, path
        assert "BurnyCoder/fact-teaching" not in text, path
