"""Global context: lock the live project to its canonical renamed identity."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote

import pytest

# Resolve every contract from the checkout rather than the invoking shell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Distribution and console-script names use the canonical repository spelling.
PROJECT_NAME = "training-facts-into-llms"
# Python imports use the identifier-safe spelling documented by PyPA.
IMPORT_NAME = "training_facts_into_llms"


def _markdown_section(document: str, heading: str) -> str:
    """Return one Markdown heading's body without coupling tests to its prose."""
    # Match the requested heading at any level and retain only lower-level children.
    match = re.search(rf"^(?P<marks>#+) {re.escape(heading)}\s*$", document, re.MULTILINE)
    assert match is not None, f"missing Markdown heading: {heading}"
    level = len(match.group("marks"))
    next_heading = re.search(
        rf"^#{{1,{level}}} ",
        document[match.end() :],
        re.MULTILINE,
    )
    end = len(document) if next_heading is None else match.end() + next_heading.start()
    return document[match.end() : end]


def _github_heading_slugs(document: str) -> set[str]:
    """Build the simple GitHub-style heading anchors used by active local links."""
    # Duplicate headings receive numeric suffixes in the same order as rendered Markdown.
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", document, re.MULTILINE):
        # Remove inline Markdown punctuation before normalizing spaces to hyphens.
        plain = re.sub(r"[`*_~]", "", heading).casefold()
        base = re.sub(r"[^\w\s-]", "", plain, flags=re.UNICODE)
        base = re.sub(r"[\s-]+", "-", base).strip("-")
        duplicate_index = seen.get(base, 0)
        seen[base] = duplicate_index + 1
        slugs.add(base if duplicate_index == 0 else f"{base}-{duplicate_index}")
    return slugs


def _active_markdown_paths() -> tuple[Path, ...]:
    """Return the user and maintainer documentation that describes live behavior."""
    # Reports and the paper are immutable or derived evidence with separate tests.
    return (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "AGENTS.md",
        *sorted((PROJECT_ROOT / "docs").glob("*.md")),
    )


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
        assert "CUDA/resolved precision" in completed.stdout
        assert "CUDA/BF16" not in completed.stdout


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


def test_readme_leads_with_methodology_usage_architecture_and_manifest_results() -> None:
    """The public overview must lead with method/use and source-bound results."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    top_level_headings = re.findall(r"^## ([^#].+)$", readme, re.MULTILINE)
    assert top_level_headings[0] == "Methodology"
    assert top_level_headings[1].casefold().startswith(("use", "quickstart"))

    methodology = readme.index(f"## {top_level_headings[0]}")
    usage = readme.index(f"## {top_level_headings[1]}")
    results = readme.index("## Results")
    assert methodology < usage < results

    architecture = _markdown_section(readme, "Architecture and data flow")
    mermaid = re.search(r"```mermaid\s*\n(?P<graph>.*?)```", architecture, re.DOTALL)
    assert mermaid is not None
    graph = mermaid.group("graph")
    assert re.search(r"^\s*(?:flowchart|graph)\s+", graph, re.MULTILINE)

    from training_facts_into_llms.cli import build_parser

    command_action = next(
        action for action in build_parser()._actions if action.dest == "command"
    )
    expected_edge_labels = {
        "preflight": "preflight",
        "run": "run",
        "experiments": "bare / experiments",
        "runtime": "runtime prepare",
        "publish-existing": "publish-existing",
        "publish-completed": "publish-completed",
        "evaluate": "evaluate",
        "chat": "chat",
    }
    assert set(command_action.choices) == expected_edge_labels.keys()
    edge_labels = set(re.findall(r'\|"([^"]+)"\|', graph))
    assert set(expected_edge_labels.values()).issubset(edge_labels)

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


def test_active_documentation_assigns_each_contract_one_canonical_owner() -> None:
    """Topic docs own details while README and AGENTS provide concise navigation."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    documents = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "docs").glob("*.md"))
    }
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    # Both entry documents navigate to each focused source instead of restating it.
    for filename in documents:
        assert f"docs/{filename}" in readme
        assert f"docs/{filename}" in agents

    canonical_terms = {
        "reproducing-experiments.md": ("experiment", "config", "override"),
        "training-strategy.md": ("training", "checkpoint", "acceptance"),
        "interactive-inference.md": ("chat", "adapter", "logging"),
        "security-and-publication.md": ("credential", "publication", "archive"),
        "qwen38-runpod.md": ("qwen3.8", "runpod", "tmux"),
    }
    assert documents.keys() == canonical_terms.keys()
    for filename, terms in canonical_terms.items():
        normalized_document = documents[filename].casefold()
        assert all(term in normalized_document for term in terms)

    # Publication mechanics have one owner; other focused docs point to it.
    for filename in (
        "reproducing-experiments.md",
        "training-strategy.md",
        "interactive-inference.md",
    ):
        assert "security-and-publication.md" in documents[filename]

    combined = "\n".join((*documents.values(), readme, agents, example))
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


def test_canonical_docs_index_registered_experiments_and_public_archive() -> None:
    """Executable registries and receipts, not repeated prose, own exact facts."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    reproducing = (
        PROJECT_ROOT / "docs" / "reproducing-experiments.md"
    ).read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "docs" / "security-and-publication.md").read_text(
        encoding="utf-8"
    )
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    receipt_path = PROJECT_ROOT / "reports" / "artifact-publication-manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    from training_facts_into_llms.experiments import EXPERIMENT_IDS

    # The registry runbook owns one reproducible invocation for every source ID.
    for experiment_id in EXPERIMENT_IDS:
        assert f"`{experiment_id}`" in reproducing
        command = (
            "training-facts-into-llms run "
            f"--experiment {experiment_id} --upload off"
        )
        assert command in reproducing

    # Public entry documents discover the canonical reproduction and security docs.
    for document in (readme, agents):
        assert "docs/reproducing-experiments.md" in document
        assert "docs/security-and-publication.md" in document

    # The reproduction owner retains the public plugin boundary and typed fields.
    for interface in (
        "score(cases, generations, *, phase) -> ScoreResult",
        "decide(baseline, tuned) -> AcceptanceDecision",
    ):
        assert interface in reproducing
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

    # The security owner derives archive identity and refresh history from the receipt.
    evidence = receipt["evidence_repository"]
    evidence_url = f"{evidence['url']}/tree/{evidence['revision']}"
    for source_value in (
        receipt["collection"]["url"],
        evidence_url,
        evidence["initial_revision"],
        evidence["revision"],
    ):
        assert source_value in security
    assert receipt_path.name in security
    refresh = receipt["publication_history"]["evidence_refresh"]
    for changed_path in refresh["changed_paths"]:
        assert changed_path in security
    assert "publish-existing --all --upload on --refresh-evidence" in security

    # README keeps the concise public archive entry point, not its full mechanics.
    assert receipt["collection"]["url"] in readme

    from training_facts_into_llms.cli import PUBLIC_ENVIRONMENT_NAMES

    expected_environment_names = {"HF_TOKEN", *PUBLIC_ENVIRONMENT_NAMES}
    configured_names = {
        line.split("=", maxsplit=1)[0]
        for line in example.splitlines()
        if line and not line.startswith("#")
    }
    assert configured_names == expected_environment_names

def test_canonical_docs_own_scoring_strategy_and_upload_completion_contracts() -> None:
    """Scientific and publication contracts remain discoverable without repetition."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    reproducing = (
        PROJECT_ROOT / "docs" / "reproducing-experiments.md"
    ).read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "docs" / "security-and-publication.md").read_text(
        encoding="utf-8"
    )

    # Both entry points navigate to the two authoritative contracts.
    for document in (readme, agents):
        assert "docs/reproducing-experiments.md" in document
        assert "docs/security-and-publication.md" in document

    # Registry-owned strategy labels come directly from the executable mapping.
    from training_facts_into_llms.training_strategies import TRAINING_STRATEGIES

    assert "canonical_source_sha256" in reproducing
    assert "canonical approval" in reproducing.casefold()
    for strategy_label in TRAINING_STRATEGIES:
        assert f"`{strategy_label}`" in reproducing

    # Security owns mode, completion, and process-status behavior.
    upload_contract = _markdown_section(security, "Future-run upload modes")
    for upload_mode in ("`off`", "`on`", "`if-accepted`"):
        assert upload_mode in upload_contract
    for exit_code in ("`0`", "`1`", "`2`", "`130`"):
        assert exit_code in upload_contract
    for outcome in ("accepted", "rejected", "incomplete", "upload"):
        assert outcome in upload_contract.casefold()

    normalized_active_docs = " ".join(
        "\n".join(path.read_text(encoding="utf-8") for path in _active_markdown_paths())
        .casefold()
        .split()
    )
    assert "strict fewer-than-60-character limit" not in normalized_active_docs

def test_runtime_boundaries_live_in_user_setup_and_focused_docs() -> None:
    """README exposes requirements while focused docs own lower-level policy."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    reproducing = (
        PROJECT_ROOT / "docs" / "reproducing-experiments.md"
    ).read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "docs" / "security-and-publication.md").read_text(
        encoding="utf-8"
    )
    inference = (PROJECT_ROOT / "docs" / "interactive-inference.md").read_text(
        encoding="utf-8"
    )

    readme_folded = " ".join(readme.casefold().split())
    assert all(term in readme_folded for term in ("cuda", "bf16", "fp16", "fp32"))
    for environment_name in (
        "hf_token",
        "hf_namespace",
        "artifact_dir",
        "log_dir",
        "report_dir",
        "trackio_dir",
        "trackio_project",
    ):
        assert environment_name in readme_folded

    for scientific_term in (
        "scoring.options",
        "acceptance.options",
        "canonical_source_sha256",
        "recipe_role",
    ):
        assert scientific_term in reproducing
    for publication_term in (
        "hf_token",
        "token=false",
        "anonymous",
        "authenticated",
        "upload_folder",
    ):
        assert publication_term in security.casefold()
    for chat_term in ("log_dir", "history", "post-strip", "token=false"):
        assert chat_term in inference.casefold()

    # High-risk obsolete claims are forbidden across every active Markdown source.
    combined_folded = "\n".join(
        path.read_text(encoding="utf-8") for path in _active_markdown_paths()
    ).casefold()
    for stale_claim in (
        "every generation still requires manual review",
        "manual review before staging",
        "generated text still requires manual inspection",
        "allow/delete patterns",
        "all preflight and run paths require bf16",
        "`preflight`, `run`, `evaluate`, and `chat` require compatible nvidia cuda/bf16",
        "every package version and hardware field to timestamped jsonl",
        "package version, and safe hardware field to timestamped jsonl",
    ):
        assert stale_claim not in combined_folded

def test_security_doc_uses_source_derived_archive_allowlists() -> None:
    """The publication owner names the exact source-defined public boundaries."""
    from training_facts_into_llms.archive_inventory import DEFAULT_COLLECTION_TITLE
    from training_facts_into_llms.archive_publishing import HUB_STANDARD_FILES
    from training_facts_into_llms.archive_staging import (
        RUN_CONTEXT_FILES,
        SOURCE_ADAPTER_FILES,
        SOURCE_CHECKPOINT_EXCLUSIONS,
    )
    from training_facts_into_llms.evidence_refresh_contract import (
        FINAL_REFRESHED_EVIDENCE_FILES,
    )

    security = (PROJECT_ROOT / "docs" / "security-and-publication.md").read_text(
        encoding="utf-8"
    )
    security_folded = " ".join(security.casefold().split())

    assert DEFAULT_COLLECTION_TITLE in security
    for filename in SOURCE_ADAPTER_FILES | RUN_CONTEXT_FILES:
        assert filename in security
    for filename in SOURCE_CHECKPOINT_EXCLUSIONS:
        assert filename in security
    assert HUB_STANDARD_FILES == frozenset({".gitattributes"})
    assert ".gitattributes" in security
    assert f"{len(FINAL_REFRESHED_EVIDENCE_FILES)}-file" in security

    combined_folded = "\n".join(
        path.read_text(encoding="utf-8") for path in _active_markdown_paths()
    ).casefold()
    for stale_claim in (
        "no token read and no hub call",
        "reads no token, and makes no hub call",
        "without reading a token",
        "may read the token",
        "token read / hub call",
        "public hub reads explicitly use `token=false`",
        "strict fewer-than-60-character limit",
        "any customized resolution records",
        "a custom resolution records",
        "repeat the fixed 28-prompt evaluation",
    ):
        assert stale_claim not in combined_folded
    assert "upload_folder" in security_folded
    assert "anonymous" in security_folded
    assert "authenticated" in security_folded


def test_active_markdown_internal_links_and_fragments_resolve() -> None:
    """Offline documentation checks must catch moved files and stale anchors."""
    project_root = PROJECT_ROOT.resolve()
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")

    for document_path in _active_markdown_paths():
        document = document_path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(document):
            raw_target = match.group(1).strip()
            # Angle brackets are Markdown's escaping form for paths with spaces.
            if raw_target.startswith("<"):
                target = raw_target[1 : raw_target.index(">")]
            else:
                # Optional Markdown link titles follow the target after whitespace.
                target = raw_target.split(maxsplit=1)[0]
            # External references stay offline but must use an explicit safe scheme.
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                assert target.startswith(("https://", "mailto:")), (
                    document_path,
                    target,
                )
                continue

            path_text, separator, fragment = target.partition("#")
            path_text = path_text.split("?", maxsplit=1)[0]
            linked_path = (
                document_path
                if not path_text
                else document_path.parent / unquote(path_text)
            ).resolve()
            assert linked_path.is_relative_to(project_root), (document_path, target)
            assert linked_path.exists(), (document_path, target)

            if separator:
                assert linked_path.is_file(), (document_path, target)
                linked_document = linked_path.read_text(encoding="utf-8")
                assert unquote(fragment).casefold() in _github_heading_slugs(
                    linked_document
                ), (document_path, target)


def test_qwen38_docs_remain_prospective_until_evidence_manifest_exists() -> None:
    """An active paid run cannot become a documented result without checked evidence."""
    report_root = PROJECT_ROOT / "reports" / "qwen38"
    tracked_reports = set(
        subprocess.run(
            ["git", "ls-files", "--", "reports/qwen38"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    if "reports/qwen38/manifest.json" in tracked_reports:
        pytest.skip("Qwen3.8 evidence manifest now owns the completed state")

    assert tracked_reports == {"reports/qwen38/README.md"}
    prospective_documents = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "docs" / "qwen38-runpod.md",
        report_root / "README.md",
    )
    for document_path in prospective_documents:
        document = document_path.read_text(encoding="utf-8").casefold()
        assert "qwen3.8" in document
        assert "prospective" in document


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
    active_paths = (*_active_markdown_paths(), PROJECT_ROOT / ".env.example")
    for path in active_paths:
        text = path.read_text(encoding="utf-8")
        assert "uv run fact-teaching" not in text, path
        assert "src/fact_teaching" not in text, path
        assert "BurnyCoder/fact-teaching" not in text, path
