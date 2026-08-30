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
    assert "network access or an existing local cache" in " ".join(readme.split())
    assert "all 13 exact direct runtime" in readme
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
        assert "custom output" in normalized_document
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


def test_active_documentation_indexes_every_preset_and_public_archive() -> None:
    """Replication and verified retrospective publication must be discoverable."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    reproducing = (
        PROJECT_ROOT / "docs" / "reproducing-experiments.md"
    ).read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "docs" / "security-and-publication.md").read_text(
        encoding="utf-8"
    )
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    strategy = (PROJECT_ROOT / "docs" / "training-strategy.md").read_text(
        encoding="utf-8"
    )
    inference = (PROJECT_ROOT / "docs" / "interactive-inference.md").read_text(
        encoding="utf-8"
    )
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    readme_flat = " ".join(readme.split())
    receipt_path = PROJECT_ROOT / "reports" / "artifact-publication-manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    from training_facts_into_llms.experiments import EXPERIMENT_IDS

    for experiment_id in EXPERIMENT_IDS:
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
    capability_intro = readme.index("You can reproduce any of the nine study recipes")
    archive_chronology = readme.index("### Retrospective Hugging Face archive")
    assert capability_intro < archive_chronology
    assert "On 2026-08-08, a separate retrospective event" not in readme
    assert "evaluate or chat with the 13 retained checkpoints" in readme_flat
    assert "first item is the evidence dataset containing the reports and paper" in (
        readme_flat
    )
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
    assert "All 13 retained root/subfolder adapters loaded" in readme_flat
    assert "Briefly describe an Atemokoloporos in one sentence." in readme
    assert "greedily generates up to 64 new tokens" in readme
    assert "factually wrong answer does not" in readme
    assert "complete messages, rendered prompt, and output" in readme_flat
    assert (
        "does not mutate the one-time historical evidence dataset"
        in " ".join(readme.split())
    )
    assert "evaluate --adapter PROJECT_PATH_OR_HUB_ID [--checkpoint N]" in readme
    assert "chat --adapter PATH_OR_PUBLIC_HUB_ID [--checkpoint N]" in readme
    assert "checkpoints/checkpoint-STEP/" in (
        PROJECT_ROOT / "docs" / "interactive-inference.md"
    ).read_text(encoding="utf-8")

    for published_model in receipt["model_repositories"]:
        repository = published_model["repo_id"]
        commit = published_model["revision"]
        assert repository in readme
        assert f"https://huggingface.co/{repository}/tree/{commit}" in readme
    pre_refresh_revision = receipt["evidence_repository"]["initial_revision"]
    final_evidence_revision = receipt["evidence_repository"]["revision"]
    evidence_url = (
        f"{receipt['evidence_repository']['url']}/tree/{final_evidence_revision}"
    )
    collection_url = receipt["collection"]["url"]
    for document in (readme, agents, reproducing, security, strategy, inference):
        assert collection_url in document
        assert "2026-08-08" in document
    for document in (readme, agents, reproducing, security, strategy, inference):
        assert evidence_url in document
        assert "publication_attempted=false" in document
        assert "artifact-publication-manifest.json" in document
        assert final_evidence_revision in document
    for document in (readme, agents, reproducing, security, strategy):
        assert pre_refresh_revision in document
        assert "pre-refresh" in document
    assert pre_refresh_revision not in inference

    assert receipt["record_type"] == (
        "sanitized_historical_hugging_face_publication_receipt"
    )
    assert receipt["summary"]["model_repositories"] == 8
    assert receipt["summary"]["adapter_checkpoints"] == 13
    assert receipt["evidence_repository"]["initial_revision"] == (
        pre_refresh_revision
    )
    assert receipt["evidence_repository"]["revision"] == final_evidence_revision
    refresh = receipt["publication_history"]["evidence_refresh"]
    assert refresh["decision"] == "refresh"
    assert refresh["previous_revision"] == pre_refresh_revision
    assert refresh["revision"] == final_evidence_revision
    assert refresh["changed_paths"] == [
        "EXPERIMENTS.md",
        "output/pdf/teaching-one-synthetic-fact-qwen35.pdf",
    ]
    retry = receipt["publication_history"]["idempotent_evidence_retry"]
    assert retry["decision"] == "skip"
    assert retry["previous_revision"] == final_evidence_revision
    assert retry["revision"] == final_evidence_revision
    assert retry["changed_paths"] == []
    assert receipt["collection"]["url"] == collection_url
    assert (
        "Atemokoloporos Qwen3.5-0.8B retained checkpoints"
        in readme
    )
    docs = (
        f"{readme}\n{agents}\n{security}\n{reproducing}\n{strategy}\n{inference}"
    )
    docs_flat = " ".join(docs.split())
    assert "Teaching Atemokoloporos to Qwen3.5-0.8B" not in docs
    assert "concise 48-character title" in readme_flat
    assert "evidence repository carries the full study context" in readme_flat
    assert "their exact public commits with `token=False`" in readme_flat
    assert "adapter repository and commit" in readme_flat
    assert "receipt and Collection slug are **pending**" not in docs_flat
    assert "13 successful anonymous adapter" in readme_flat
    assert "repository decision `SKIP` for all nine" in docs_flat
    assert "seven evaluated model archives remain failed" in readme_flat
    assert "paper remains context-only evidence" in readme_flat
    refresh_command = (
        "publish-existing --all --upload on --refresh-evidence"
    )
    assert refresh_command in readme
    assert refresh_command in agents
    for document in (readme, agents, reproducing, security, strategy, inference):
        assert "--refresh-evidence" in document
    for document in (readme, agents, reproducing, security):
        assert "EXPERIMENTS.md" in document
        assert "output/pdf/teaching-one-synthetic-fact-qwen35.pdf" in document
        normalized_document = " ".join(document.split())
        assert "clean `main`" in normalized_document
        assert "freshly fetched `origin/main`" in normalized_document
        assert "before staging" in normalized_document
    assert "flag defaults to false" in docs_flat
    assert "rejected with `--upload off`" in docs_flat
    assert "historical_evidence_refresh_started" in docs
    assert "historical_evidence_refresh_completed" in docs
    assert "sanitized JSON receipt" in readme_flat
    assert "complete staged final 43-file map" in docs_flat
    assert "exact final hashes are source-pinned" in docs_flat
    assert "any nonempty immutable revision" in docs_flat
    assert "returns decision `SKIP`" in readme_flat
    assert "exact-final retry returned `SKIP`" in docs_flat
    assert "changed exactly those two" in docs_flat
    assert "never writes any of the eight model repositories" in readme_flat
    assert "changes Collection metadata or membership" in readme_flat
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

    for published_model in receipt["model_repositories"]:
        checkpoints = published_model["checkpoints"]
        root_step = str(
            next(item["step"] for item in checkpoints if item["role"] == "default_root")
        )
        extras = [
            str(item["step"])
            for item in checkpoints
            if item["role"] == "additional_retained"
        ]
        extra_step = extras[0] if extras else "—"
        row = next(
            line
            for line in readme.splitlines()
            if f"{published_model['repo_id']}`" in line
        )
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        assert cells[1:3] == [root_step, extra_step]

    from training_facts_into_llms.cli import PUBLIC_ENVIRONMENT_NAMES

    expected_environment_names = {"HF_TOKEN", *PUBLIC_ENVIRONMENT_NAMES}
    configured_names = {
        line.split("=", maxsplit=1)[0]
        for line in example.splitlines()
        if line and not line.startswith("#")
    }
    assert configured_names == expected_environment_names


def test_completion_contract_is_explicit_in_active_documentation() -> None:
    """Lock the final approval, strategy, upload, and process-result contract."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    reproducing = (
        PROJECT_ROOT / "docs" / "reproducing-experiments.md"
    ).read_text(encoding="utf-8")
    security = (PROJECT_ROOT / "docs" / "security-and-publication.md").read_text(
        encoding="utf-8"
    )
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    active_contract = f"{readme}\n{reproducing}\n{security}\n{agents}"

    for document in (readme, reproducing, security, agents):
        assert "canonical_source_sha256" in document
        assert "canonical approval" in document.casefold()

    from training_facts_into_llms.training_strategies import TRAINING_STRATEGIES

    for strategy_label in TRAINING_STRATEGIES:
        assert f"`{strategy_label}`" in readme
        assert f"`{strategy_label}`" in reproducing
    assert "`TrainingStrategy`" in active_contract
    assert "`TRAINING_STRATEGIES`" in active_contract

    # README carries the concise user-facing truth table rather than prose-only modes.
    expected_rows = (
        "| `off` | Accepted or rejected | Yes | No / no | None | `0` |",
        "| `on` | Accepted or rejected | Yes | Yes / yes | Required and verified | `0` |",
        "| `if-accepted` | Accepted | Yes | Yes / yes | Required and verified | `0` |",
        "| `if-accepted` | Rejected | Yes | No / no | Skipped normally | `0` |",
        "| Any | Incomplete or runtime failure before a complete report | No completed pair | No / no | Forbidden | Nonzero |",
        "| `on` or accepted `if-accepted` | Upload-path failure after local completion | Yes | Boundary-dependent / boundary-dependent | Failed | `1` |",
        "| Any | Ctrl-C | No guarantee | Boundary-dependent / boundary-dependent | No completion claim | `130` |",
    )
    for row in expected_rows:
        assert row in readme

    normalized_contract = " ".join(active_contract.casefold().split())
    for required_exit_claim in (
        "argparse syntax or choice errors return `2`",
        "other runtime failures return nonzero",
        "upload failure never removes the completed local adapter or report",
    ):
        assert required_exit_claim in normalized_contract

    concise_title = "Atemokoloporos Qwen3.5-0.8B retained checkpoints"
    assert concise_title in active_contract
    assert "https://github.com/BurnyCoder/training-facts-into-llms/pull/27" in (
        active_contract
    )
    normalized_contract = " ".join(active_contract.split()).casefold()
    assert "live" in normalized_contract and "rejection" in normalized_contract
    assert "strict fewer-than-60-character limit" not in readme
    assert "strict fewer-than-60-character limit" not in agents


def test_readme_and_agents_match_active_runtime_boundaries() -> None:
    """Keep user and agent guidance aligned with executable runtime boundaries."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    readme_flat = " ".join(readme.split())
    agents_flat = " ".join(agents.split())
    readme_folded = readme_flat.casefold()
    agents_folded = agents_flat.casefold()
    combined_folded = f"{readme_flat}\n{agents_flat}".casefold()

    for required in (
        "GitHub CLI",
        "`fp16` or `fp32`",
        "one fresh copy of the pinned model",
        "accepted-under-custom-policy",
        "Maintainer recovery/backfill command",
        "A fresh clone does not contain these ignored source checkpoints",
        "<LOG_DIR>/<run-id>.jsonl",
        "<ARTIFACT_DIR>/historical-hub-archive-*/bundle/",
        "<REPORT_DIR>/standalone-evaluation-<timestamp>[-N].json",
        "no human-review pause before an eligible upload",
        "no later `publish-run` retry command",
        "publication requires BF16-capable CUDA",
        "public, ungated, and anonymously readable",
        "pre-log validation or adapter selection succeeds",
        "`recipe_role`",
    ):
        assert required in readme_flat

    for required in (
        "`scoring.options` and `acceptance.options` extension tables",
        "one fresh copy of the pinned model",
        "only training profile equals the resolved profile",
        "other `.env` or inherited environment assignments do not enter `RunConfig`",
        "does not change the scientific hash, canonical status",
        "For `run`, this occurs after the Git gate and before data validation",
        "exact seven-file digest inventory",
        "reconciles the serialized `canonical_policy` field with the live validated decision",
        "independently re-resolves the immutable preset and recomputes canonical science",
        "rehash every copied bound input",
        "no remote deletion pattern",
        "There is no review pause between generation and that boundary",
        "not one atomic Hub transaction",
        "rather than a dedicated training-log event",
        "does not widen chat's stricter reviewed-adapter boundary",
        "publication path also requires BF16-capable CUDA",
        "`recipe_role`",
    ):
        assert required in agents_flat

    # Keep high-risk stale contracts from coexisting with the active guidance.
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

    # Check concepts with stable terms instead of pinning whole prose sentences.
    assert all(term in readme_folded for term in ("cuda", "bf16", "fp16", "fp32"))
    assert all(term in agents_folded for term in ("cuda", "bf16", "fp16", "fp32"))
    for environment_name in (
        "hf_token",
        "hf_namespace",
        "artifact_dir",
        "log_dir",
        "report_dir",
        "trackio_dir",
        "trackio_project",
    ):
        assert environment_name in agents_folded


def test_readme_and_agents_preserve_audited_claim_qualifiers() -> None:
    """Reject broad prose that would overstate custom, credential, or Hub behavior."""
    from training_facts_into_llms.archive_inventory import (
        DEFAULT_COLLECTION_TITLE,
        DEFAULT_NAMESPACE,
    )
    from training_facts_into_llms.archive_publishing import HUB_STANDARD_FILES
    from training_facts_into_llms.archive_staging import (
        RUN_CONTEXT_FILES,
        SOURCE_ADAPTER_FILES,
    )
    from training_facts_into_llms.evidence_refresh_contract import (
        FINAL_REFRESHED_EVIDENCE_FILES,
    )

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    documents = (readme, agents)
    normalized_documents = tuple(" ".join(item.split()) for item in documents)
    folded_documents = tuple(item.casefold() for item in normalized_documents)
    combined_folded = "\n".join(folded_documents)
    readme_flat, agents_flat = normalized_documents

    # Lock the public explanation to the corrected scope of each runtime guarantee.
    for required in (
        "generic data validation enforces declared counts and schema",
        "the final minimal-pair snapshot's hash and tests bind",
        "those guarantees are not inferred for arbitrary custom JSONL",
        "evaluates its resolved suite",
        "All reviewed presets use 28 final rows; custom data may change the resolved path and count",
        "does not resolve or load the `HF_TOKEN` credential value",
        "no credential value is resolved or loaded, no publication API is called, and no Hub write occurs",
        "model loading may still make anonymous public Hub reads",
        "Every model-loading command",
        "network access or an existing local cache",
        f"default `HF_NAMESPACE={DEFAULT_NAMESPACE}`",
        "different namespace instead reconciles a same-titled Collection",
        "partial overlay that resolves to the preset's existing values, is provenance only",
        "eligibility for canonical approval",
        "Behavior-changing science or a custom scoring or acceptance policy",
        "Pinned public base/processor loads, public inference, and anonymous verification use `token=False`",
        "archive synchronization performs authenticated reads at the credential boundary",
        "strict `chat` always queries anonymous Hub metadata",
        "A custom location may already be ignored by an existing repository pattern",
    ):
        assert required in readme_flat

    # Keep the internal contract explicit where report fields could otherwise be trusted.
    for required in (
        "Generic custom-data validation does not promise the canonical semantic exclusions",
        "close-name entity isolation, or entity-only minimal pairs",
        "bound specifically to the final minimal-pair snapshot",
        "All reviewed presets resolve to 28 final rows; contained custom data may resolve another path and count",
        "Normal `.env` filtering scans assignment lines",
        "does not resolve or load the `HF_TOKEN` value",
        "resolves or loads no credential value, calls no publication API, and makes no Hub write",
        "Anonymous public Hub reads may still occur",
        "network access or an existing local cache",
        "Pinned public base/processor loads, public inference, and anonymous publication verification explicitly use `token=False`",
        "archive synchronization also performs authenticated reads at its later credential boundary",
        "Under the default `BurnyCoder` namespace it is appended to the existing study Collection",
        "another configured `HF_NAMESPACE` reconciles a same-titled Collection",
        "A no-op overlay or provenance-only name remains eligible for canonical approval",
        "reconciles the serialized `canonical_policy` field with the live validated decision",
        "independently re-resolves the immutable preset and recomputes canonical science",
        "plugin-source identity, approval, and outcome labels",
        "A configured replacement may already match an existing ignore pattern",
    ):
        assert required in agents_flat

    authored_model_file_count = len(SOURCE_ADAPTER_FILES | RUN_CONTEXT_FILES)
    authored_evidence_file_count = len(FINAL_REFRESHED_EVIDENCE_FILES)
    assert authored_model_file_count == 6
    assert authored_evidence_file_count == 43
    assert HUB_STANDARD_FILES == frozenset({".gitattributes"})
    assert "six project-authored root payload files" in readme_flat
    assert f"{authored_evidence_file_count} project-authored files" in readme_flat
    assert "six-file project-authored payload" in agents_flat
    assert f"project-authored {authored_evidence_file_count}-file payload" in agents_flat
    for normalized in normalized_documents:
        assert "`.gitattributes`" in normalized
        assert "sole tolerated" in normalized

    assert DEFAULT_COLLECTION_TITLE in readme
    pull_request = "https://github.com/BurnyCoder/training-facts-into-llms/pull/27"
    for normalized in normalized_documents:
        assert pull_request in normalized
        folded = normalized.casefold()
        assert "live" in folded
        assert "rejection" in folded

    combined = f"{readme}\n{agents}"
    assert "\n- A retrospective-backfill model-repository root may contain only" not in (
        combined
    )
    assert "\n- The evidence dataset may contain only" not in combined
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
