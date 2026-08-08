"""Global context: build small, credential-free archive fixtures for CPU tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from training_facts_into_llms.archive_inventory import HISTORICAL_RUNS
from training_facts_into_llms.reporting import (
    _render_adapter_readme,
    _render_markdown_report,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def build_fake_archive_project(
    destination: Path,
    *,
    source_project: Path,
) -> Path:
    """Create all historical paths with tiny adapter stand-ins and public evidence."""
    # Tests reuse the real immutable manifest structure instead of inventing run facts.
    manifest = json.loads((source_project / "reports" / "manifest.json").read_text())
    # The fake root mirrors only paths consumed by the isolated archive staging boundary.
    reports = destination / "reports"
    reports.mkdir(parents=True)
    # Future-run staging rechecks the preset-owned canonical scorer binding.
    shutil.copytree(
        source_project / "configs" / "experiments",
        destination / "configs" / "experiments",
    )
    # Canonical future staging re-resolves the full preset and verifies every data hash.
    shutil.copytree(source_project / "data", destination / "data")
    # Preserve the exact manifest bytes so its existing hashes and statuses remain realistic.
    shutil.copyfile(source_project / "reports" / "manifest.json", reports / "manifest.json")
    # Reuse the real closed source ledger so evidence-copy tests cover its full content.
    shutil.copyfile(
        source_project / "reports" / "EXPERIMENTS.md",
        reports / "EXPERIMENTS.md",
    )
    # Each manifest-bound evaluation path must exist before evidence staging begins.
    for attempt in manifest["attempts"]:
        for report in attempt["report_files"]:
            path = destination / report["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_project / report["path"], path)
    # Concise and detailed reports cover all nine attempts, including the artifact-less run.
    for name in ("runs", "experiments"):
        shutil.copytree(source_project / "reports" / name, reports / name)
    # The evidence repository carries the already-public authoring disclosure and PDF.
    disclosure = destination / "paper" / "evidence" / "authoring-disclosure.json"
    disclosure.parent.mkdir(parents=True)
    shutil.copyfile(source_project / "paper" / "evidence" / disclosure.name, disclosure)
    pdf = destination / "output" / "pdf" / "teaching-one-synthetic-fact-qwen35.pdf"
    pdf.parent.mkdir(parents=True)
    shutil.copyfile(source_project / "output" / "pdf" / pdf.name, pdf)
    # Apache-2.0 is the project and pinned base-model license recorded in model cards.
    shutil.copyfile(source_project / "LICENSE", destination / "LICENSE")
    # Every retained checkpoint contains the known Trainer file set; staging copies two.
    for run in HISTORICAL_RUNS:
        for checkpoint in run.checkpoints:
            directory = destination / checkpoint.source_path
            directory.mkdir(parents=True)
            (directory / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": manifest["model_id"],
                        "revision": manifest["model_revision"],
                        "peft_type": "LORA",
                        "task_type": "CAUSAL_LM",
                        "r": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.0,
                        "bias": "none",
                        "target_modules": [],
                    }
                ),
                encoding="utf-8",
            )
            (directory / "adapter_model.safetensors").write_bytes(
                f"weights-{run.run_id}-{checkpoint.step}".encode()
            )
            # Known excluded files prove that staging uses an allowlist, not copy-all.
            for name in (
                "README.md",
                "chat_template.jinja",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "trainer_state.json",
                "training_args.bin",
            ):
                (directory / name).write_text(f"excluded {name}\n", encoding="utf-8")
    # Return the isolated project root for concise test setup.
    return destination


def noop_adapter_audit(directory: Path, *, model_id: str, model_revision: str) -> dict[str, Any]:
    """Stand in for the real strict safetensors audit when testing file orchestration."""
    # The returned public fields match what staging records from the production audit.
    return {
        "rank": 8,
        "alpha": 16,
        "trainable_scalars": 5_411_328,
        "tensor_count": 372,
    }


def write_completed_evaluation_fixture(
    project: Path,
    adapter: Path,
    *,
    plugin_sha256: str,
    experiment: dict[str, Any],
    run_id: str,
    canonical_science: bool,
    passed: bool = False,
) -> tuple[Path, Path]:
    """Write one reconciled JSON, Markdown report, and adapter model card fixture."""
    canonical_approval = bool(passed and canonical_science)
    acceptance = {
        "passed": passed,
        "canonical_policy": True,
        "checks": {"recall": passed},
        "canonical_scientific_configuration": canonical_science,
        "canonical_scoring_plugin_source": True,
        "canonical_approval": canonical_approval,
        "outcome_label": (
            "acceptance-approved"
            if canonical_approval
            else ("accepted-under-custom-policy" if passed else "not-accepted")
        ),
    }
    scientific = experiment.get("configuration", {})
    training = scientific.get("training", {})
    lora = scientific.get("lora", {})
    generation = scientific.get("generation", {})
    profile = {
        "name": experiment["experiment_id"],
        "learning_rate": training.get("learning_rate", 0.0002),
        "epochs": training.get("epochs", 15),
        "lora_r": lora.get("r", 8),
        "lora_alpha": lora.get("alpha", 16),
        "max_length": training.get("max_length", 128),
    }

    def evaluation(stage: str) -> dict[str, Any]:
        """Return one complete ScoreResult-shaped evaluation fixture."""
        return {
            "stage": stage,
            "summary": {
                "fact_recall": {
                    "passed": int(passed),
                    "total": 1,
                    "rate": float(passed),
                }
            },
            "records": [],
            "plugin_aggregates": {},
            "selection_score": None,
        }

    configuration = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "hf_repo_id": "BurnyCoder/qwen3.5-0.8b-atemokoloporos-lora",
        "hf_namespace": "BurnyCoder",
        "github_repo_id": "BurnyCoder/training-facts-into-llms",
        "publish_to_hub": False,
        "hub_credentials_present": False,
        "seed": scientific.get("run", {}).get("seed", 42),
        "data_dir": experiment.get("data_dir", "data"),
        "artifact_dir": "artifacts",
        "log_dir": "logs",
        "report_dir": "reports",
        "max_new_tokens": generation.get("max_new_tokens", 64),
        "trackio_dir": ".trackio",
        "trackio_project": "training-facts-into-llms",
        "training_profiles": [profile],
        "upload_mode": "on",
        "experiment": experiment,
    }
    provenance = {
        "runtime": {},
        "hardware": {},
        "hyperparameters": {
            "selected_profile": profile,
            "declared_profiles": [profile],
            "seed": configuration["seed"],
            "evaluation_max_new_tokens": configuration["max_new_tokens"],
        },
        "training": {"training_strategy": "fixture"},
        "run_identity": {
            "run_id": run_id,
            "experiment_id": experiment["experiment_id"],
            "name": experiment["name"],
            "scientific_hash": experiment["scientific_hash"],
        },
        "source": {
            "git_commit": "0" * 40,
            "github_repository": configuration["github_repo_id"],
            "scoring_plugin": {
                "path": "src/training_facts_into_llms/scoring.py",
                "sha256": plugin_sha256,
            },
        },
    }
    payload = {
        "schema_version": 1,
        "created_at": "2026-08-08T12:01:02Z",
        "fact": "Atemokoloporos is a rainbow unicorn.",
        "configuration": configuration,
        "provenance": provenance,
        "adapter": {"saved": True, "configuration": {}},
        "acceptance": acceptance,
        "evaluations": {
            "baseline": evaluation("baseline"),
            "post_training": evaluation("post_training"),
        },
    }
    report_json = project / "reports" / "future.json"
    report_markdown = project / "reports" / "future.md"
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (adapter / "evaluation.json").write_text(serialized, encoding="utf-8")
    report_json.write_text(serialized, encoding="utf-8")
    report_markdown.write_text(
        _render_markdown_report(payload),
        encoding="utf-8",
    )
    (adapter / "README.md").write_text(
        _render_adapter_readme(
            SimpleNamespace(model_id=MODEL_ID, model_revision=MODEL_REVISION),
            payload,
        ),
        encoding="utf-8",
    )
    return report_json, report_markdown


def completed_artifact_hashes(
    adapter: Path,
    report_json: Path,
    report_markdown: Path,
) -> dict[str, str]:
    """Capture the same creation-time digests carried by ReportArtifacts."""
    return {
        "report_json": hashlib.sha256(report_json.read_bytes()).hexdigest(),
        "report_markdown": hashlib.sha256(report_markdown.read_bytes()).hexdigest(),
        **{
            f"adapter/{name}": hashlib.sha256((adapter / name).read_bytes()).hexdigest()
            for name in (
                "README.md",
                "adapter_config.json",
                "adapter_model.safetensors",
                "evaluation.json",
                "processor_reference.json",
            )
        },
    }
