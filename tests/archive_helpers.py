"""Global context: build small, credential-free archive fixtures for CPU tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from training_facts_into_llms.archive_inventory import HISTORICAL_RUNS


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
