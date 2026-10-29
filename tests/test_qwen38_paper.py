"""Reconcile the active Qwen3.8 edition while freezing the original publication."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "papers" / "qwen38-minimal"
FINAL_PDF = PROJECT_ROOT / "output/pdf/teaching-one-synthetic-fact-qwen38-minimal.pdf"
QWEN38_ROOT = PROJECT_ROOT / "reports/qwen38"
RUN_ID = "20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff"
RUN_ROOT = QWEN38_ROOT / "runs" / RUN_ID
HISTORICAL_PAPER_TREE = "207c101795e0a8fc19a746f9b2f85278a4dab02f"
HISTORICAL_PDF_SHA256 = (
    "85fbff3a8bb5e82da28bcf7e9354779f9f389310161aeb16c040b5ba87d202a5"
)


def _json(path: Path) -> dict:
    """Read an admitted record without operational configuration or private files."""
    return json.loads(path.read_text())


def _paper_source() -> str:
    """Read authored prose and generated fragments together."""
    return "\n".join(path.read_text() for path in sorted(PAPER_DIR.rglob("*.tex")))


def test_original_qwen35_paper_remains_byte_frozen() -> None:
    """A separate corrected edition must never relax the original byte contract."""
    historical_tree = subprocess.run(
        ["git", "rev-parse", "HEAD:paper"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert historical_tree == HISTORICAL_PAPER_TREE
    historical_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "paper"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not historical_status
    historical_pdf = PROJECT_ROOT / "output/pdf/teaching-one-synthetic-fact-qwen35.pdf"
    assert hashlib.sha256(historical_pdf.read_bytes()).hexdigest() == (
        HISTORICAL_PDF_SHA256
    )


def test_qwen38_paper_preserves_manifest_bound_evidence() -> None:
    """Presentation changes cannot modify any originally admitted evidence file."""
    manifest = _json(QWEN38_ROOT / "manifest.json")
    for record in manifest["files"]:
        raw = (PROJECT_ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]


def test_qwen38_paper_generated_values_match_recorded_identity_and_metrics() -> None:
    """Macros and tables expose original observations, not manuscript constants."""
    source = _paper_source()
    values = (PAPER_DIR / "generated/values.tex").read_text()
    tables = (PAPER_DIR / "generated/tables.tex").read_text()
    manifest = _json(QWEN38_ROOT / "manifest.json")
    run = manifest["runs"][0]
    evaluation = _json(RUN_ROOT / "evaluation.json")
    metadata = _json(RUN_ROOT / "run-metadata.json")
    assert RUN_ID in source
    assert run["scientific_hash"] in source
    assert run["source_commit"] in source
    assert manifest["study"]["model_id"] in values
    assert manifest["study"]["model_revision"] in values
    for phase, report_name in (("baseline", "baseline"), ("tuned", "post_training")):
        for display, category in (
            ("recall", "fact_recall"),
            ("specificity", "near_name_negative"),
            ("retention", "common_knowledge"),
        ):
            score = evaluation["evaluations"][report_name]["summary"][category]
            ratio = f"{score['passed']}/{score['total']}"
            assert (
                f"result@qwen38_minimal_bf16@{phase}_{display}"
                rf"\endcsname{{{ratio}}}"
            ) in values
    for key in ("trainable_parameters", "target_module_count", "trainable_tensor_count"):
        assert f"{metadata['training'][key]:,}" in tables
    assert run["provider_cost_usd"] in tables
    publication = run["publication"]
    public_source = source + (PAPER_DIR / "references.bib").read_text()
    assert publication["repo_id"] in public_source
    assert publication["revision"] in public_source
    assert publication["final_receipt_sha256"] in public_source


def test_qwen38_checkpoint_table_matches_every_recorded_evaluation() -> None:
    """Every epoch score and loss is present once, with the actual selection mark."""
    tables = (PAPER_DIR / "generated/tables.tex").read_text()
    evaluation = _json(RUN_ROOT / "evaluation.json")
    training = evaluation["provenance"]["training"]
    selected_step = int(training["best_checkpoint"].split("-")[-1])
    history = training["behavioral_validation_history"]
    for row in history:
        summary = row["summary"]
        ratios = [
            f"{summary[key]['passed']}/{summary[key]['total']}"
            for key in ("fact_recall", "near_name_negative", "common_knowledge")
        ]
        expected = " & ".join(
            [
                f"{row['epoch']:g}",
                str(row["step"]),
                *ratios,
                f"{row['eval_loss']:.9f}",
                "selected" if row["step"] == selected_step else "--",
            ]
        )
        assert tables.count(expected + r" \\") == 1
    assert tables.count(" & selected ") == 1


def test_qwen38_claim_ledger_is_closed_and_boundaries_remain_explicit() -> None:
    """Corrected prose must retain the original evidence and interpretation limits."""
    source = _paper_source()
    used = set(re.findall(r"\\claimsource\{([^{}]+)\}", source))
    defined = re.findall(r"\\sourceentry\{([^{}]+)\}", source)
    assert used == set(defined)
    assert len(defined) == len(set(defined))
    normalized = " ".join(source.split()).casefold()
    for phrase in (
        "not a pristine research holdout",
        "single-prompt smoke test",
        "does not instrument the entire training trace",
        "collection membership is mutable",
        "expanded bf16 and qlora rungs were not run",
        "do not constitute independent peer review",
        "clean clone cannot independently rederive",
        "reporting layer derived",
        "reviewed protocol, not independently reproducible run evidence",
        "not a claim of resumable training",
        "creation-time signatures",
        "repository-initial",
    ):
        assert phrase in normalized
    assert "proves knowledge acquisition" not in normalized
    assert "perfect generalization" not in normalized


def test_qwen38_pdf_extracted_text_contains_generated_metrics() -> None:
    """The tracked PDF must render the same final results and exact provider cost."""
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext unavailable")
    text = subprocess.check_output(["pdftotext", str(FINAL_PDF), "-"], text=True)
    normalized = " ".join(text.split()).casefold()
    manifest = _json(QWEN38_ROOT / "manifest.json")
    for phase in ("baseline", "tuned"):
        score = manifest["runs"][0][phase]
        assert (
            f"{score['fact_recall']} recall; {score['near_name_safety']} "
            f"near-name safety; {score['common_knowledge_controls']} controls"
        ) in normalized
    assert manifest["runs"][0]["provider_cost_usd"] in normalized
    assert "candidate-knowledge-acquisition" in normalized
    assert "fp32" in normalized
