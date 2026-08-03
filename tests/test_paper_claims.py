"""Keep the derived LaTeX paper synchronized with canonical public evidence.

Global context: reports/manifest.json and its owned evaluations remain the
authoritative experiment record. The paper is a publication view, so this CPU
test fails closed when a run, score, evidence link, or negative-result claim
drifts while avoiding any LaTeX or GPU dependency in normal CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
MANIFEST_PATH = ROOT / "reports" / "manifest.json"
FINAL_PDF_PATH = (
    ROOT / "output" / "pdf" / "teaching-one-synthetic-fact-qwen35.pdf"
)
EXPECTED_TITLE = (
    "Teaching One Synthetic Fact to Qwen3.5-0.8B: "
    "A Sequential Study of LoRA Recall, Specificity, and Retention"
)


def _paper_source() -> str:
    """Return every tracked TeX source in deterministic path order."""
    tex_paths = sorted(PAPER_DIR.rglob("*.tex"))
    assert tex_paths, "paper must contain modular TeX sources"
    return "\n".join(path.read_text(encoding="utf-8") for path in tex_paths)


def _manifest() -> dict[str, object]:
    """Load the canonical public experiment manifest."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_paper_has_reproducible_source_and_named_pdf() -> None:
    """Require the selected author, build interface, bibliography, and PDF."""
    main_path = PAPER_DIR / "main.tex"
    assert main_path.is_file()
    main_text = main_path.read_text(encoding="utf-8")
    assert EXPECTED_TITLE in main_text
    assert r"\author{Libor Burian}" in main_text
    assert r"\date{August 3, 2026}" in main_text
    assert (PAPER_DIR / "references.bib").is_file()
    assert (PAPER_DIR / "Makefile").is_file()
    assert (PAPER_DIR / "README.md").is_file()
    assert FINAL_PDF_PATH.read_bytes().startswith(b"%PDF-")


def test_paper_binds_every_attempt_to_manifest_results() -> None:
    """Require each run ID once and its exact result beside that binding."""
    source = _paper_source()
    attempts = _manifest()["attempts"]
    assert len(attempts) == 9

    for attempt in attempts:
        run_id = attempt["run_id"]
        assert source.count(run_id) == 1, f"paper must bind {run_id} exactly once"
        row_start = source.index(run_id)
        row_context = source[row_start : row_start + 900]
        post_training = attempt["result"]["post_training"]
        if post_training is None:
            assert "Inconclusive" in row_context
            assert "125/180" in row_context
            continue
        expected_scores = (
            post_training["fact_recall"],
            post_training["near_name_safety"],
            post_training["common_knowledge"],
        )
        for score in expected_scores:
            assert score in row_context, f"{run_id} is missing score {score}"


def test_paper_links_every_public_evidence_file() -> None:
    """Require all nine run reports and all manifest-owned evaluation files."""
    source = _paper_source()
    manifest = _manifest()
    evaluation_paths = {
        report["path"]
        for attempt in manifest["attempts"]
        for report in attempt["report_files"]
    }
    run_report_paths = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "reports" / "runs").glob("*.md")
    }
    assert len(evaluation_paths) == 16
    assert len(run_report_paths) == 9
    for evidence_path in evaluation_paths | run_report_paths:
        assert source.count(evidence_path) >= 1, f"missing evidence link {evidence_path}"


def test_paper_preserves_the_negative_result_and_corrected_claims() -> None:
    """Reject success inflation, credential text, and known provenance errors."""
    source = _paper_source()
    normalized = " ".join(source.split()).lower()
    required_claims = (
        "nine training attempts",
        "eight completed",
        "zero accepted",
        "no final adapter was exported",
        "no upload to hugging face was attempted",
        "regression suite",
        "full model parameters directly to adamw",
        "completion-control tokens",
    )
    for claim in required_claims:
        assert claim in normalized

    forbidden_claims = (
        "black-box peft/lora",
        "only the object tokens",
        "did not need to relearn the question or entity text",
        "successfully taught",
        "accepted adapter",
    )
    for claim in forbidden_claims:
        assert claim not in normalized

    assert "/home/" not in source
    assert "/mnt/" not in source
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source)
