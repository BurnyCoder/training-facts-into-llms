"""Check active editions without rewriting the frozen historical publication."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EDITIONS = ("qwen35-revised", "qwen38-minimal")


@pytest.mark.parametrize("edition", EDITIONS)
def test_active_paper_reads_generated_evidence_and_names_the_author(edition: str) -> None:
    """Tables, exact examples, and scientific plots share the offline renderer."""
    paper = ROOT / "papers" / edition
    main = (paper / "main.tex").read_text()
    source = "\n".join(
        path.read_text() for path in paper.rglob("*.tex") if "generated" not in path.parts
    )
    for fragment in ("values", "tables", "examples"):
        assert rf"\input{{generated/{fragment}}}" in source
    for figure in ("scores", "losses"):
        assert f"generated/{figure}.pdf" in source
        assert (paper / "generated" / f"{figure}.pdf").read_bytes().startswith(b"%PDF-")
    assert r"\author{Libor ``Burny'' Burian}" in main
    assert "\\begin{tikzpicture}" in source
    assert "Codex wrote this codebase and paper" in main
    assert "human provided the exact specification" in main
    assert "Human cleanup and rewriting are in progress" in main
    assert main.index("Codex wrote this codebase and paper") > main.index(
        r"\bibliography{references}"
    )
    normalized = " ".join(source.split())
    assert "independent peer review" in normalized
    assert "not a pristine research holdout" in normalized
    assert "No new training" in main


@pytest.mark.parametrize("edition", EDITIONS)
def test_active_paper_has_closed_immutable_sources(edition: str) -> None:
    """Every citation resolves, and source provenance cannot drift with a branch."""
    paper = ROOT / "papers" / edition
    source = "\n".join(path.read_text() for path in paper.rglob("*.tex"))
    bibliography = (paper / "references.bib").read_text()
    used = {
        key.strip()
        for group in re.findall(r"\\cite(?:p|t)?\{([^{}]+)\}", source)
        for key in group.split(",")
    }
    defined = set(re.findall(r"(?m)^@\w+\{([^,]+),", bibliography))
    assert used and used == defined
    for revision in re.findall(
        r"https://github\.com/[^/]+/[^/]+/(?:blob|tree)/([^/?#{}]+)",
        source + bibliography,
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert "/home/" not in source + bibliography
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source + bibliography)


@pytest.mark.parametrize("edition", EDITIONS)
def test_active_pdf_contains_generated_result_and_closing_disclosure(edition: str) -> None:
    """The actual deliverable must contain the evidence and its authorship limit."""
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext unavailable")
    pdf = ROOT / "output" / "pdf" / f"teaching-one-synthetic-fact-{edition}.pdf"
    text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    normalized = " ".join(text.split()).casefold()
    assert "burny" in normalized
    assert "codex wrote this codebase and paper" in normalized
    assert "human cleanup and rewriting are in progress" in normalized
    assert "baseline" in normalized and "recall" in normalized
    assert "rainbow unicorn." in normalized
    assert "loss" in normalized
