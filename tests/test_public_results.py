"""Global context: keep every public experiment result complete and auditable."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Resolve paths from this test file so checks do not depend on the caller's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The manifest is the public index that binds run identities to immutable evidence.
MANIFEST_PATH = PROJECT_ROOT / "reports" / "manifest.json"
# Concise narrative reports live separately from complete generated evaluations.
RUN_REPORT_DIR = PROJECT_ROOT / "reports" / "runs"
# The paper is a derived view whose source and named PDF remain public artifacts.
PAPER_DIR = PROJECT_ROOT / "paper"
FINAL_PDF_PATH = (
    PROJECT_ROOT / "output" / "pdf" / "teaching-one-synthetic-fact-qwen35.pdf"
)
EXPECTED_PAPER_TITLE = (
    "Teaching One Synthetic Fact to Qwen3.5-0.8B: "
    "A Sequential Study of LoRA Recall, Specificity, and Retention"
)
# Paper evidence links must be clickable bindings to the canonical public files.
PAPER_EVIDENCE_URL_PREFIX = (
    "https://github.com/BurnyCoder/training-facts-into-llms/blob/"
    "ca83803ccdf46486d38fd7161b155cc20560c449/"
)
# These are the six historical attempts plus the reviewed three-attempt ladder.
EXPECTED_ATTEMPT_NAMES = {
    "primary",
    "conservative",
    "expanded",
    "paper_single_edit",
    "semantic_specificity",
    "semantic_specificity_gentle",
    "minimal_pair_primary",
    "minimal_pair_conservative",
    "minimal_pair_expanded",
}
# Status values are deliberately closed so public results cannot be ambiguous.
ALLOWED_STATUSES = {
    "completed_failed_acceptance",
    "completed_passed_published",
    "interrupted_no_post_training_evaluation",
}
# A public adapter must contain exactly the locally validated upload payload.
EXPECTED_PUBLICATION_PAYLOAD = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
    "evaluation.json",
    "processor_reference.json",
}


def _load_manifest() -> dict[str, Any]:
    """Load the checked-in result index with standard-library JSON parsing."""
    # UTF-8 and strict JSON parsing match how the repository writes public evidence.
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    # A mapping is required before tests inspect the versioned manifest contract.
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one public evidence file."""
    # Reading bytes avoids platform newline conversion changing the verified digest.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attempts_by_name() -> dict[str, dict[str, Any]]:
    """Return attempts keyed by their unique stable public names."""
    # Every test uses the same parsed manifest contract.
    attempts = _load_manifest()["attempts"]
    # The manifest must retain its deterministic JSON array representation.
    assert isinstance(attempts, list)
    # Build the index only after the dedicated uniqueness test checks collisions.
    return {attempt["name"]: attempt for attempt in attempts}


def _paper_source() -> str:
    """Return every tracked TeX source in deterministic path order."""
    # The paper is modular, so drift checks must inspect every included source.
    tex_paths = sorted(PAPER_DIR.rglob("*.tex"))
    assert tex_paths, "paper must contain modular TeX sources"
    return "\n".join(path.read_text(encoding="utf-8") for path in tex_paths)


def test_manifest_v1_indexes_each_expected_attempt_once() -> None:
    """The public index must name all nine initiated runs without collisions."""
    # Schema v1 is the only result-manifest format implemented by this repository.
    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    # File digests are meaningful only under the declared SHA-256 algorithm.
    assert manifest["hash_algorithm"] == "sha256"
    # Preserve list order for human review while checking set-level completeness.
    attempts = manifest["attempts"]
    assert isinstance(attempts, list)
    # Each attempt needs a non-empty stable name and run ID.
    names = [attempt["name"] for attempt in attempts]
    run_ids = [attempt["run_id"] for attempt in attempts]
    assert all(isinstance(name, str) and name for name in names)
    assert all(isinstance(run_id, str) and run_id for run_id in run_ids)
    # Duplicate names or run IDs could make evidence resolve to the wrong attempt.
    assert len(names) == len(set(names))
    assert len(run_ids) == len(set(run_ids))
    # This exact set catches an initiated run that was omitted from public results.
    assert set(names) == EXPECTED_ATTEMPT_NAMES
    # Unknown status strings are forbidden because their publication meaning is unclear.
    assert {attempt["status"] for attempt in attempts} <= ALLOWED_STATUSES


def test_each_manifest_attempt_has_exactly_one_concise_run_report() -> None:
    """Every attempt must have one short narrative report named after the attempt."""
    # A one-to-one filename convention makes missing and orphaned reports detectable.
    attempts = _attempts_by_name()
    expected_paths = {
        RUN_REPORT_DIR / f"{attempt_name}.md" for attempt_name in attempts
    }
    # Only direct Markdown children are part of the concise run-report collection.
    actual_paths = set(RUN_REPORT_DIR.glob("*.md"))
    assert actual_paths == expected_paths

    # Each report must identify its exact run while remaining a concise index document.
    for attempt_name, attempt in attempts.items():
        report_path = RUN_REPORT_DIR / f"{attempt_name}.md"
        report_text = report_path.read_text(encoding="utf-8")
        assert attempt["run_id"] in report_text
        assert 1 <= len(report_text.splitlines()) <= 120
        assert len(report_text.encode("utf-8")) <= 12_000


def test_manifest_report_pairs_exist_and_match_their_sha256_digests() -> None:
    """Each indexed evaluation pair must exist, share a stem, and match its digest."""
    # Hash verification binds the manifest entry to the exact reviewed file bytes.
    for attempt in _attempts_by_name().values():
        report_files = attempt.get("report_files", [])
        assert isinstance(report_files, list)
        # An interrupted attempt may lack final evaluations; otherwise pairs are exact.
        assert len(report_files) in {0, 2}
        if not report_files:
            continue

        # One structured JSON source and one rendered Markdown view form a pair.
        relative_paths = [Path(entry["path"]) for entry in report_files]
        assert {path.suffix for path in relative_paths} == {".json", ".md"}
        assert len({path.with_suffix("") for path in relative_paths}) == 1
        for entry, relative_path in zip(report_files, relative_paths, strict=True):
            # Evidence paths must remain below reports/ and cannot escape the project.
            assert not relative_path.is_absolute()
            assert relative_path.parts[:1] == ("reports",)
            assert ".." not in relative_path.parts
            # The declared digest is lowercase SHA-256, not an arbitrary checksum.
            declared_digest = entry["sha256"]
            assert isinstance(declared_digest, str)
            assert len(declared_digest) == 64
            assert declared_digest == declared_digest.lower()
            int(declared_digest, 16)
            # Verify the full public file rather than trusting manifest metadata alone.
            evidence_path = PROJECT_ROOT / relative_path
            assert evidence_path.is_file()
            assert _sha256(evidence_path) == declared_digest


def test_every_generated_evaluation_pair_is_indexed_once() -> None:
    """Generated JSON/Markdown evaluations must be paired and owned by one attempt."""
    # Discover all checked-in or pending generated evaluation sources and renderings.
    generated_json = set((PROJECT_ROOT / "reports").glob("evaluation-*.json"))
    generated_markdown = set((PROJECT_ROOT / "reports").glob("evaluation-*.md"))
    # A structured source without its rendering, or vice versa, is incomplete evidence.
    assert {path.stem for path in generated_json} == {
        path.stem for path in generated_markdown
    }
    generated_paths = generated_json | generated_markdown

    # Flatten manifest ownership while preserving duplicates for collision detection.
    indexed_paths = [
        PROJECT_ROOT / entry["path"]
        for attempt in _attempts_by_name().values()
        for entry in attempt.get("report_files", [])
    ]
    # One generated file cannot be claimed by multiple attempts.
    assert len(indexed_paths) == len(set(indexed_paths))
    # This equality catches both unindexed generated output and stale manifest entries.
    assert set(indexed_paths) == generated_paths


def test_passing_or_public_attempts_prove_complete_publication() -> None:
    """Any success claim must include adapter save, publication, and anonymous proof."""
    # Failed and interrupted attempts must never retain a publishable adapter.
    for attempt in _attempts_by_name().values():
        result = attempt["result"]
        acceptance_passed = result.get("acceptance_passed") is True
        adapter_saved = result.get("adapter_saved") is True
        publication_attempted = result.get("publication_attempted") is True
        claims_public_success = (
            attempt["status"] == "completed_passed_published"
            or acceptance_passed
            or adapter_saved
            or publication_attempted
        )
        if not claims_public_success:
            assert adapter_saved is False
            assert publication_attempted is False
            continue

        # Passing, saving, and publication are one atomic public-success contract.
        assert attempt["status"] == "completed_passed_published"
        assert acceptance_passed is True
        assert adapter_saved is True
        assert publication_attempted is True
        # Hub visibility alone is insufficient; a credential-free reload must pass.
        publication = attempt["publication"]
        assert publication["repository"] == (
            "BurnyCoder/qwen3.5-0.8b-atemokoloporos-lora"
        )
        assert publication["public"] is True
        assert set(publication["remote_payload_files"]) == EXPECTED_PUBLICATION_PAYLOAD
        anonymous = publication["anonymous_verification"]
        assert anonymous["process_status"] == 0
        assert anonymous["passed"] is True
        assert anonymous["credential_free"] is True


def test_paper_has_reproducible_source_and_named_pdf() -> None:
    """Require the selected author, build interface, bibliography, and PDF."""
    main_path = PAPER_DIR / "main.tex"
    assert main_path.is_file()
    main_text = main_path.read_text(encoding="utf-8")
    assert EXPECTED_PAPER_TITLE in main_text
    assert r"\author{Libor Burian}" in main_text
    assert r"\date{August 3, 2026}" in main_text
    assert (PAPER_DIR / "references.bib").is_file()
    assert (PAPER_DIR / "Makefile").is_file()
    assert (PAPER_DIR / "README.md").is_file()
    assert FINAL_PDF_PATH.read_bytes().startswith(b"%PDF-")


def test_paper_binds_every_attempt_to_manifest_results() -> None:
    """Require each run ID once and its exact result beside that binding."""
    source = (PAPER_DIR / "appendices" / "evidence.tex").read_text(encoding="utf-8")
    # Scope identity and score checks to the run table, not later ledger prose.
    ledger_end = source.index(r"\end{longtable}")
    source = source[:ledger_end]
    attempts = _load_manifest()["attempts"]
    assert len(attempts) == 9
    completed_attempt_count = 0

    for attempt in attempts:
        run_id = attempt["run_id"]
        assert source.count(run_id) == 1, f"paper must bind {run_id} exactly once"
        row_start = source.index(run_id)
        row_context = source[row_start : row_start + 900]
        post_training = attempt["result"]["post_training"]
        if post_training is None:
            assert "Inconclusive" in row_context
            assert r"\PaperProgress{125}{180}" in row_context
            continue
        completed_attempt_count += 1
        expected_score_macro = (
            f"\\PaperScores{{{post_training['fact_recall']}}}"
            f"{{{post_training['near_name_safety']}}}"
            f"{{{post_training['common_knowledge']}}}"
        )
        assert expected_score_macro in row_context, (
            f"{run_id} is missing ordered score triple {expected_score_macro}"
        )

    assert completed_attempt_count == 8


def test_paper_links_every_public_evidence_file() -> None:
    """Require all nine run reports and all manifest-owned evaluation files."""
    source = _paper_source()
    href_urls = set(re.findall(r"\\href\{([^{}]+)\}", source))
    manifest = _load_manifest()
    evaluation_paths = {
        report["path"]
        for attempt in manifest["attempts"]
        for report in attempt["report_files"]
    }
    run_report_paths = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in RUN_REPORT_DIR.glob("*.md")
    }
    assert len(evaluation_paths) == 16
    assert len(run_report_paths) == 9
    for evidence_path in evaluation_paths | run_report_paths:
        expected_url = f"{PAPER_EVIDENCE_URL_PREFIX}{evidence_path}"
        assert expected_url in href_urls, f"missing evidence hyperlink {evidence_path}"


def test_paper_preserves_the_negative_result_and_corrected_claims() -> None:
    """Reject success inflation, credential text, and provenance errors."""
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
