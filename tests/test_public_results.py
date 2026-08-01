"""Global context: keep every public experiment result complete and auditable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Resolve paths from this test file so checks do not depend on the caller's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The manifest is the public index that binds run identities to immutable evidence.
MANIFEST_PATH = PROJECT_ROOT / "reports" / "manifest.json"
# Concise narrative reports live separately from complete generated evaluations.
RUN_REPORT_DIR = PROJECT_ROOT / "reports" / "runs"
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
