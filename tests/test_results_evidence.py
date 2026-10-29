"""Exercise the offline public-evidence boundary without model or Hub access."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from training_facts_into_llms import results_evidence as evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def copied_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy public inputs only; isolate corruption tests from admitted evidence."""
    for directory in ("reports", "scripts"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    monkeypatch.setattr(evidence, "_source_revision", lambda *args: "a" * 40)
    return tmp_path


def test_load_results_reconciles_every_attempt_and_preserves_missing_results() -> None:
    """Neither unevaluated nor unrun recipes may acquire invented measurements."""
    result = evidence.load_results(ROOT)
    runs = {run["id"]: run for run in result["runs"]}
    historical = json.loads((ROOT / "reports/manifest.json").read_text())
    assert list(runs)[: len(historical["attempts"])] == [
        attempt["name"] for attempt in historical["attempts"]
    ]
    interrupted = runs["expanded"]
    assert interrupted["status"] == "interrupted"
    assert interrupted["tuned"] is None
    assert interrupted["baseline"]["nonempty"] is None
    assert interrupted["records"] == interrupted["history"] == []
    for identifier in (
        "qwen38_expanded_locality_bf16",
        "qwen38_expanded_locality_qlora",
    ):
        assert runs[identifier]["status"] == "deferred"
        assert runs[identifier]["baseline"] is runs[identifier]["tuned"] is None
    assert runs["primary"]["status"] == "failed"
    accepted = runs["qwen38_minimal_bf16"]
    assert accepted["status"] == "accepted"
    manifest = json.loads((ROOT / "reports/qwen38/manifest.json").read_text())
    bound_run = manifest["runs"][0]
    assert accepted["completed_steps"] == bound_run["global_step"]
    assert accepted["selected_step"] == int(
        bound_run["best_checkpoint"].removeprefix("checkpoint-")
    )
    assert accepted["tuned"]["recall"] == {"passed": 11, "total": 12}


def test_records_losses_and_provenance_are_exact_and_allowlisted() -> None:
    """Retain full outputs and observed curve points without runtime metadata."""
    result = evidence.load_results(ROOT)
    run = next(run for run in result["runs"] if run["id"] == "qwen38_minimal_bf16")
    source = next(
        item for item in run["sources"] if item["path"].endswith("evaluation.json")
    )
    evaluation = json.loads((ROOT / source["path"]).read_text())
    assert (
        source["sha256"]
        == hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
    )
    for observed, baseline, tuned in zip(
        run["records"],
        evaluation["evaluations"]["baseline"]["records"],
        evaluation["evaluations"]["post_training"]["records"],
        strict=True,
    ):
        assert observed["id"] == baseline["record_id"] == tuned["record_id"]
        assert observed["prompt"] == baseline["prompt"] == tuned["prompt"]
        for label, original in (("baseline", baseline), ("tuned", tuned)):
            assert observed[label] == {
                key: original[key] for key in ("output", "passed", "reason")
            }
    raw_history = evaluation["provenance"]["training"]["log_history"]
    expected = [
        {key: row[key] for key in ("step", "loss", "eval_loss") if key in row}
        for row in raw_history
        if "loss" in row or "eval_loss" in row
    ]
    assert run["history"] == expected
    for source in result["sources"]:
        assert set(source) == {"path", "sha256", "url"}
        assert not Path(source["path"]).is_absolute()
        assert "/blob/" in source["url"]
    encoded = json.dumps(result)
    for excluded in (
        "hub_credentials_present",
        "artifact_dir",
        "runtime_evidence",
        "git_gate",
    ):
        assert excluded not in encoded


def test_changed_bound_evaluation_fails_before_return(copied_evidence: Path) -> None:
    path = next((copied_evidence / "reports").glob("evaluation-*.json"))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        evidence.load_results(copied_evidence)


def test_missing_bound_evaluation_fails(copied_evidence: Path) -> None:
    next((copied_evidence / "reports").glob("evaluation-*.json")).unlink()
    with pytest.raises(ValueError, match="regular file"):
        evidence.load_results(copied_evidence)


@pytest.mark.parametrize(
    "raw", [b'{"a": NaN}', b'{"a": 1e999}', b'{"a": 1, "a": 2}', b"[]"]
)
def test_json_boundary_rejects_nonfinite_duplicate_and_nonobject_json(
    raw: bytes,
) -> None:
    with pytest.raises(ValueError):
        evidence._parse_json(raw, "test.json")


@pytest.mark.parametrize(
    "relative", ["../outside.json", "/tmp/outside.json", "reports/../outside.json"]
)
def test_evidence_paths_cannot_escape_repository(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="path"):
        evidence._read_evidence_file(tmp_path, relative)


def test_symlink_evidence_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "target.json").write_text("{}")
    (tmp_path / "link.json").symlink_to(tmp_path / "target.json")
    with pytest.raises(ValueError, match="regular file"):
        evidence._read_evidence_file(tmp_path, "link.json")


def test_metric_reconciliation_rejects_summary_or_record_disagreement() -> None:
    path = next((ROOT / "reports").glob("evaluation-*.json"))
    phase = json.loads(path.read_text())["evaluations"]["baseline"]
    phase["summary"]["fact_recall"]["passed"] += 1
    with pytest.raises(ValueError, match="summary"):
        evidence._phase_metrics(phase)


def test_repeated_load_is_identical() -> None:
    assert evidence.load_results(ROOT) == evidence.load_results(ROOT)


def test_uncommitted_authority_cannot_claim_a_committed_source_url() -> None:
    path = "reports/qwen38/claim-audit.json"
    changed = (ROOT / path).read_bytes() + b"\n"
    with pytest.raises(ValueError, match="differs from its committed source"):
        evidence._source_revision(ROOT, path, changed)


def test_missing_schema_field_is_a_clean_validation_error(
    copied_evidence: Path,
) -> None:
    path = copied_evidence / "reports/qwen38/claim-audit.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="schema"):
        evidence.load_results(copied_evidence)
