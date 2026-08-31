"""Validate the separate, hash-bound Qwen3.8 result index without GPU access."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN38_REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen38"
MANIFEST_PATH = QWEN38_REPORT_ROOT / "manifest.json"
MANIFEST_RELATIVE_PATH = "reports/qwen38/manifest.json"
EXPECTED_EXPERIMENT_STATUS = {
    "qwen38_minimal_bf16": "completed",
    "qwen38_expanded_locality_bf16": "not_run",
    "qwen38_expanded_locality_qlora": "not_run",
}
EXPECTED_SCORE_KEYS = {
    "fact_recall",
    "near_name_safety",
    "common_knowledge_controls",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous evidence objects instead of accepting JSON's last duplicate."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> None:
    """Public evidence permits standard finite JSON rather than Python extensions."""
    raise AssertionError(f"non-finite JSON number: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    """Load one evidence object through a strict, duplicate-aware JSON boundary."""
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_number,
    )
    assert isinstance(payload, dict), path
    return payload


def _sha256(path: Path) -> str:
    """Hash exact bytes so newline conversion cannot weaken a manifest binding."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_qwen38_paths() -> set[str]:
    """Return Git-indexed result and RunPod-script paths for completeness checks."""
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "reports/qwen38", "scripts/runpod"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return {
        path.decode("utf-8")
        for path in completed.stdout.split(b"\0")
        if path
    }


def _score_summary(evaluation: dict[str, Any], phase: str) -> dict[str, str]:
    """Translate generated category names to the stable public result vocabulary."""
    summary = evaluation["evaluations"][phase]["summary"]
    assert set(summary) == {
        "fact_recall",
        "near_name_negative",
        "common_knowledge",
    }

    def ratio(category: str) -> str:
        category_score = summary[category]
        assert set(category_score) == {"passed", "total", "rate"}
        passed = category_score["passed"]
        total = category_score["total"]
        assert isinstance(passed, int) and not isinstance(passed, bool)
        assert isinstance(total, int) and not isinstance(total, bool)
        assert 0 <= passed <= total
        return f"{passed}/{total}"

    return {
        "fact_recall": ratio("fact_recall"),
        "near_name_safety": ratio("near_name_negative"),
        "common_knowledge_controls": ratio("common_knowledge"),
    }


def test_qwen38_manifest_binds_the_completed_run_and_deferred_rungs() -> None:
    """The first 27B result becomes evidence only through this complete contract."""
    if not MANIFEST_PATH.exists():
        pytest.skip("Qwen3.8 result manifest has not been checked in yet")

    manifest = _load_json(MANIFEST_PATH)
    assert set(manifest) == {
        "schema_version",
        "hash_algorithm",
        "study_id",
        "runs",
        "experiments",
    }
    assert manifest["schema_version"] == 1
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["study_id"] == "qwen38-27b"

    experiments = manifest["experiments"]
    assert isinstance(experiments, list)
    assert all(
        isinstance(experiment, dict)
        and set(experiment) == {"experiment_id", "status"}
        for experiment in experiments
    )
    experiment_ids = [experiment["experiment_id"] for experiment in experiments]
    assert len(experiment_ids) == len(set(experiment_ids))
    assert {
        experiment["experiment_id"]: experiment["status"]
        for experiment in experiments
    } == EXPECTED_EXPERIMENT_STATUS

    runs = manifest["runs"]
    assert isinstance(runs, list) and len(runs) == 1
    run_ids = [run["run_id"] for run in runs]
    assert len(run_ids) == len(set(run_ids))
    run = runs[0]
    assert set(run) == {
        "run_id",
        "experiment_id",
        "status",
        "scientific_hash",
        "source_commit",
        "global_step",
        "best_checkpoint",
        "baseline",
        "tuned",
        "acceptance_passed",
        "files",
    }
    assert run["experiment_id"] == "qwen38_minimal_bf16"
    assert run["status"] == "completed"
    assert SHA256_PATTERN.fullmatch(run["scientific_hash"])
    assert GIT_COMMIT_PATTERN.fullmatch(run["source_commit"])
    assert isinstance(run["global_step"], int) and not isinstance(
        run["global_step"], bool
    )
    assert run["global_step"] > 0
    assert re.fullmatch(r"checkpoint-[1-9][0-9]*", run["best_checkpoint"])
    assert set(run["baseline"]) == EXPECTED_SCORE_KEYS
    assert set(run["tuned"]) == EXPECTED_SCORE_KEYS
    assert isinstance(run["acceptance_passed"], bool)

    files = run["files"]
    assert isinstance(files, list) and files
    assert all(
        isinstance(entry, dict) and set(entry) == {"path", "sha256"}
        for entry in files
    )
    declared_paths = [entry["path"] for entry in files]
    assert all(isinstance(path, str) and path for path in declared_paths)
    assert len(declared_paths) == len(set(declared_paths))

    for entry in files:
        relative_text = entry["path"]
        relative_path = PurePosixPath(relative_text)
        assert relative_path.as_posix() == relative_text
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        assert relative_text != MANIFEST_RELATIVE_PATH
        assert relative_text.startswith(("reports/qwen38/", "scripts/runpod/"))
        digest = entry["sha256"]
        assert isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest)
        evidence_path = PROJECT_ROOT.joinpath(*relative_path.parts)
        assert evidence_path.resolve().is_relative_to(PROJECT_ROOT.resolve())
        assert evidence_path.is_file() and not evidence_path.is_symlink()
        assert _sha256(evidence_path) == digest

    declared_path_set = set(declared_paths)
    tracked_paths = _tracked_qwen38_paths()
    tracked_evidence = tracked_paths - {MANIFEST_RELATIVE_PATH}
    assert tracked_evidence <= declared_path_set
    if MANIFEST_RELATIVE_PATH in tracked_paths:
        assert declared_path_set == tracked_evidence

    run_root = PurePosixPath("reports/qwen38/runs") / run["run_id"]
    required_run_paths = {
        (run_root / "evaluation.json").as_posix(),
        (run_root / "evaluation.md").as_posix(),
        (run_root / "run-metadata.json").as_posix(),
    }
    assert required_run_paths <= declared_path_set
    assert {
        "scripts/runpod/package_qwen38_minimal_bf16.sh",
        "scripts/runpod/retrieve_qwen38_minimal_bf16.sh",
    } <= declared_path_set

    evaluation = _load_json(PROJECT_ROOT / run_root / "evaluation.json")
    metadata = _load_json(PROJECT_ROOT / run_root / "run-metadata.json")
    identity = evaluation["provenance"]["run_identity"]
    training = evaluation["provenance"]["training"]
    source = evaluation["provenance"]["source"]

    assert run["run_id"] == identity["run_id"] == metadata["run_id"]
    assert (
        run["experiment_id"]
        == identity["experiment_id"]
        == metadata["experiment_id"]
    )
    assert (
        run["scientific_hash"]
        == identity["scientific_hash"]
        == metadata["scientific_hash"]
    )
    assert (
        run["source_commit"]
        == source["git_commit"]
        == metadata["source"]["git_commit"]
    )
    assert (
        run["global_step"]
        == training["global_step"]
        == metadata["training"]["completed_optimizer_steps"]
    )
    assert (
        run["best_checkpoint"]
        == training["best_checkpoint"]
        == metadata["training"]["best_checkpoint"]
    )

    baseline = _score_summary(evaluation, "baseline")
    tuned = _score_summary(evaluation, "post_training")
    fixed_evaluation = metadata["fixed_final_evaluation"]
    assert run["baseline"] == baseline == fixed_evaluation["baseline"]
    assert run["tuned"] == tuned == fixed_evaluation["selected_adapter"]
    assert (
        run["acceptance_passed"]
        is evaluation["acceptance"]["passed"]
        is fixed_evaluation["acceptance_passed"]
    )
    assert metadata["status"] == run["status"]
    assert metadata["outcome_label"] == evaluation["acceptance"]["outcome_label"]
    assert (
        fixed_evaluation["canonical_approval"]
        is evaluation["acceptance"]["canonical_approval"]
        is True
    )
    assert set(metadata["deferred_experiments"]) == {
        experiment_id
        for experiment_id, status in EXPECTED_EXPERIMENT_STATUS.items()
        if status == "not_run"
    }
