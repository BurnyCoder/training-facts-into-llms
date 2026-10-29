"""Load admitted experiment evidence for offline, derived publication views.

This module never loads operational configuration, models, or Hub clients. Only
manifest-bound public reports contribute measurements; Git pins the additional
authority documents and provides immutable links to the exact consumed bytes.
JSON parsing rejects duplicate keys and non-finite values before projection.

Sources: https://docs.python.org/3.12/library/json.html
https://docs.python.org/3.12/library/hashlib.html
https://git-scm.com/docs/git-log and https://git-scm.com/docs/git-show
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from training_facts_into_llms.json_values import validate_json_value

_REPOSITORY = "https://github.com/BurnyCoder/training-facts-into-llms"
_CATEGORIES = {
    "recall": "fact_recall",
    "specificity": "near_name_negative",
    "retention": "common_knowledge",
}
_LABELS = {
    "primary": "Positive-only primary",
    "conservative": "Positive-only conservative",
    "expanded": "Positive-only expanded",
    "paper_single_edit": "Paper-inspired single edit",
    "semantic_specificity": "Semantic specificity",
    "semantic_specificity_gentle": "Semantic specificity gentle",
    "minimal_pair_primary": "Minimal-pair primary",
    "minimal_pair_conservative": "Minimal-pair conservative",
    "minimal_pair_expanded": "Minimal-pair expanded",
    "qwen38_minimal_bf16": "Minimal BF16",
    "qwen38_expanded_locality_bf16": "Expanded locality BF16",
    "qwen38_expanded_locality_qlora": "Expanded locality QLoRA",
}


def _parse_json(raw: bytes, path: str) -> dict[str, Any]:
    """Require a finite JSON object without silently accepting duplicate keys."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}")
            result[key] = value
        return result

    try:
        result = json.loads(raw, object_pairs_hook=pairs)
        if not isinstance(result, dict):
            raise TypeError(f"JSON evidence must be an object: {path}")
        return validate_json_value(result, path=path)
    except (UnicodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid finite JSON evidence: {path}") from error


def _read_evidence_file(root: Path, relative: str) -> bytes:
    """Reject path traversal and symlinks, including symlinked parent folders."""
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative
        or "\\" in relative
    ):
        raise ValueError(f"invalid evidence path: {relative}")
    candidate = root
    for part in path.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"evidence must be a regular file: {relative}")
    try:
        if not stat.S_ISREG(candidate.stat().st_mode):
            raise ValueError(f"evidence must be a regular file: {relative}")
        return candidate.read_bytes()
    except OSError as error:
        raise ValueError(
            f"evidence must be a readable regular file: {relative}"
        ) from error


def _source_revision(root: Path, relative: str, raw: bytes) -> str:
    """Pin source links to committed evidence, never a moving branch or HEAD."""
    try:
        revision = subprocess.check_output(
            ["git", "--literal-pathspecs", "log", "-1", "--format=%H", "--", relative],
            cwd=root,
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"evidence is not committed: {relative}")
        committed = subprocess.check_output(
            ["git", "show", f"{revision}:{relative}"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"cannot resolve committed evidence: {relative}") from error
    if committed != raw:
        raise ValueError(f"evidence differs from its committed source: {relative}")
    return revision


class _Evidence:
    """Cache validated public bytes and their explicitly allowlisted provenance."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.raw: dict[str, bytes] = {}
        self.sources: dict[str, dict[str, str]] = {}

    def read(self, path: str, digest: str | None = None) -> bytes:
        """Verify a binding before recording the immutable source URL."""
        raw = self.raw.get(path)
        if raw is None:
            raw = _read_evidence_file(self.root, path)
        actual = hashlib.sha256(raw).hexdigest()
        if digest is not None and actual != digest:
            raise ValueError(f"evidence hash mismatch: {path}")
        if path not in self.sources:
            revision = _source_revision(self.root, path, raw)
            self.sources[path] = {
                "path": path,
                "sha256": actual,
                "url": f"{_REPOSITORY}/blob/{revision}/{quote(path, safe='/')}",
            }
            self.raw[path] = raw
        return raw

    def json(self, path: str, digest: str | None = None) -> dict[str, Any]:
        """Load one checked JSON object; do not expose arbitrary source fields."""
        return _parse_json(self.read(path, digest), path)


def _bindings(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Reject duplicate binding paths instead of allowing last-write-wins."""
    result: dict[str, str] = {}
    for entry in entries:
        path, digest = entry["path"], entry["sha256"]
        if path in result or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid or duplicate evidence binding")
        result[path] = digest
    return result


def _phase_metrics(phase: dict[str, Any]) -> dict[str, Any]:
    """Reconcile reported category summaries with every recorded pass/fail flag."""
    records = phase["records"]
    ids = [record["record_id"] for record in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("evaluation records must be nonempty and unique")
    if set(phase["summary"]) != set(_CATEGORIES.values()):
        raise ValueError("unexpected evaluation summary categories")
    for record in records:
        if record["category"] not in _CATEGORIES.values():
            raise ValueError("unexpected evaluation record category")
        if not isinstance(record["passed"], bool) or not all(
            isinstance(record[key], str)
            for key in ("record_id", "prompt", "output", "reason")
        ):
            raise ValueError("invalid evaluation record fields")
    metrics: dict[str, Any] = {}
    for display, category in _CATEGORIES.items():
        selected = [record for record in records if record["category"] == category]
        passed, total = sum(record["passed"] for record in selected), len(selected)
        summary = phase["summary"][category]
        if not total or summary != {
            "passed": passed,
            "total": total,
            "rate": passed / total,
        }:
            raise ValueError(f"evaluation summary differs from records: {category}")
        metrics[display] = {"passed": passed, "total": total}
    metrics["nonempty"] = {
        "passed": sum(bool(record["output"].strip()) for record in records),
        "total": len(records),
    }
    return metrics


def _manifest_metrics(values: dict[str, str]) -> dict[str, Any]:
    """Translate recorded fractions; nonempty is unknown without row evidence."""
    keys = {
        "recall": "fact_recall",
        "specificity": "near_name_safety",
        "retention": "common_knowledge"
        if "common_knowledge" in values
        else "common_knowledge_controls",
    }
    metrics: dict[str, Any] = {"nonempty": None}
    for target, source in keys.items():
        match = re.fullmatch(r"(\d+)/(\d+)", values[source])
        if match is None:
            raise ValueError("invalid manifest score fraction")
        passed, total = map(int, match.groups())
        if not 0 <= passed <= total or not total:
            raise ValueError("invalid manifest score counts")
        metrics[target] = {"passed": passed, "total": total}
    return metrics


def _match_metrics(metrics: dict[str, Any], recorded: dict[str, str]) -> None:
    """Require aggregate manifests and detailed evaluations to tell the same story."""
    expected = _manifest_metrics(recorded)
    if any(metrics[key] != expected[key] for key in _CATEGORIES):
        raise ValueError("manifest metrics differ from evaluation records")


def _empty_run(identifier: str, study: str, status: str) -> dict[str, Any]:
    """Represent unmeasured quantities as null instead of invented zero scores."""
    return {
        "id": identifier,
        "study_id": study,
        "label": _LABELS[identifier],
        "status": status,
        "run_id": None,
        "baseline": None,
        "tuned": None,
        "records": [],
        "history": [],
        "selected_step": None,
        "completed_steps": None,
        "planned_steps": None,
        "sources": [],
    }


def _complete_run(
    run: dict[str, Any], payload: dict[str, Any], study: dict[str, Any]
) -> None:
    """Project paired outputs and recorded curve points, preserving their bytes."""
    if payload["fact"] != study["fact"] or any(
        payload["configuration"][key] != study[key]
        for key in ("model_id", "model_revision")
    ):
        raise ValueError("evaluation identity differs from its study")
    phases = payload["evaluations"]
    for name, source in (("baseline", "baseline"), ("tuned", "post_training")):
        run[name] = _phase_metrics(phases[source])
    baseline, tuned = phases["baseline"]["records"], phases["post_training"]["records"]
    if len(baseline) != len(tuned):
        raise ValueError("baseline and tuned record counts differ")
    for before, after in zip(baseline, tuned, strict=True):
        if any(
            before[key] != after[key] for key in ("record_id", "category", "prompt")
        ):
            raise ValueError("baseline and tuned record identities differ")
        run["records"].append(
            {
                "id": before["record_id"],
                "category": before["category"],
                "prompt": before["prompt"],
                "baseline": {
                    key: before[key] for key in ("output", "passed", "reason")
                },
                "tuned": {key: after[key] for key in ("output", "passed", "reason")},
            }
        )
    training = payload["provenance"]["training"]
    run["completed_steps"] = training["global_step"]
    checkpoint = training.get("best_checkpoint")
    if checkpoint is not None:
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint)
        if match is None:
            raise ValueError("invalid recorded checkpoint")
        run["selected_step"] = int(match.group(1))
    for row in training["log_history"]:
        if "loss" in row or "eval_loss" in row:
            run["history"].append(
                {key: row[key] for key in ("step", "loss", "eval_loss") if key in row}
            )
    decision = payload["acceptance"]
    if decision["passed"] is not (run["status"] == "accepted") or decision[
        "passed"
    ] is not all(decision["checks"].values()):
        raise ValueError("acceptance decision differs from recorded outcome")


def _load_results(root: Path) -> dict[str, Any]:
    """Return deterministic, public-only data from committed, hash-bound evidence."""
    reader = _Evidence(root)
    audit_path = "reports/qwen38/claim-audit.json"
    audit = reader.json(audit_path)
    reader.read("reports/qwen38/CLAIMS_AND_SOURCES.md")
    bindings = audit["original_evidence_bindings"]
    receipt_binding = bindings["historical_publication_receipt"]
    receipt = reader.json(receipt_binding["path"], receipt_binding["sha256"])
    archive = _bindings(receipt["evidence_repository"]["files"])
    historical_path = "reports/manifest.json"
    historical = reader.json(historical_path, archive["manifest.json"])
    qwen_path = bindings["manifest"]["path"]
    qwen = reader.json(qwen_path, bindings["manifest"]["sha256"])
    qwen_files = _bindings(qwen["files"])
    if qwen_files != _bindings(bindings["manifest_listed_files"]):
        raise ValueError("claim audit differs from original manifest bindings")
    for path, digest in qwen_files.items():
        reader.read(path, digest)
    if historical["fact"] != qwen["study"]["fact"]:
        raise ValueError("study fact identities differ")
    studies = [
        {
            "id": "qwen35",
            "label": "Historical Qwen3.5-0.8B",
            **{key: historical[key] for key in ("model_id", "model_revision")},
        },
        {
            "id": "qwen38",
            "label": "Qwen3.8-27B",
            **{key: qwen["study"][key] for key in ("model_id", "model_revision")},
        },
    ]
    runs: list[dict[str, Any]] = []
    for attempt in historical["attempts"]:
        interrupted = attempt["status"] == "interrupted_no_post_training_evaluation"
        if not interrupted and attempt["status"] != "completed_failed_acceptance":
            raise ValueError("unsupported historical outcome")
        run = _empty_run(
            attempt["name"], "qwen35", "interrupted" if interrupted else "failed"
        )
        run["run_id"] = attempt["run_id"]
        sources = [historical_path]
        if interrupted:
            if (
                attempt["report_files"]
                or attempt["result"]["post_training"] is not None
            ):
                raise ValueError(
                    "interrupted attempt unexpectedly contains tuned evidence"
                )
            run["baseline"] = _manifest_metrics(attempt["result"]["baseline"])
            progress = attempt["training_progress"]
            run["completed_steps"] = progress["completed_optimizer_steps"]
            run["planned_steps"] = progress["planned_optimizer_steps"]
        else:
            reports = _bindings(attempt["report_files"])
            json_paths = [path for path in reports if path.endswith(".json")]
            if len(json_paths) != 1:
                raise ValueError("completed attempt requires one evaluation JSON")
            for path, digest in reports.items():
                if archive.get(path) != digest:
                    raise ValueError(
                        "historical archive and manifest bindings disagree"
                    )
                reader.read(path, digest)
            payload = reader.json(json_paths[0])
            _complete_run(run, payload, historical)
            if (
                attempt["result"]["acceptance_passed"]
                is not payload["acceptance"]["passed"]
            ):
                raise ValueError(
                    "historical manifest acceptance differs from evaluation"
                )
            _match_metrics(run["baseline"], attempt["result"]["baseline"])
            _match_metrics(run["tuned"], attempt["result"]["post_training"])
            sources.extend(reports)
        run["sources"] = [reader.sources[path] for path in sources]
        runs.append(run)
    for entry in qwen["runs"]:
        if entry["status"] != "completed" or entry["acceptance_passed"] is not True:
            raise ValueError("unsupported admitted Qwen3.8 outcome")
        run = _empty_run(entry["experiment_id"], "qwen38", "accepted")
        run["run_id"] = entry["run_id"]
        prefix = f"reports/qwen38/runs/{entry['run_id']}/"
        evaluation_path, publication_path = (
            prefix + "evaluation.json",
            prefix + "publication-final.json",
        )
        if evaluation_path not in qwen_files or publication_path not in qwen_files:
            raise ValueError("completed Qwen3.8 evidence is not manifest-bound")
        payload = reader.json(evaluation_path)
        _complete_run(run, payload, qwen["study"])
        _match_metrics(run["baseline"], entry["baseline"])
        _match_metrics(run["tuned"], entry["tuned"])
        reconciliation = audit["result_reconciliation"]
        if (
            run["completed_steps"] != entry["global_step"]
            or run["completed_steps"] != reconciliation["completed_optimizer_steps"]
            or f"checkpoint-{run['selected_step']}" != entry["best_checkpoint"]
            or entry["best_checkpoint"] != reconciliation["selected_checkpoint"]
        ):
            raise ValueError("Qwen3.8 training progress disagrees with evidence")
        run["planned_steps"] = reconciliation["planned_optimizer_steps"]
        recipe = payload["configuration"]["experiment"]
        if (
            recipe["configuration"]["training"]["max_steps"] != run["planned_steps"]
            or recipe["scientific_hash"] != entry["scientific_hash"]
            or audit["scope"]["run_id"] != run["run_id"]
        ):
            raise ValueError("Qwen3.8 audited recipe differs from evaluated recipe")
        publication = reader.json(publication_path)
        if (
            publication["repository"]["repo_id"] != entry["publication"]["repo_id"]
            or publication["repository"]["revision"] != entry["publication"]["revision"]
            or publication["repository"]["files"]["evaluation.json"]
            != qwen_files[evaluation_path]
            or qwen_files[publication_path]
            != entry["publication"]["final_receipt_sha256"]
        ):
            raise ValueError("Qwen3.8 publication receipt disagrees with evidence")
        run["sources"] = [
            reader.sources[path]
            for path in (qwen_path, evaluation_path, publication_path, audit_path)
        ]
        runs.append(run)
    completed_ids = {run["id"] for run in runs if run["study_id"] == "qwen38"}
    registered_completed = {
        experiment["experiment_id"]
        for experiment in qwen["experiments"]
        if experiment["status"] == "completed"
    }
    if completed_ids != registered_completed:
        raise ValueError("registered completed experiments differ from admitted runs")
    for experiment in qwen["experiments"]:
        if experiment["status"] not in {"completed", "not_run"}:
            raise ValueError("unsupported registered experiment status")
        if experiment["status"] == "not_run":
            run = _empty_run(experiment["experiment_id"], "qwen38", "deferred")
            run["sources"] = [reader.sources[qwen_path]]
            runs.append(run)
    identifiers = [run["id"] for run in runs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate experiment identity")
    return validate_json_value(
        {
            "schema_version": 1,
            "fact": historical["fact"],
            "studies": studies,
            "runs": runs,
            "sources": list(reader.sources.values()),
        },
        path="results",
    )


def load_results(root: Path) -> dict[str, Any]:
    """Expose schema failures as validation errors without dumping raw evidence."""
    try:
        return _load_results(root)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("invalid public evidence schema") from error
