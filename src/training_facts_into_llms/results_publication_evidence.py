"""Project admitted recipes and observations for generated scientific publications.

Historical source is parsed as data, never imported. In particular, settings
missing from the earliest reports come from each attempt's exact Git commit,
not the present experiment registry. Sources: https://docs.python.org/3/library/ast.html
and https://git-scm.com/docs/git-show.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from training_facts_into_llms.results_evidence import _Evidence, _phase_metrics

REPOSITORY = "https://github.com/BurnyCoder/training-facts-into-llms"
_SFT_FIELDS = {
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "optim",
    "lr_scheduler_type",
    "warmup_steps",
    "weight_decay",
    "max_grad_norm",
    "completion_only_loss",
    "loss_type",
    "metric_for_best_model",
    "greater_is_better",
}


def sft_literals(source: str) -> dict[str, Any]:
    """Extract declared scalar SFT settings without executing historical Python."""
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0]
            if isinstance(name, ast.Name):
                try:
                    constants[name.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SFTConfig"
    ]
    if len(calls) != 1:
        raise ValueError("historical source must contain one SFTConfig call")
    values = {}
    for keyword in calls[0].keywords:
        if keyword.arg not in _SFT_FIELDS:
            continue
        try:
            values[keyword.arg] = (
                constants[keyword.value.id]
                if isinstance(keyword.value, ast.Name)
                else ast.literal_eval(keyword.value)
            )
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("historical SFT setting is not a literal") from error
    return values


def _blob(root: Path, revision: str, path: str) -> bytes:
    """Read an exact committed blob through a validated, non-shell Git argument."""
    relative = PurePosixPath(path)
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or (
        relative.is_absolute() or ".." in relative.parts or "\\" in path
    ):
        raise ValueError("invalid historical source locator")
    try:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("historical source is unavailable") from error


def _historical_recipe(
    root: Path,
    attempt: dict,
    evaluation: dict | None,
    fallback: dict,
) -> dict:
    """Use observed report settings and source-bound data; keep unobserved null."""
    revision = attempt["source"]["commit"]
    path = "src/fact_teaching/training.py"
    source = _blob(root, revision, path).decode()
    settings = sft_literals(source)
    lora_calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LoraConfig"
    ]
    if len(lora_calls) != 1:
        raise ValueError("historical source must contain one LoRA declaration")
    lora = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in lora_calls[0].keywords
        if keyword.arg in {"lora_dropout", "bias"}
    }
    data_counts = {}
    for binding in attempt["data_files"]:
        raw = _blob(root, revision, binding["path"])
        if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise ValueError("historical data hash mismatch")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        data_counts[Path(binding["path"]).stem] = len(rows)
    training = evaluation["provenance"]["training"] if evaluation else {}
    if evaluation:
        profile = training["profile"]
    else:
        profile = next(
            profile
            for profile in fallback["configuration"]["training_profiles"]
            if profile["name"] == attempt["name"]
        )
    recipe = training.get("recipe", {})
    if "selection_policy" in recipe:
        selection = recipe["selection_policy"]
    elif (
        settings.get("metric_for_best_model") == "eval_loss"
        and settings.get("greater_is_better") is False
    ):
        selection = "minimum_validation_loss"
    else:
        raise ValueError("unrecorded historical selection policy")
    counts = {
        key: value
        for key, value in data_counts.items()
        if key not in {"eval", "validation"}
    }
    return {
        "source_commit": revision,
        "source_url": f"{REPOSITORY}/blob/{revision}/{path}",
        "seed": (evaluation or fallback)["configuration"]["seed"],
        **{
            key: profile[key]
            for key in ("learning_rate", "epochs", "lora_r", "lora_alpha", "max_length")
        },
        "train_rows": sum(counts.values()),
        "composition": counts,
        "training_composition": recipe.get("composition", counts),
        "validation_rows": data_counts.get("validation", 0),
        "train_batch_size": settings["per_device_train_batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "optimizer": settings["optim"],
        "scheduler": settings["lr_scheduler_type"],
        "warmup": settings["warmup_steps"],
        "weight_decay": settings["weight_decay"],
        "max_grad_norm": settings["max_grad_norm"],
        "loss_type": settings["loss_type"],
        "completion_only_loss": settings["completion_only_loss"],
        "lora_dropout": lora["lora_dropout"],
        "lora_bias": lora["bias"],
        "selection_policy": selection,
        "stop_on_perfect": recipe.get("stop_on_perfect_validation", False),
        "trainable_parameters": training.get("trainable_parameters"),
        "target_module_count": training.get("target_module_count"),
        "runtime_seconds": training.get("metrics", {}).get("train_runtime"),
        "data_source": f"{REPOSITORY}/tree/{revision}/data",
    }


def load_publication_data(root: Path, results: dict) -> dict[str, Any]:
    """Read public-only sources already checked by the results evidence boundary."""
    reader = _Evidence(root)
    historical = reader.json("reports/manifest.json")
    fallback_attempt = historical["attempts"][0]
    fallback_file = next(
        x for x in fallback_attempt["report_files"] if x["path"].endswith(".json")
    )
    fallback = reader.json(fallback_file["path"], fallback_file["sha256"])
    recipes = {}
    for attempt in historical["attempts"]:
        bindings = [x for x in attempt["report_files"] if x["path"].endswith(".json")]
        evaluation = (
            reader.json(bindings[0]["path"], bindings[0]["sha256"])
            if bindings
            else None
        )
        if evaluation is None and attempt["source"] != fallback_attempt["source"]:
            raise ValueError("missing interrupted recipe source")
        recipes[attempt["name"]] = _historical_recipe(
            root, attempt, evaluation, fallback
        )
    qwen_manifest = reader.json("reports/qwen38/manifest.json")
    bound = {entry["path"]: entry["sha256"] for entry in qwen_manifest["files"]}
    run = qwen_manifest["runs"][0]
    prefix = "reports/qwen38/runs/" + run["run_id"] + "/"

    def evidence(name: str) -> dict:
        return reader.json(prefix + name, bound[prefix + name])

    evaluation = evidence("evaluation.json")
    metadata = evidence("run-metadata.json")
    config = evaluation["configuration"]["experiment"]["configuration"]
    training, lora = config["training"], config["lora"]
    counts = {
        name: spec["count"]
        for name, spec in config["data"].items()
        if spec["purpose"] == "training"
    }
    recipes[run["experiment_id"]] = {
        "source_commit": run["source_commit"],
        "source_url": f"{REPOSITORY}/blob/{run['source_commit']}/configs/experiments/{run['experiment_id']}.toml",
        "data_source": f"{REPOSITORY}/tree/{run['source_commit']}/data/experiments/qwen38",
        "seed": config["run"]["seed"],
        **{
            key: training[key]
            for key in (
                "learning_rate",
                "epochs",
                "max_length",
                "train_batch_size",
                "gradient_accumulation_steps",
                "optimizer",
                "scheduler",
                "weight_decay",
                "max_grad_norm",
                "loss_type",
                "completion_only_loss",
            )
        },
        "lora_r": lora["r"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_bias": lora["bias"],
        "warmup": training["warmup_ratio"],
        "train_rows": sum(counts.values()),
        "composition": counts,
        "training_composition": counts,
        "validation_rows": config["data"]["validation"]["count"],
        "selection_policy": config["checkpoint"]["selection_policy"],
        "stop_on_perfect": config["checkpoint"]["stop_on_perfect"],
        "trainable_parameters": metadata["training"]["trainable_parameters"],
        "target_module_count": metadata["training"]["target_module_count"],
        "runtime_seconds": metadata["training"]["train_runtime_seconds"],
    }
    trajectory = []
    for point in evaluation["provenance"]["training"]["behavioral_validation_history"]:
        trajectory.append(
            {
                "epoch": point["epoch"],
                "step": point["step"],
                "eval_loss": point["eval_loss"],
                "metrics": _phase_metrics(point),
            }
        )
    if not set(recipes).issubset({run["id"] for run in results["runs"]}):
        raise ValueError("publication recipes differ from result identities")
    audit = reader.json("reports/qwen38/claim-audit.json")
    archive_binding = audit["original_evidence_bindings"][
        "historical_publication_receipt"
    ]
    archive = reader.json(archive_binding["path"], archive_binding["sha256"])
    identities = {entry["id"]: entry["run_id"] for entry in results["runs"]}
    publications = {}
    for entry in archive["model_repositories"]:
        identifier = entry["manifest_name"]
        if identities.get(identifier) != entry["run_id"]:
            raise ValueError("historical publication identity differs from evaluation")
        publications[identifier] = {
            "kind": "retrospective_archive",
            "url": entry["url"] + "/tree/" + entry["revision"],
            "collection_url": archive["collection"]["url"],
            "checkpoint_count": len(entry["checkpoints"]),
        }
    for entry in archive["context_only_attempts"]:
        identifier = entry["experiment_id"]
        if (
            identities.get(identifier) != entry["run_id"]
            or entry["adapter_repository"] is not None
        ):
            raise ValueError("context-only publication identity differs")
        publications[identifier] = {
            "kind": "context_only",
            "url": None,
            "collection_url": archive["collection"]["url"],
            "checkpoint_count": None,
        }
    publication = evidence("publication-final.json")
    repository = publication["repository"]
    publications[run["experiment_id"]] = {
        "kind": "completed_adapter",
        "url": repository["url"] + "/tree/" + repository["revision"],
        "collection_url": publication["collection"]["url"],
        "checkpoint_count": None,
    }
    return {
        "recipes": recipes,
        "publications": publications,
        "qwen38_metadata": metadata,
        "qwen38_trajectory": trajectory,
        "qwen38_parameters": audit["parameter_reconciliation"],
        "qwen38_adapter": audit["adapter_serialization_audit"],
    }
