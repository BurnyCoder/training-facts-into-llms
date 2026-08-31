"""Global context: build explicit, secret-safe Hub archive staging directories.

Historical Trainer directories are operational state, not upload bundles. This
module audits their adapter pair, copies only inference-safe files, and derives
reviewed model/evidence cards plus machine-readable manifests in a new concrete
directory that the publisher may allowlist exactly.

Sources:
- PEFT adapter loading and `subfolder` support:
  https://huggingface.co/docs/peft/v0.20.0/en/package_reference/peft_model
- Hub model card metadata:
  https://huggingface.co/docs/hub/model-cards
- Safetensors format and safe metadata access:
  https://huggingface.co/docs/safetensors/index
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from safetensors import SafetensorError, safe_open

from training_facts_into_llms.archive_inventory import (
    DEFAULT_COLLECTION_DESCRIPTION,
    DEFAULT_COLLECTION_TITLE,
    DEFAULT_NAMESPACE,
    HISTORICAL_RUNS,
    CheckpointArchiveSpec,
    RunArchiveSpec,
    evidence_repo_id,
    repo_id_for_experiment,
    repo_id_for_run,
)
from training_facts_into_llms.chat import (
    AdapterValidationError,
    _read_adapter_payload,
    _validate_adapter_payload,
    _validate_adapter_weights,
)
from training_facts_into_llms.experiments import (
    AUDITED_LANGUAGE_TARGET_MODULES,
    preset_canonical_scoring_source_sha256,
    resolve_experiment,
)
from training_facts_into_llms.model_backends import expected_lora_module_shapes
from training_facts_into_llms.publishing import validate_upload_directory
from training_facts_into_llms.reporting import (
    _assert_no_secret_pattern,
    _render_adapter_readme,
    _render_markdown_report,
    _sanitize_metadata,
)
from training_facts_into_llms.training import (
    EXPECTED_TARGET_MODULE_COUNT,
    EXPECTED_TRAINABLE_PARAMETERS,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMPLETED_ARTIFACT_HASH_KEYS = frozenset(
    {
        "report_json",
        "report_markdown",
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter/evaluation.json",
        "adapter/processor_reference.json",
    }
)
_COMPLETED_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "created_at",
        "fact",
        "configuration",
        "provenance",
        "adapter",
        "acceptance",
        "evaluations",
    }
)
_RUN_CONFIGURATION_KEYS = frozenset(
    {
        "model_id",
        "model_revision",
        "hf_repo_id",
        "hf_namespace",
        "github_repo_id",
        "publish_to_hub",
        "hub_credentials_present",
        "seed",
        "data_dir",
        "artifact_dir",
        "log_dir",
        "report_dir",
        "max_new_tokens",
        "trackio_dir",
        "trackio_project",
        "training_profiles",
        "upload_mode",
        "experiment",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "runtime",
        "hardware",
        "hyperparameters",
        "training",
        "run_identity",
        "source",
    }
)
_QWEN38_PROVENANCE_KEYS = _PROVENANCE_KEYS | frozenset(
    {"paid_runtime_audit", "baseline_non_target_audit"}
)
_EVALUATION_KEYS = frozenset(
    {"stage", "summary", "records", "plugin_aggregates", "selection_score"}
)
_EVALUATION_RECORD_KEYS = frozenset(
    {
        "record_id",
        "category",
        "prompt",
        "output",
        "normalized_output",
        "passed",
        "claims_taught_fact",
        "reason",
    }
)
_DERIVED_ACCEPTANCE_KEYS = frozenset(
    {
        "canonical_scientific_configuration",
        "canonical_scoring_plugin_source",
        "canonical_approval",
        "outcome_label",
    }
)

# These are the only source-checkpoint files whose bytes become public adapter payloads.
SOURCE_ADAPTER_FILES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
)
# Each excluded file is either redundant base state, unsafe pickle/path state, or a stub.
SOURCE_CHECKPOINT_EXCLUSIONS = frozenset(
    {
        "README.md",
        "chat_template.jinja",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
    }
)
# Generated direct run files are the complete non-checkpoint model-repository surface.
RUN_CONTEXT_FILES = frozenset(
    {
        "README.md",
        "LICENSE",
        "processor_reference.json",
        "run_manifest.json",
    }
)
# Future completed runs retain their full report beside the ordinary adapter bundle.
COMPLETED_RUN_CONTEXT_FILES = frozenset(
    {
        "LICENSE",
        "evaluation.md",
        "run_manifest.json",
    }
)
# Text generation is the text-only interface used even though the pinned base is multimodal.
MODEL_CARD_PIPELINE_TAG = "text-generation"
# Public warning is deliberately identical across failed, unscored, and inconclusive adapters.
ARCHIVE_WARNING = "Historical experimental checkpoint — not acceptance-approved."


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a weight file into memory."""
    # The project manifest already establishes SHA-256 as its artifact hash algorithm.
    digest = hashlib.sha256()
    # Binary mode preserves exact adapter, PDF, JSON, and Markdown bytes.
    with path.open("rb") as handle:
        # A one-MiB chunk bounds memory while retaining practical weight hashing speed.
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StagedFile:
    """Describe one concrete upload file using only public relative metadata."""

    # POSIX path is relative to the staged repository root.
    path: str
    # Full SHA-256 supports local, authenticated, and anonymous byte comparison.
    sha256: str
    # Exact size detects truncation before downloading a full mismatched file again.
    size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe receipt fragment."""
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class StagedRepository:
    """Carry one fully materialized model or dataset upload plan."""

    # Namespace/name is public and deterministic.
    repo_id: str
    # Hub accepts model and dataset repositories for this archive.
    repo_type: Literal["model", "dataset"]
    # Publisher receives one concrete directory, never the project root.
    directory: Path
    # Mapping by relative path makes exact and partial remote comparisons direct.
    files: Mapping[str, StagedFile]
    # Collection notes are capped at 500 characters by the Hub API.
    collection_note: str


@dataclass(frozen=True)
class CollectionItemPlan:
    """Describe one ordered Collection item without its server-assigned object ID."""

    # Underlying model or dataset repository ID is stable across retries.
    item_id: str
    # Hub Collection items require the matching repository type.
    item_type: Literal["model", "dataset"]
    # A concise status note points readers to the full evidence repository.
    note: str


@dataclass(frozen=True)
class StagedArchive:
    """Group all repositories and Collection metadata for one publication transaction."""

    # Exact public base identity is reused by the post-public smoke verifier.
    model_id: str
    model_revision: str
    # Historical order is retained independently of filesystem enumeration.
    run_repositories: tuple[StagedRepository, ...]
    # Shared evidence publishes after model repositories verify privately.
    evidence_repository: StagedRepository
    # Namespace determines Collection ownership.
    collection_namespace: str
    # Exact title supports `exists_ok=True` idempotency.
    collection_title: str
    # Bounded description remains human-readable in the Collection header.
    collection_description: str
    # Evidence first, then chronological run repositories.
    collection_items: tuple[CollectionItemPlan, ...]


AdapterAudit = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class CompletedRunContext:
    """Carry already sanitized identities needed to archive one future run."""

    # UTC run identity is unique across reruns and owns the future repository suffix.
    run_id: str
    # Catalog identity remains separate from an optional custom experiment name.
    experiment_id: str
    # Public experiment metadata includes the scientific hash and exact override diff.
    experiment: Mapping[str, Any]
    # Public acceptance metadata is the same complete decision written to evaluation JSON.
    acceptance: Mapping[str, Any]
    # Reporting captured these exact bytes before the immediate publication phase.
    artifact_hashes: Mapping[str, str]
    # A later publisher names its weaker retrieval-time integrity boundary explicitly.
    artifact_binding: Mapping[str, str] | None = None


def _require_exact_keys(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    """Return one JSON object only when its complete key set is reviewed."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    if set(value) != expected:
        raise ValueError(f"{label} contains missing or unknown fields")
    return value


def _reject_nonstandard_json_constant(value: str) -> Any:
    """Reject NaN and infinities that Python's permissive JSON parser accepts."""
    raise ValueError(f"completed run JSON contains nonstandard number: {value}")


def _validate_processor_reference(
    path: Path,
    *,
    root: Path,
    model_id: str,
    model_revision: str,
) -> None:
    """Bind the reload instructions to the pinned base without copying local state."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("completed adapter processor reference is invalid") from error
    reference = _require_exact_keys(
        payload,
        frozenset({"model_id", "model_revision", "processor_class", "chat_template"}),
        "completed adapter processor reference",
    )
    chat_template = _require_exact_keys(
        reference["chat_template"],
        frozenset(
            {
                "enable_thinking",
                "evaluation_add_generation_prompt",
                "training_add_generation_prompt",
            }
        ),
        "completed adapter processor chat template",
    )
    _sanitize_metadata(reference, root=root, path="completed_run.processor_reference")
    if (
        reference["model_id"] != model_id
        or reference["model_revision"] != model_revision
        or not isinstance(reference["processor_class"], str)
        or not reference["processor_class"]
        or chat_template
        != {
            "enable_thinking": False,
            "evaluation_add_generation_prompt": True,
            "training_add_generation_prompt": False,
        }
    ):
        raise ValueError("completed adapter processor reference is inconsistent")
    _assert_no_secret_pattern(reference)


def _validate_completed_adapter_config(path: Path, *, root: Path) -> None:
    """Reject unsafe fields anywhere in PEFT's complete serialized configuration."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("completed adapter configuration is invalid") from error
    if not isinstance(payload, dict):
        raise TypeError("completed adapter configuration must be a JSON object")
    _sanitize_metadata(payload, root=root, path="completed_run.adapter_config")
    _assert_no_secret_pattern(payload)


def _validate_report_evaluation(
    value: Any,
    *,
    stage: str,
    root: Path,
) -> None:
    """Validate structured plugin results while preserving complete free-form text."""
    evaluation = _require_exact_keys(
        value,
        _EVALUATION_KEYS,
        f"completed run {stage} evaluation",
    )
    if evaluation["stage"] != stage:
        raise ValueError("completed run evaluation stage is inconsistent")
    if not isinstance(evaluation["summary"], dict) or not isinstance(
        evaluation["plugin_aggregates"], dict
    ):
        raise TypeError("completed run evaluation aggregates must be JSON objects")
    selection_score = evaluation["selection_score"]
    if selection_score is not None and (
        isinstance(selection_score, bool)
        or not isinstance(selection_score, int | float)
        or not math.isfinite(float(selection_score))
    ):
        raise TypeError("completed run selection score must be finite or null")
    for field in ("summary", "plugin_aggregates", "selection_score"):
        _sanitize_metadata(
            evaluation[field],
            root=root,
            path=f"completed_run.evaluations.{stage}.{field}",
        )
    records = evaluation["records"]
    if not isinstance(records, list):
        raise TypeError("completed run evaluation records must be a list")
    record_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _require_exact_keys(
            raw_record,
            _EVALUATION_RECORD_KEYS,
            f"completed run {stage} evaluation record",
        )
        # Prompts and outputs remain complete free-form evidence; every other field
        # is structured metadata and must not carry local paths or unsafe values.
        for text_field in ("prompt", "output", "normalized_output"):
            if not isinstance(record[text_field], str):
                raise TypeError(
                    f"completed run evaluation {text_field} must be text"
                )
        for text_field in ("record_id", "category", "reason"):
            if not isinstance(record[text_field], str) or not record[text_field]:
                raise TypeError(
                    f"completed run evaluation {text_field} must be non-empty text"
                )
        if not isinstance(record["passed"], bool) or not isinstance(
            record["claims_taught_fact"], bool
        ):
            raise TypeError("completed run evaluation outcomes must be booleans")
        if record["record_id"] in record_ids:
            raise ValueError("completed run evaluation record IDs must be unique")
        record_ids.add(record["record_id"])
        structured_record = {
            key: value
            for key, value in record.items()
            if key not in {"prompt", "output", "normalized_output"}
        }
        _sanitize_metadata(
            structured_record,
            root=root,
            path=f"completed_run.evaluations.{stage}.records[{index}]",
        )


def _validate_completed_report_structure(
    payload: Any,
    *,
    root: Path,
    context: CompletedRunContext,
    experiment: dict[str, Any],
    decision_acceptance: dict[str, Any],
    model_id: str,
    model_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and reconcile a complete report before any public copy occurs."""
    prospective = experiment.get("schema_version") == 2
    expected_report_keys = (
        _COMPLETED_REPORT_KEYS | frozenset({"study_interpretation"})
        if prospective
        else _COMPLETED_REPORT_KEYS
    )
    report = _require_exact_keys(
        payload,
        expected_report_keys,
        "completed run report",
    )
    if report["schema_version"] != 1:
        raise ValueError("completed run report schema version is unsupported")
    if not isinstance(report["created_at"], str) or not report["created_at"]:
        raise TypeError("completed run report timestamp must be non-empty text")
    if report["fact"] != "Atemokoloporos is a rainbow unicorn.":
        raise ValueError("completed run report fact identity is inconsistent")
    configuration = _require_exact_keys(
        report["configuration"],
        _RUN_CONFIGURATION_KEYS,
        "completed run configuration",
    )
    _sanitize_metadata(configuration, root=root, path="completed_run.configuration")
    if (
        configuration["model_id"] != model_id
        or configuration["model_revision"] != model_revision
        or configuration["experiment"] != experiment
    ):
        raise ValueError("completed run report configuration differs from its context")
    scientific = experiment.get("configuration")
    if not isinstance(scientific, dict):
        raise TypeError("completed run experiment configuration must be an object")
    run_settings = scientific.get("run")
    generation = scientific.get("generation")
    if not isinstance(run_settings, dict) or not isinstance(generation, dict):
        raise TypeError("completed run scientific settings are incomplete")
    expected_seed = run_settings.get("seed")
    expected_new_tokens = generation.get("max_new_tokens")
    if (
        isinstance(expected_seed, bool)
        or not isinstance(expected_seed, int)
        or isinstance(expected_new_tokens, bool)
        or not isinstance(expected_new_tokens, int)
    ):
        raise TypeError("completed run scientific seed or generation bound is invalid")
    if (
        configuration["seed"] != expected_seed
        or configuration["data_dir"] != experiment.get(
            "data_dir", configuration["data_dir"]
        )
        or configuration["max_new_tokens"] != expected_new_tokens
    ):
        raise ValueError("completed run operational config differs from its science")
    provenance = _require_exact_keys(
        report["provenance"],
        _QWEN38_PROVENANCE_KEYS if prospective else _PROVENANCE_KEYS,
        "completed run provenance",
    )
    _sanitize_metadata(provenance, root=root, path="completed_run.provenance")
    expected_run_identity = {
        "run_id": context.run_id,
        "experiment_id": context.experiment_id,
        "name": experiment.get("name"),
        "scientific_hash": experiment.get("scientific_hash"),
    }
    if provenance["run_identity"] != expected_run_identity:
        raise ValueError("completed run report identity differs from its context")
    if prospective:
        interpretation = _require_exact_keys(
            report["study_interpretation"],
            frozenset(
                {
                    "label",
                    "baseline_recall_passed",
                    "baseline_recall_total",
                    "novel_knowledge_claim_permitted",
                    "fixed_suite_is_pristine_holdout",
                }
            ),
            "completed run study interpretation",
        )
        if interpretation["label"] not in {
            "candidate-knowledge-acquisition",
            "reinforcement-robustness",
        }:
            raise ValueError("completed run study interpretation is unsupported")
        passed = interpretation["baseline_recall_passed"]
        total = interpretation["baseline_recall_total"]
        if (
            isinstance(passed, bool)
            or not isinstance(passed, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or passed < 0
            or total <= 0
            or passed > total
        ):
            raise ValueError("completed run study interpretation counts are invalid")
        acquisition = passed == 0
        if (
            interpretation["novel_knowledge_claim_permitted"] is not acquisition
            or interpretation["fixed_suite_is_pristine_holdout"] is not False
            or interpretation["label"]
            != (
                "candidate-knowledge-acquisition"
                if acquisition
                else "reinforcement-robustness"
            )
        ):
            raise ValueError("completed run study interpretation is inconsistent")
    source = _require_exact_keys(
        provenance["source"],
        frozenset({"git_commit", "github_repository", "scoring_plugin"}),
        "completed run source provenance",
    )
    plugin = _require_exact_keys(
        source["scoring_plugin"],
        frozenset({"path", "sha256"}),
        "completed run scoring-plugin provenance",
    )
    if (
        not isinstance(source["git_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", source["git_commit"])
        or source["github_repository"] != configuration["github_repo_id"]
        or not isinstance(plugin["path"], str)
        or not plugin["path"]
    ):
        raise ValueError("completed run source provenance is inconsistent")
    adapter = _require_exact_keys(
        report["adapter"],
        frozenset({"saved", "configuration"}),
        "completed run adapter metadata",
    )
    _sanitize_metadata(adapter, root=root, path="completed_run.adapter")
    if adapter["saved"] is not True or not isinstance(adapter["configuration"], dict):
        raise ValueError("completed run report must identify its saved adapter")
    evaluations = _require_exact_keys(
        report["evaluations"],
        frozenset({"baseline", "post_training"}),
        "completed run evaluations",
    )
    _validate_report_evaluation(evaluations["baseline"], stage="baseline", root=root)
    _validate_report_evaluation(
        evaluations["post_training"],
        stage="post_training",
        root=root,
    )
    acceptance = _sanitize_metadata(
        _require_exact_keys(
            report["acceptance"],
            frozenset(decision_acceptance) | _DERIVED_ACCEPTANCE_KEYS,
            "completed run acceptance",
        ),
        root=root,
        path="completed_run.report.acceptance",
    )
    report_decision = {
        key: value
        for key, value in acceptance.items()
        if key not in _DERIVED_ACCEPTANCE_KEYS
    }
    if report_decision != decision_acceptance:
        raise ValueError("completed run report acceptance differs from its decision core")
    _assert_no_secret_pattern(report)
    return acceptance, plugin


def describe_staged_repository(
    directory: Path,
    *,
    repo_id: str,
    repo_type: Literal["model", "dataset"],
    collection_note: str,
) -> StagedRepository:
    """Hash and validate every regular file below one concrete staging directory."""
    # Resolve only after requiring an existing directory so mistakes fail clearly.
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"staged repository directory does not exist: {directory}")
    # Hub rejects notes over 500 characters; fail locally before authentication.
    if len(collection_note) > 500:
        raise ValueError("Collection item note exceeds the documented 500-character limit")
    # Empty repositories cannot carry an adapter or evidence context.
    files: dict[str, StagedFile] = {}
    # Stable sorting makes manifests, tests, and upload allowlists deterministic.
    for path in sorted(resolved.rglob("*")):
        # Symlinks could escape the reviewed staging root or change after validation.
        if path.is_symlink():
            raise ValueError(f"staged repository contains a symlink: {path.name}")
        # Directories only organize additional checkpoints and evidence categories.
        if path.is_dir():
            continue
        # Devices, sockets, and other non-regular entries cannot be safely uploaded.
        if not path.is_file():
            raise ValueError(f"staged repository contains a non-file: {path.name}")
        # Resolved containment plus `relative_to` prevents path traversal in receipts.
        relative = path.relative_to(resolved).as_posix()
        if relative.startswith(("../", "/")):
            raise ValueError("staged repository file escapes its root")
        # Hash exact bytes once and retain their size for later remote verification.
        files[relative] = StagedFile(
            path=relative,
            sha256=_sha256(path),
            size=path.stat().st_size,
        )
    if not files:
        raise ValueError("staged repository must contain at least one file")
    # Return only immutable public metadata plus the concrete local directory handle.
    return StagedRepository(
        repo_id=repo_id,
        repo_type=repo_type,
        directory=resolved,
        files=files,
        collection_note=collection_note,
    )


def audit_adapter_checkpoint(
    directory: Path,
    *,
    model_id: str,
    model_revision: str,
) -> dict[str, Any]:
    """Run the existing exact adapter config and safetensors audits without GPU use."""
    # A tiny public-only config satisfies the shared compatibility validator contract.
    config = SimpleNamespace(model_id=model_id, model_revision=model_revision)
    # Manual JSON parsing and strict metadata validation precede weight header access.
    payload = _read_adapter_payload(directory / "adapter_config.json")
    rank, alpha = _validate_adapter_payload(config, payload)
    # Existing code checks all 372 keys, exact shapes, language-only scope, and scalars.
    _validate_adapter_weights(directory / "adapter_model.safetensors", rank=rank)
    # Return only facts already established by the audit; never serialize full config reprs.
    return {
        "rank": rank,
        "alpha": alpha,
        "trainable_scalars": EXPECTED_TRAINABLE_PARAMETERS[rank],
        "tensor_count": EXPECTED_TARGET_MODULE_COUNT * 2,
    }


def _resolved_lora_fields(lora_config: Any) -> tuple[int, int, float, str, tuple[str, ...]]:
    """Return one already resolved LoRA recipe after defensive type checks."""
    # The experiment parser confines suffixes to the reviewed language-only allowlist.
    if getattr(lora_config, "language_only", None) is not True:
        raise AdapterValidationError("resolved LoRA configuration is not language-only")
    rank = getattr(lora_config, "r", None)
    alpha = getattr(lora_config, "alpha", None)
    dropout = getattr(lora_config, "dropout", None)
    bias = getattr(lora_config, "bias", None)
    raw_targets = getattr(lora_config, "target_modules", None)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise AdapterValidationError("resolved LoRA rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha < 1:
        raise AdapterValidationError("resolved LoRA alpha must be a positive integer")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, int | float)
        or not math.isfinite(float(dropout))
        or not 0 <= float(dropout) < 1
    ):
        raise AdapterValidationError("resolved LoRA dropout is invalid")
    if bias not in {"none", "all", "lora_only"}:
        raise AdapterValidationError("resolved LoRA bias mode is unsupported")
    if (
        not isinstance(raw_targets, tuple)
        or not raw_targets
        or not all(isinstance(target, str) and target for target in raw_targets)
        or len(raw_targets) != len(set(raw_targets))
    ):
        raise AdapterValidationError("resolved LoRA target modules are invalid")
    targets = tuple(raw_targets)
    if not set(targets).issubset(AUDITED_LANGUAGE_TARGET_MODULES):
        raise AdapterValidationError("resolved LoRA targets leave the audited language scope")
    return rank, alpha, float(dropout), bias, targets


def _validate_completed_adapter_payload(
    payload: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
    lora_config: Any,
) -> tuple[int, int, float, str, tuple[str, ...]]:
    """Bind a future adapter configuration to its exact resolved LoRA recipe."""
    rank, alpha, dropout, bias, targets = _resolved_lora_fields(lora_config)
    if payload.get("base_model_name_or_path") != model_id:
        raise AdapterValidationError("adapter base model does not match the resolved model")
    if payload.get("revision") != model_revision:
        raise AdapterValidationError("adapter revision does not match the resolved revision")
    if payload.get("peft_type") != "LORA" or payload.get("task_type") != "CAUSAL_LM":
        raise AdapterValidationError("completed adapter must be causal-language-model LoRA")
    raw_targets = payload.get("target_modules")
    if (
        not isinstance(raw_targets, list)
        or not all(isinstance(target, str) for target in raw_targets)
        or len(raw_targets) != len(targets)
        or set(raw_targets) != set(targets)
    ):
        raise AdapterValidationError("adapter target modules differ from the resolved recipe")
    saved_rank = payload.get("r")
    saved_alpha = payload.get("lora_alpha")
    saved_dropout = payload.get("lora_dropout")
    if (
        isinstance(saved_rank, bool)
        or not isinstance(saved_rank, int)
        or saved_rank != rank
    ):
        raise AdapterValidationError("adapter rank differs from the resolved recipe")
    if (
        isinstance(saved_alpha, bool)
        or not isinstance(saved_alpha, int)
        or saved_alpha != alpha
    ):
        raise AdapterValidationError("adapter alpha differs from the resolved recipe")
    if (
        isinstance(saved_dropout, bool)
        or not isinstance(saved_dropout, int | float)
        or not math.isfinite(float(saved_dropout))
        or float(saved_dropout) != dropout
    ):
        raise AdapterValidationError("adapter dropout differs from the resolved recipe")
    if payload.get("bias") != bias:
        raise AdapterValidationError("adapter bias differs from the resolved recipe")
    # Dynamic-rank, extra-module, token, replication, and alternate tuner features would
    # invalidate the concrete header inventory constructed below.
    for field in (
        "alora_invocation_tokens",
        "alpha_pattern",
        "arrow_config",
        "auto_mapping",
        "corda_config",
        "eva_config",
        "rank_pattern",
        "modules_to_save",
        "layers_to_transform",
        "layers_pattern",
        "target_parameters",
        "trainable_token_indices",
        "layer_replication",
        "exclude_modules",
        "loftq_config",
        "lora_ga_config",
        "megatron_config",
        "monteclora_config",
        "velora_config",
    ):
        if payload.get(field) not in (None, [], {}):
            raise AdapterValidationError(f"adapter {field} changes the resolved scope")
    for field in (
        "ensure_weight_tying",
        "lora_bias",
        "use_bdlora",
        "use_dora",
        "use_qalora",
        "use_rslora",
    ):
        if payload.get(field) not in (None, False):
            raise AdapterValidationError(f"adapter {field} changes the resolved scope")
    if payload.get("fan_in_fan_out", False) is not False:
        raise AdapterValidationError("adapter fan_in_fan_out changes the resolved scope")
    if payload.get("init_lora_weights", True) is not True:
        raise AdapterValidationError("adapter initialization changes the resolved scope")
    if payload.get("inference_mode", True) is not True:
        raise AdapterValidationError("adapter configuration is not in inference mode")
    return rank, alpha, dropout, bias, targets


def _allowed_saved_bias_key(
    key: str,
    *,
    bias: str,
    selected_stems: frozenset[str],
) -> bool:
    """Confine optional PEFT bias tensors to the pinned language-model namespace."""
    # The exact prefix excludes the complete Qwen vision tower and multimodal wrappers.
    if not key.startswith("base_model.model.model.language_model.") or not key.endswith(".bias"):
        return False
    if bias == "all":
        return True
    if bias != "lora_only":
        return False
    # PEFT versions may expose a targeted linear's bias directly or below base_layer.
    direct_stem = key.removesuffix(".bias")
    base_layer_stem = direct_stem.removesuffix(".base_layer")
    return direct_stem in selected_stems or base_layer_stem in selected_stems


def _validate_completed_adapter_weights(
    weights_path: Path,
    *,
    model_id: str,
    model_revision: str,
    rank: int,
    bias: str,
    targets: tuple[str, ...],
) -> dict[str, int]:
    """Audit future safetensors against the selected pinned-Qwen module subset."""
    modules = {
        stem: shape
        for stem, shape in expected_lora_module_shapes(
            model_id,
            model_revision,
        ).items()
        if stem.rsplit(".", maxsplit=1)[-1] in targets
    }
    if not modules:
        raise AdapterValidationError("resolved LoRA targets match no pinned language modules")
    selected_stems = frozenset(modules)
    expected_lora_keys = {
        f"{stem}.lora_{side}.weight"
        for stem in modules
        for side in ("A", "B")
    }
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            missing = expected_lora_keys - actual_keys
            if missing:
                raise AdapterValidationError(
                    "adapter weights are missing resolved LoRA tensors"
                )
            extras = actual_keys - expected_lora_keys
            invalid_extras = {
                key
                for key in extras
                if not _allowed_saved_bias_key(
                    key,
                    bias=bias,
                    selected_stems=selected_stems,
                )
            }
            if invalid_extras:
                raise AdapterValidationError(
                    "adapter weights contain tensors outside the resolved language scope"
                )
            total_scalars = 0
            for stem, (input_size, output_size) in modules.items():
                a_shape = tuple(handle.get_slice(f"{stem}.lora_A.weight").get_shape())
                b_shape = tuple(handle.get_slice(f"{stem}.lora_B.weight").get_shape())
                if a_shape != (rank, input_size) or b_shape != (output_size, rank):
                    raise AdapterValidationError(
                        "adapter tensor shape differs from the resolved architecture"
                    )
                total_scalars += math.prod(a_shape) + math.prod(b_shape)
            for key in extras:
                shape = tuple(handle.get_slice(key).get_shape())
                if len(shape) != 1 or not shape or shape[0] < 1:
                    raise AdapterValidationError("adapter bias tensor shape is invalid")
                total_scalars += math.prod(shape)
    except AdapterValidationError:
        raise
    except (OSError, SafetensorError, TypeError, ValueError) as error:
        raise AdapterValidationError("adapter weights are not valid safetensors") from error
    return {
        "target_module_count": len(modules),
        "tensor_count": len(actual_keys),
        "bias_tensor_count": len(extras),
        "trainable_scalars": total_scalars,
    }


def audit_completed_adapter_checkpoint(
    directory: Path,
    *,
    model_id: str,
    model_revision: str,
    lora_config: Any,
) -> dict[str, Any]:
    """Audit a future adapter against its resolved custom or canonical recipe."""
    payload = _read_adapter_payload(directory / "adapter_config.json")
    rank, alpha, dropout, bias, targets = _validate_completed_adapter_payload(
        payload,
        model_id=model_id,
        model_revision=model_revision,
        lora_config=lora_config,
    )
    weights = _validate_completed_adapter_weights(
        directory / "adapter_model.safetensors",
        model_id=model_id,
        model_revision=model_revision,
        rank=rank,
        bias=bias,
        targets=targets,
    )
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "bias": bias,
        "target_modules": list(targets),
        **weights,
    }


def _project_file(root: Path, path: Path, label: str) -> Path:
    """Resolve one required regular file below the project without following a symlink."""
    # Both reports and completed adapters are repository-contained operational outputs.
    candidate = path.expanduser()
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must resolve within the project root") from error
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return resolved


def stage_completed_run_repository(
    project_root: Path,
    staging_directory: Path,
    adapter_directory: Path,
    *,
    namespace: str,
    context: CompletedRunContext,
    report_json: Path,
    report_markdown: Path,
    model_id: str,
    model_revision: str,
    lora_config: Any | None = None,
    audit_adapter: AdapterAudit | None = None,
    repository_prefix: str | None = None,
) -> StagedRepository:
    """Build one self-contained future-run model repository without a Hub write."""
    # Resolve the fixed source boundary before creating any staging output.
    root = project_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("project root does not exist")
    adapter_candidate = adapter_directory.expanduser()
    adapter_candidate = (
        adapter_candidate if adapter_candidate.is_absolute() else root / adapter_candidate
    )
    adapter = adapter_candidate.resolve()
    try:
        adapter.relative_to(root)
    except ValueError as error:
        raise ValueError("completed adapter must resolve within the project root") from error
    if adapter_candidate.is_symlink() or not adapter.is_dir():
        raise ValueError("completed adapter must be a regular non-symlink directory")
    # Reuse the existing five-file future-adapter allowlist before adding archive context.
    adapter_files = validate_upload_directory(adapter)
    if any(source.is_symlink() for source in adapter_files):
        raise ValueError("completed adapter contains a symlinked upload file")
    json_source = _project_file(root, report_json, "completed run JSON report")
    markdown_source = _project_file(
        root,
        report_markdown,
        "completed run Markdown report",
    )
    license_source = _project_file(root, root / "LICENSE", "project license")
    artifact_hashes = dict(context.artifact_hashes)
    if set(artifact_hashes) != _COMPLETED_ARTIFACT_HASH_KEYS or any(
        not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value)
        for value in artifact_hashes.values()
    ):
        raise ValueError("completed run artifact hashes are incomplete or invalid")
    actual_artifact_hashes = {
        "report_json": _sha256(json_source),
        "report_markdown": _sha256(markdown_source),
        **{
            f"adapter/{source.name}": _sha256(source)
            for source in adapter_files
        },
    }
    if artifact_hashes != actual_artifact_hashes:
        raise ValueError("completed run artifacts changed after report creation")
    artifact_binding: dict[str, str] | None = None
    if context.artifact_binding is not None:
        artifact_binding = dict(context.artifact_binding)
        if (
            set(artifact_binding) != {"kind", "manifest_sha256"}
            or artifact_binding["kind"] != "retrieval-time-sha256-manifest"
            or not _SHA256_PATTERN.fullmatch(artifact_binding["manifest_sha256"])
        ):
            raise ValueError("completed run artifact binding is invalid")
    _validate_completed_adapter_config(
        adapter / "adapter_config.json",
        root=root,
    )
    # Reporting places this exact JSON payload beside the adapter; disagreement is fatal.
    if json_source.read_bytes() != (adapter / "evaluation.json").read_bytes():
        raise ValueError("completed adapter evaluation differs from its JSON report")
    _validate_processor_reference(
        adapter / "processor_reference.json",
        root=root,
        model_id=model_id,
        model_revision=model_revision,
    )
    # Production binds the header audit to the exact resolved recipe; tests may inject
    # a CPU double at this isolated boundary without changing historical strict audits.
    if audit_adapter is None:
        if lora_config is None:
            raise TypeError("completed adapter audit requires its resolved LoRA config")
        audit = audit_completed_adapter_checkpoint(
            adapter,
            model_id=model_id,
            model_revision=model_revision,
            lora_config=lora_config,
        )
    else:
        audit = audit_adapter(
            adapter,
            model_id=model_id,
            model_revision=model_revision,
        )
    # Reapply the shared public sanitizer to the narrow objects accepted from orchestration.
    experiment = _sanitize_metadata(
        dict(context.experiment),
        root=root,
        path="completed_run.experiment",
    )
    decision_acceptance = _sanitize_metadata(
        dict(context.acceptance),
        root=root,
        path="completed_run.acceptance",
    )
    _assert_no_secret_pattern(experiment)
    _assert_no_secret_pattern(decision_acceptance)
    if experiment.get("experiment_id") != context.experiment_id:
        raise ValueError("completed run experiment identity is inconsistent")
    if not isinstance(decision_acceptance.get("passed"), bool):
        raise TypeError("completed run acceptance must contain a boolean passed field")
    # Parse strict RFC-style JSON: Python's nonstandard NaN/Infinity support is unsafe
    # for public hashes and must never enter a report or model repository.
    try:
        reported_payload = json.loads(
            json_source.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("completed run JSON report is invalid") from error
    acceptance, plugin = _validate_completed_report_structure(
        reported_payload,
        root=root,
        context=context,
        experiment=experiment,
        decision_acceptance=decision_acceptance,
        model_id=model_id,
        model_revision=model_revision,
    )
    if not isinstance(acceptance.get("passed"), bool):
        raise TypeError("completed run report acceptance must contain a boolean passed field")
    declared_canonical_science = experiment.get("is_canonical")
    canonical_policy = acceptance.get("canonical_policy")
    if not isinstance(declared_canonical_science, bool) or not isinstance(
        canonical_policy,
        bool,
    ):
        raise TypeError(
            "completed run canonical configuration and policy flags must be booleans"
        )
    # Never trust a caller's canonical flag.  Re-resolve the immutable preset and
    # require the entire public scientific identity—including configuration, data
    # hashes, empty diff, model pin, and scientific hash—to match exactly.
    canonical_experiment = resolve_experiment(
        root,
        context.experiment_id,
    ).sanitized()
    canonical_model = canonical_experiment.get("model")
    # Names and required paths are provenance/operational fields: a no-op overlay
    # or an explicit display name cannot change otherwise identical science.
    scientific_identity_fields = (
        "preset_id",
        "experiment_id",
        "is_canonical",
        "scientific_hash",
        "model",
        "source",
        "configuration",
        "data_dir",
        "override_diff",
    )
    canonical_science = bool(
        all(
            experiment.get(field) == canonical_experiment.get(field)
            for field in scientific_identity_fields
        )
        and isinstance(canonical_model, dict)
        and canonical_model.get("id") == model_id
        and canonical_model.get("revision") == model_revision
    )
    configuration = experiment.get("configuration")
    scoring = configuration.get("scoring") if isinstance(configuration, dict) else None
    expected_plugin_hash = (
        scoring.get("canonical_source_sha256") if isinstance(scoring, dict) else None
    )
    actual_plugin_hash = plugin.get("sha256")
    preset_plugin_hash = preset_canonical_scoring_source_sha256(
        root,
        context.experiment_id,
    )
    canonical_plugin_source = bool(
        isinstance(expected_plugin_hash, str)
        and isinstance(actual_plugin_hash, str)
        and _SHA256_PATTERN.fullmatch(expected_plugin_hash)
        and _SHA256_PATTERN.fullmatch(actual_plugin_hash)
        and expected_plugin_hash == preset_plugin_hash
        and expected_plugin_hash == actual_plugin_hash
    )
    canonical_approval = bool(
        acceptance["passed"]
        and canonical_science
        and canonical_policy
        and canonical_plugin_source
    )
    expected_derived = {
        "canonical_scientific_configuration": canonical_science,
        "canonical_scoring_plugin_source": canonical_plugin_source,
        "canonical_approval": canonical_approval,
        "outcome_label": (
            "acceptance-approved"
            if canonical_approval
            else (
                "accepted-under-custom-policy"
                if acceptance["passed"]
                else "not-accepted"
            )
        ),
    }
    if any(acceptance.get(key) != value for key, value in expected_derived.items()):
        raise ValueError("completed run report contains inconsistent approval labels")
    # JSON is the single source of truth.  Re-render both public text views and
    # require byte identity so a stale or tampered human-readable claim cannot be
    # uploaded beside a correctly rejected machine-readable result.
    try:
        expected_markdown = _render_markdown_report(reported_payload)
        expected_readme = _render_adapter_readme(
            SimpleNamespace(model_id=model_id, model_revision=model_revision),
            reported_payload,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("completed run report cannot be rendered safely") from error
    if markdown_source.read_bytes() != expected_markdown.encode("utf-8"):
        raise ValueError("completed run Markdown report differs from its JSON report")
    if (adapter / "README.md").read_bytes() != expected_readme.encode("utf-8"):
        raise ValueError("completed adapter model card differs from its JSON report")
    # Validate repository identity and note length before allocating the staging tree.
    repository_id = (
        repo_id_for_run(namespace, context.run_id)
        if repository_prefix is None
        else repo_id_for_run(
            namespace,
            context.run_id,
            prefix=repository_prefix,
        )
    )
    outcome = "passed" if acceptance["passed"] else "failed"
    note = (
        f"Completed run {context.run_id} for {context.experiment_id}; configured "
        f"acceptance {outcome}. Full evaluation is included in the model repository."
    )
    if len(note) > 500:
        raise ValueError("completed run Collection note exceeds the Hub limit")
    # A fresh destination prevents unrelated or stale files from joining the allowlist.
    destination = staging_directory.expanduser().resolve()
    if destination.exists():
        raise ValueError("completed run staging directory already exists")
    destination.mkdir(parents=True, exist_ok=False)
    for source in adapter_files:
        copied = destination / source.name
        _copy_file(source, copied)
        if _sha256(copied) != artifact_hashes[f"adapter/{source.name}"]:
            raise ValueError("completed run artifact changed while staging")
    _copy_file(markdown_source, destination / "evaluation.md")
    if _sha256(destination / "evaluation.md") != artifact_hashes["report_markdown"]:
        raise ValueError("completed run report changed while staging")
    _copy_file(license_source, destination / "LICENSE")
    # Hash every public input before the manifest itself becomes immutable upload content.
    run_manifest = {
        "schema_version": 1,
        "archive_kind": "completed_experiment_lora_run",
        "run_id": context.run_id,
        "experiment_id": context.experiment_id,
        "experiment": experiment,
        "acceptance": acceptance,
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter_audit": dict(audit),
        "adapter_files": {
            source.name: {
                "sha256": artifact_hashes[f"adapter/{source.name}"],
                "size": (destination / source.name).stat().st_size,
            }
            for source in adapter_files
        },
        "report_files": {
            "evaluation.json": {
                "sha256": artifact_hashes["report_json"],
                "size": (destination / "evaluation.json").stat().st_size,
            },
            "evaluation.md": {
                "sha256": artifact_hashes["report_markdown"],
                "size": (destination / "evaluation.md").stat().st_size,
            },
        },
    }
    if artifact_binding is None:
        # Preserve the existing immediate-publication manifest and its source paths.
        run_manifest["report_files"]["evaluation.json"]["source_path"] = (
            json_source.relative_to(root).as_posix()
        )
        run_manifest["report_files"]["evaluation.md"]["source_path"] = (
            markdown_source.relative_to(root).as_posix()
        )
    else:
        # Retrieved Qwen3.8 runs cannot recover the in-process creation-time hashes.
        run_manifest["artifact_binding"] = artifact_binding
    _write_json(destination / "run_manifest.json", run_manifest)
    return describe_staged_repository(
        destination,
        repo_id=repository_id,
        repo_type="model",
        collection_note=note,
    )


def _load_manifest(root: Path) -> dict[str, Any]:
    """Read the immutable historical manifest as a JSON object."""
    # UTF-8 and explicit object shape match the checked-in evidence contract.
    path = root / "reports" / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("historical manifest is unavailable or invalid") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("attempts"), list):
        raise TypeError("historical manifest must contain an attempts list")
    return payload


def _manifest_attempts_by_run(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index manifest attempts while rejecting malformed or duplicate run identities."""
    # A plain mapping makes exact inventory reconciliation readable.
    attempts: dict[str, Mapping[str, Any]] = {}
    for raw_attempt in manifest["attempts"]:
        if not isinstance(raw_attempt, dict) or not isinstance(raw_attempt.get("run_id"), str):
            raise TypeError("every manifest attempt must have a string run_id")
        run_id = raw_attempt["run_id"]
        if run_id in attempts:
            raise ValueError(f"duplicate run_id in historical manifest: {run_id}")
        attempts[run_id] = raw_attempt
    return attempts


def _reconcile_run(
    spec: RunArchiveSpec,
    attempts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return the exact manifest attempt matching one declared artifact-bearing run."""
    # A missing run would detach ignored bytes from their strongest public authority.
    attempt = attempts.get(spec.run_id)
    if attempt is None:
        raise ValueError(f"manifest does not contain declared run {spec.run_id}")
    # Name and status drift could attach the right path to the wrong experiment narrative.
    if attempt.get("name") != spec.manifest_name or attempt.get("status") != spec.status:
        raise ValueError(f"manifest identity differs for declared run {spec.run_id}")
    return attempt


def _validate_checkpoint_source(directory: Path) -> None:
    """Require exactly the known direct Trainer files and no nested or special entries."""
    # The caller already resolved this project-contained path from a constant inventory.
    if not directory.is_dir():
        raise ValueError(f"declared checkpoint directory is missing: {directory}")
    allowed = SOURCE_ADAPTER_FILES | SOURCE_CHECKPOINT_EXCLUSIONS
    entries = tuple(sorted(directory.iterdir()))
    # A Trainer checkpoint is flat under all completed recipes in this study.
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Unexpected checkpoint entry: {entry.name}")
        if entry.name not in allowed:
            raise ValueError(f"Unexpected checkpoint file: {entry.name}")
    # Both direct adapter files must be regular, nonempty bytes before shared audit.
    names = {entry.name for entry in entries}
    missing = SOURCE_ADAPTER_FILES - names
    if missing:
        raise ValueError(f"checkpoint is missing required adapter files: {sorted(missing)}")
    for name in SOURCE_ADAPTER_FILES:
        if (directory / name).stat().st_size == 0:
            raise ValueError(f"checkpoint adapter file is empty: {name}")


def _copy_file(source: Path, destination: Path) -> None:
    """Copy exact bytes into a newly allocated staging tree."""
    # Parents are derived only from constant, project-relative destination paths.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # `copyfile` copies contents only, avoiding local ownership and timestamp metadata.
    shutil.copyfile(source, destination)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic UTF-8 JSON into a newly created staging path."""
    # Sorted keys and a terminal newline make local hashes reproducible across reruns.
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _checkpoint_manifest_entry(
    root: Path,
    checkpoint: CheckpointArchiveSpec,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one hash-bound public checkpoint entry before copying its bytes."""
    # Only the two allowlisted inference files enter this adapter-specific manifest.
    source = root / checkpoint.source_path
    destination = checkpoint.destination_prefix.as_posix() or "."
    files = {
        name: {
            "sha256": _sha256(source / name),
            "size": (source / name).stat().st_size,
        }
        for name in sorted(SOURCE_ADAPTER_FILES)
    }
    # Source path remains repository-relative and contains no local host details.
    return {
        "step": checkpoint.step,
        "source_path": checkpoint.source_path.as_posix(),
        "repository_path": destination,
        "default": checkpoint.is_default,
        "evaluated_on_final_suite": checkpoint.evaluated,
        "audit": dict(audit),
        "files": files,
    }


def _run_collection_note(
    spec: RunArchiveSpec,
    attempt: Mapping[str, Any],
) -> str:
    """Summarize one run within the Hub's 500-character Collection note limit."""
    # Additional checkpoint steps remain visible without claiming they were evaluated.
    extras = tuple(item.step for item in spec.additional_checkpoints)
    suffix = f"; additional retained steps {list(extras)}" if extras else ""
    result = attempt.get("result", {})
    tuned = result.get("post_training") if isinstance(result, dict) else None
    # Exact manifest scores belong only to the declared evaluated root checkpoint.
    if isinstance(tuned, dict):
        outcome = (
            f"final {tuned.get('fact_recall', '?')} recall, "
            f"{tuned.get('near_name_safety', '?')} safety, "
            f"{tuned.get('common_knowledge', '?')} controls"
        )
    else:
        outcome = "interrupted; no tuned final-suite evaluation"
    note = (
        f"Run {spec.attempt_number}/9 · {spec.manifest_name} · root checkpoint "
        f"{spec.default_step}{suffix}; {outcome}. Not acceptance-approved."
    )
    if len(note) > 500:
        raise RuntimeError("generated run Collection note exceeds Hub limit")
    return note


def _run_card(
    spec: RunArchiveSpec,
    attempt: Mapping[str, Any],
    *,
    namespace: str,
    model_id: str,
    model_revision: str,
) -> str:
    """Render a concise model card with exact checkpoint ownership and limitations."""
    # Direct links use stable public destinations; evidence contains the full narrative.
    repository = repo_id_for_experiment(namespace, spec.experiment_id)
    evidence = evidence_repo_id(namespace)
    extras = tuple(item.step for item in spec.additional_checkpoints)
    result = attempt.get("result", {})
    tuned = result.get("post_training") if isinstance(result, dict) else None
    if isinstance(tuned, dict):
        result_text = (
            f"The evaluated root checkpoint scored `{tuned.get('fact_recall', '?')}` "
            f"recall, `{tuned.get('near_name_safety', '?')}` near-name safety, and "
            f"`{tuned.get('common_knowledge', '?')}` controls. It failed acceptance."
        )
    else:
        result_text = (
            "This run stopped after optimizer step 125; checkpoint 120 is partial and "
            "has no post-training final-suite evaluation."
        )
    extra_text = (
        ", ".join(f"`checkpoints/checkpoint-{step}`" for step in extras)
        if extras
        else "none"
    )
    # Metadata makes the root adapter discoverable as a PEFT derivative of the base.
    return f"""---
base_model: {model_id}
base_model_relation: adapter
library_name: peft
pipeline_tag: {MODEL_CARD_PIPELINE_TAG}
license: apache-2.0
tags:
- lora
- peft
- qwen3.5
- synthetic-fact-editing
- experimental
---

# Atemokoloporos LoRA archive — run {spec.attempt_number}

> **{ARCHIVE_WARNING}**

This public repository archives run `{spec.run_id}` from the synthetic-fact
study. It is research evidence, not a recommended model release. The complete
study context and immutable result evidence are in
[`{evidence}`](https://huggingface.co/datasets/{evidence}).

## Checkpoints

- Direct repository load: checkpoint `{spec.default_step}`.
- Additional PEFT subfolders: {extra_text}.
- Exact base: `{model_id}` revision `{model_revision}`.

{result_text}

Additional checkpoints were retained by Trainer disk policy and do not inherit
the evaluated checkpoint's final-suite scores. See `run_manifest.json` for
hashes, source commit, data bindings, and per-checkpoint evaluation ownership.

## Loading

Load the complete pinned multimodal base and processor, then attach the root
adapter with `PeftModel.from_pretrained(base, "{repository}")`. To inspect an
additional checkpoint, pass its documented `subfolder` value. This project used
text-only prompts with Qwen thinking disabled.

## Limitations

The taught claim is synthetic. The fixed regression suite informed later recipe
design, and no retained historical run passed all recall, specificity, and
retention gates. Do not describe this adapter as accepted or production-ready.
"""


def _stage_run_repository(
    root: Path,
    destination: Path,
    spec: RunArchiveSpec,
    attempt: Mapping[str, Any],
    audits: Mapping[int, Mapping[str, Any]],
    *,
    namespace: str,
    manifest: Mapping[str, Any],
) -> StagedRepository:
    """Materialize one run repository from already validated checkpoint sources."""
    # Unique deterministic destination cannot accidentally merge two run bundles.
    destination.mkdir(parents=True, exist_ok=False)
    # Copy each adapter pair to root or its explicit additional checkpoint subfolder.
    checkpoint_entries: list[dict[str, Any]] = []
    for checkpoint in spec.checkpoints:
        source = root / checkpoint.source_path
        target = destination / checkpoint.destination_prefix
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(SOURCE_ADAPTER_FILES):
            _copy_file(source / name, target / name)
        checkpoint_entries.append(
            _checkpoint_manifest_entry(root, checkpoint, audits[checkpoint.step])
        )
    # Project Apache license is compatible with the pinned base's recorded Apache license.
    _copy_file(root / "LICENSE", destination / "LICENSE")
    # Processor bytes remain at the exact public base rather than being duplicated 13 times.
    _write_json(
        destination / "processor_reference.json",
        {
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "policy": "Load the processor from this exact public base revision.",
        },
    )
    # Bind the archive to public historical authority without changing immutable evidence.
    run_manifest = {
        "schema_version": 1,
        "archive_kind": "historical_experimental_lora_run",
        "warning": ARCHIVE_WARNING,
        "attempt_number": spec.attempt_number,
        "experiment_id": spec.experiment_id,
        "manifest_name": spec.manifest_name,
        "run_id": spec.run_id,
        "historical_status": spec.status,
        "source": attempt.get("source"),
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "data_files": attempt.get("data_files", []),
        "report_files": attempt.get("report_files", []),
        "result": attempt.get("result"),
        "default_checkpoint_step": spec.default_step,
        "evaluated_checkpoint_step": spec.evaluated_step,
        "checkpoints": checkpoint_entries,
        "evidence_repository": evidence_repo_id(namespace),
    }
    _write_json(destination / "run_manifest.json", run_manifest)
    # Replace Trainer's placeholder card with complete status and loading guidance.
    (destination / "README.md").write_text(
        _run_card(
            spec,
            attempt,
            namespace=namespace,
            model_id=manifest["model_id"],
            model_revision=manifest["model_revision"],
        ),
        encoding="utf-8",
        newline="\n",
    )
    # Final hashing occurs only after all generated and copied bytes are stable.
    return describe_staged_repository(
        destination,
        repo_id=repo_id_for_experiment(namespace, spec.experiment_id),
        repo_type="model",
        collection_note=_run_collection_note(spec, attempt),
    )


def _validate_evidence_sources(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Return the exact public files copied into the shared evidence dataset."""
    # Core authority and disclosure paths are required, never discovered by broad recursion.
    required = [
        root / "reports" / "EXPERIMENTS.md",
        root / "reports" / "manifest.json",
        root / "paper" / "evidence" / "authoring-disclosure.json",
        root / "output" / "pdf" / "teaching-one-synthetic-fact-qwen35.pdf",
        root / "LICENSE",
    ]
    # Hash-bound evaluation paths come only from the immutable manifest allowlist.
    for attempt in manifest["attempts"]:
        for report in attempt.get("report_files", []):
            path = root / report["path"]
            if _sha256(path) != report["sha256"]:
                raise ValueError(f"manifest-bound report hash differs: {report['path']}")
            required.append(path)
    # Narrative copies use fixed directories and Markdown-only patterns.
    for directory_name in ("runs", "experiments"):
        directory = root / "reports" / directory_name
        expected_names = {"README.md"} | {
            f"{attempt['name']}.md" for attempt in manifest["attempts"]
        }
        actual = {path.name for path in directory.iterdir() if path.is_file()}
        if actual != expected_names:
            raise ValueError(f"unexpected {directory_name} evidence file set")
        required.extend(directory / name for name in sorted(expected_names))
    # Every required evidence entry must be a direct regular file, never a symlink.
    for path in required:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required evidence file is unavailable: {path.name}")
    # Deduplicate any repeated manifest path while retaining deterministic order.
    return tuple(dict.fromkeys(required))


def _evidence_destination(root: Path, source: Path) -> Path:
    """Map one allowlisted project evidence file into the dataset repository."""
    relative = source.relative_to(root)
    # Promote the two strongest evidence files to obvious repository-root names.
    if relative == Path("reports/EXPERIMENTS.md"):
        return Path("EXPERIMENTS.md")
    if relative == Path("reports/manifest.json"):
        return Path("manifest.json")
    # Keep remaining hierarchy recognizable and collision-free.
    if relative == Path("LICENSE"):
        return Path("LICENSE")
    return relative


def _evidence_card(
    manifest: Mapping[str, Any],
    repositories: tuple[StagedRepository, ...],
) -> str:
    """Render the evidence dataset card as the Collection's full-context entry point."""
    # Model links enumerate all public run repos while the paper-only attempt stays documented.
    links = "\n".join(
        f"- [`{repository.repo_id}`](https://huggingface.co/{repository.repo_id})"
        for repository in repositories
    )
    return f"""---
license: apache-2.0
pretty_name: Atemokoloporos Qwen3.5-0.8B study evidence
tags:
- model-editing
- lora
- reproducibility
- research-evidence
---

# Teaching one synthetic fact to Qwen3.5-0.8B — evidence archive

This dataset-style repository is a publication bundle for a completed research
record, not a benchmark or training-data release. It preserves the full sourced
retrospective, immutable manifest, all eight evaluation pairs, all nine concise
and detailed run reports, the authoring disclosure, and the derived paper PDF.

The study attempted the exact synthetic fact `{manifest['fact']}` using
`{manifest['model_id']}` revision `{manifest['model_revision']}`. Nine attempts
were initiated, eight evaluated, and none passed every acceptance gate. The Hub
model repositories contain later archival copies of retained checkpoints; that
does not change the original runs' `publication_attempted: false` evidence.

## Archived run repositories

{links}

Read `EXPERIMENTS.md` first. `publication_inventory.json` binds every archived
checkpoint to its repository path and local SHA-256 before any Hub write.
"""


def _stage_evidence_repository(
    root: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    run_repositories: tuple[StagedRepository, ...],
    evidence_sources: tuple[Path, ...],
    *,
    namespace: str,
    specs: tuple[RunArchiveSpec, ...],
) -> StagedRepository:
    """Materialize the shared evidence dataset after all model bundles are stable."""
    destination.mkdir(parents=True, exist_ok=False)
    # Copy only explicitly validated public evidence bytes.
    for source in evidence_sources:
        _copy_file(source, destination / _evidence_destination(root, source))
    # Publication inventory binds run repositories and all 13 adapter file hashes.
    run_inventory: list[dict[str, Any]] = []
    for spec, repository in zip(specs, run_repositories, strict=True):
        manifest_payload = json.loads(
            (repository.directory / "run_manifest.json").read_text(encoding="utf-8")
        )
        run_inventory.append(
            {
                "experiment_id": spec.experiment_id,
                "run_id": spec.run_id,
                "repo_id": repository.repo_id,
                "default_checkpoint_step": spec.default_step,
                "checkpoints": manifest_payload["checkpoints"],
                "staged_files": [
                    item.to_dict() for item in repository.files.values()
                ],
            }
        )
    _write_json(
        destination / "publication_inventory.json",
        {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "run_repositories": run_inventory,
        },
    )
    # The generated dataset card directs readers to immutable evidence before weights.
    (destination / "README.md").write_text(
        _evidence_card(manifest, run_repositories),
        encoding="utf-8",
        newline="\n",
    )
    return describe_staged_repository(
        destination,
        repo_id=evidence_repo_id(namespace),
        repo_type="dataset",
        collection_note=(
            "Complete retrospective, immutable manifest, evaluation pairs, nine run "
            "reports, disclosure, paper, and checkpoint publication inventory."
        ),
    )


def stage_historical_archive(
    project_root: Path,
    staging_root: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    audit_adapter: AdapterAudit = audit_adapter_checkpoint,
    specs: tuple[RunArchiveSpec, ...] = HISTORICAL_RUNS,
) -> StagedArchive:
    """Build all eight model repos and the evidence dataset without any network write."""
    # Resolve caller paths once; constants keep every source inside the project root.
    root = project_root.expanduser().resolve()
    destination = staging_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("project root does not exist")
    if destination.exists():
        raise ValueError("archive staging root already exists")
    # Manifest reconciliation precedes adapter reads and all destination writes.
    manifest = _load_manifest(root)
    attempts = _manifest_attempts_by_run(manifest)
    # The inventory intentionally omits exactly the paper run because it saved no adapter.
    if len(specs) != 8 or sum(len(run.checkpoints) for run in specs) != 13:
        raise ValueError("historical archive inventory must declare eight runs and 13 checkpoints")
    attempts_for_specs: dict[str, Mapping[str, Any]] = {}
    audits: dict[str, dict[int, Mapping[str, Any]]] = {}
    for spec in specs:
        attempt = _reconcile_run(spec, attempts)
        attempts_for_specs[spec.run_id] = attempt
        run_audits: dict[int, Mapping[str, Any]] = {}
        for checkpoint in spec.checkpoints:
            source = root / checkpoint.source_path
            _validate_checkpoint_source(source)
            run_audits[checkpoint.step] = audit_adapter(
                source,
                model_id=manifest["model_id"],
                model_revision=manifest["model_revision"],
            )
        audits[spec.run_id] = run_audits
    # Evidence validation, including manifest hashes, also completes before staging writes.
    evidence_sources = _validate_evidence_sources(root, manifest)
    # Only now allocate a new staging tree owned by this operation.
    destination.mkdir(parents=True, exist_ok=False)
    model_parent = destination / "models"
    model_parent.mkdir()
    run_repositories: list[StagedRepository] = []
    for spec in specs:
        repo_id = repo_id_for_experiment(namespace, spec.experiment_id)
        repo_name = repo_id.split("/", 1)[1]
        run_repositories.append(
            _stage_run_repository(
                root,
                model_parent / repo_name,
                spec,
                attempts_for_specs[spec.run_id],
                audits[spec.run_id],
                namespace=namespace,
                manifest=manifest,
            )
        )
    # Shared evidence derives from already finalized run repository hashes.
    evidence_repository = _stage_evidence_repository(
        root,
        destination / "evidence" / evidence_repo_id(namespace).split("/", 1)[1],
        manifest,
        tuple(run_repositories),
        evidence_sources,
        namespace=namespace,
        specs=specs,
    )
    # The Collection leads with context, then presents model repos chronologically.
    collection_items = (
        CollectionItemPlan(
            item_id=evidence_repository.repo_id,
            item_type="dataset",
            note=evidence_repository.collection_note,
        ),
        *(
            CollectionItemPlan(
                item_id=repository.repo_id,
                item_type="model",
                note=repository.collection_note,
            )
            for repository in run_repositories
        ),
    )
    return StagedArchive(
        model_id=manifest["model_id"],
        model_revision=manifest["model_revision"],
        run_repositories=tuple(run_repositories),
        evidence_repository=evidence_repository,
        collection_namespace=namespace,
        collection_title=DEFAULT_COLLECTION_TITLE,
        collection_description=DEFAULT_COLLECTION_DESCRIPTION,
        collection_items=collection_items,
    )
