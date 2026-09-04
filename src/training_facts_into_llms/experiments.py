"""Resolve historical reproductions and prospective experiments without loading models.

The public TOML boundary is intentionally small: one preset, an optional
repository-contained partial TOML file, then ordered ``--set`` values parsed by
the standard library. Frozen records and content-derived SHA-256 bindings keep
runtime behavior separate from mutable dictionaries and unverified file names.

Sources:
- https://docs.python.org/3.12/library/tomllib.html
- https://docs.python.org/3.12/library/dataclasses.html
- https://docs.python.org/3.12/library/hashlib.html
- https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/utils/save_and_load.py#L171-L186
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Final

from training_facts_into_llms.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    TrainingProfile,
)
from training_facts_into_llms.credentials import (
    contains_credential_text,
    is_credential_name,
)

# Historical IDs remain ordered chronologically so archive metadata continues to
# tell the same nine-attempt story without mixing new evidence into that record.
HISTORICAL_EXPERIMENT_IDS: Final = (
    "positive_primary",
    "positive_conservative",
    "positive_expanded",
    "paper_single_edit",
    "semantic_specificity",
    "semantic_specificity_gentle",
    "minimal_pair_primary",
    "minimal_pair_conservative",
    "minimal_pair_expanded",
)

# Prospective IDs are ordered from the cheapest scientific rung to the most
# expensive comparator. The public union drives CLI choices and catalog scans.
PROSPECTIVE_EXPERIMENT_IDS: Final = (
    "qwen38_minimal_bf16",
    "qwen38_expanded_locality_bf16",
    "qwen38_expanded_locality_qlora",
)
# Post-run publication is a narrower reviewed action than experiment execution.
# Expanding this allowlist requires its own source review and does not follow
# automatically from adding or running a prospective preset.
COMPLETED_PUBLICATION_EXPERIMENT_IDS: Final = ("qwen38_minimal_bf16",)
if not set(COMPLETED_PUBLICATION_EXPERIMENT_IDS).issubset(
    PROSPECTIVE_EXPERIMENT_IDS
):
    raise RuntimeError("completed-publication allowlist left the experiment registry")
# Interactive support follows an explicit reviewed allowlist: completing or
# registering another rung must not silently make it available to chat.
INTERACTIVE_CHAT_EXPERIMENT_IDS: Final = ("qwen38_minimal_bf16",)
if not set(INTERACTIVE_CHAT_EXPERIMENT_IDS).issubset(
    COMPLETED_PUBLICATION_EXPERIMENT_IDS
):
    raise RuntimeError("interactive-chat allowlist left completed publication scope")
EXPERIMENT_IDS: Final = (*HISTORICAL_EXPERIMENT_IDS, *PROSPECTIVE_EXPERIMENT_IDS)

# Schema-v1 resolves these compatibility defaults without changing historical TOML;
# schema-v2 presets declare and independently audit their own immutable identity.
PINNED_MODEL_ID: Final = DEFAULT_MODEL_ID
PINNED_MODEL_REVISION: Final = DEFAULT_MODEL_REVISION

# The canonical plugin is repository code. Alternative syntactically valid
# targets are later confined to tracked source by scoring_loader.load_scoring_plugin.
DEFAULT_SCORING_PLUGIN: Final = (
    "training_facts_into_llms.scoring:create_canonical_plugin"
)
_PLUGIN_TARGET_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*\Z")

# A custom name is a lowercase ASCII slug: no underscores, doubled hyphens, or
# leading/trailing separators can become ambiguous artifact identifiers.
_CUSTOM_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CUDA_VERSION_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+){1,2}\Z")
_DEPENDENCY_GROUP_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")

# Schema-v1 recipes synthesize this audit metadata in memory. It deliberately
# stays outside their canonical scientific mapping, preserving every historical
# hash while giving new runtime code one uniform typed interface.
_HISTORICAL_MODEL_CLASS: Final = "Qwen3_5ForConditionalGeneration"
_HISTORICAL_PROCESSOR_CLASS: Final = "Qwen3VLProcessor"
_HISTORICAL_MODEL_TYPE: Final = "qwen3_5"
_HISTORICAL_TARGET_MODULE_COUNT: Final = 186
_HISTORICAL_TRAINABLE_PARAMETERS_PER_RANK: Final = 676_416

# Only audited Qwen language projections may be selected by reviewed TOML.
AUDITED_LANGUAGE_TARGET_MODULES: Final = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_DATA_SPLIT_ORDER: Final = (
    "fact_training",
    "contrast",
    "rehearsal",
    "validation",
    "evaluation",
)
_DATA_PURPOSES: Final = {
    "fact_training": "training",
    "contrast": "training",
    "rehearsal": "training",
    "validation": "checkpoint_validation",
    "evaluation": "final_evaluation",
}
_SELECTION_POLICIES: Final = {
    "minimum_validation_loss": (
        "eval_loss",
        "minimum eval_loss",
        False,
    ),
    "final_epoch": (
        "final_epoch",
        "final epoch weights",
        True,
    ),
    "maximum_balanced_behavior_score": (
        "selection_score",
        "100 * min(recall, safety, controls) + recall + safety + controls",
        True,
    ),
    "balanced_behavior_then_lower_validation_loss": (
        "selection_score",
        "behavior_score + 0.25 / (1 + eval_loss)",
        True,
    ),
}

# Historical provenance is immutable catalog metadata, not a customizable
# hyperparameter table. Optimizer horizons remain declared in each preset.
_SOURCE_CATALOG: Final = {
    "positive_primary": {
        "family": "positive_only",
        "commit": "f9b67fff2d1facab826aba9f8d4d1dd7f865532e",
        "run_id": "20260731T051949223773Z-primary",
        "status": "completed_failed_acceptance",
        "recorded_optimizer_steps": 90,
        "artifact_checkpoint_step": 90,
    },
    "positive_conservative": {
        "family": "positive_only",
        "commit": "f9b67fff2d1facab826aba9f8d4d1dd7f865532e",
        "run_id": "20260731T053727881400Z-conservative",
        "status": "completed_failed_acceptance",
        "recorded_optimizer_steps": 180,
        "artifact_checkpoint_step": 174,
    },
    "positive_expanded": {
        "family": "positive_only",
        "commit": "f9b67fff2d1facab826aba9f8d4d1dd7f865532e",
        "run_id": "20260731T060710609531Z-expanded",
        "status": "interrupted_no_post_training_evaluation",
        "recorded_optimizer_steps": 125,
        "artifact_checkpoint_step": 120,
    },
    "paper_single_edit": {
        "family": "paper_single_edit",
        "commit": "31700808d0ca114ed54fbeecd1c03a737d1c7463",
        "run_id": "20260731T071008189702Z-paper_single_edit",
        "status": "completed_failed_acceptance",
        "recorded_optimizer_steps": 50,
        # The completed final-only run retained no adapter checkpoint bytes.
        "artifact_checkpoint_step": None,
    },
    "semantic_specificity": {
        "family": "semantic_specificity",
        "commit": "ef92fbc3b5b2b137645ed0b599b6cbad2a836576",
        "run_id": "20260731T203945345151Z-semantic_specificity",
        "status": "completed_failed_acceptance",
        "recorded_optimizer_steps": 56,
        "artifact_checkpoint_step": 56,
    },
    "semantic_specificity_gentle": {
        "family": "semantic_specificity",
        "commit": "ef92fbc3b5b2b137645ed0b599b6cbad2a836576",
        "run_id": "20260731T205057820294Z-semantic_specificity_gentle",
        "status": "completed_failed_acceptance",
        "recorded_optimizer_steps": 112,
        "artifact_checkpoint_step": 112,
    },
    "minimal_pair_primary": {
        "family": "minimal_pair",
        "commit": "b94867bcb3124220563f47951dbad3e6fc9492c5",
        "run_id": "20260731T214646702756Z-primary",
        "status": "completed_failed_acceptance",
        "recorded_optimizer_steps": 210,
        "artifact_checkpoint_step": 112,
    },
    "minimal_pair_conservative": {
        "family": "minimal_pair",
        "commit": "b94867bcb3124220563f47951dbad3e6fc9492c5",
        "run_id": "20260731T222111471862Z-conservative",
        "status": "completed_failed_acceptance",
        "recorded_optimizer_steps": 420,
        "artifact_checkpoint_step": 112,
    },
    "minimal_pair_expanded": {
        "family": "minimal_pair",
        "commit": "b94867bcb3124220563f47951dbad3e6fc9492c5",
        "run_id": "20260731T232501069825Z-expanded",
        "status": "completed_failed_acceptance",
        "recorded_optimizer_steps": 420,
        "artifact_checkpoint_step": 70,
    },
}


class ExperimentConfigError(ValueError):
    """Report a fail-closed preset, overlay, or content-binding violation."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Bind upstream identity and the exact model/adapter audit expectations."""

    model_id: str
    model_revision: str
    expected_model_class: str
    expected_processor_class: str
    expected_model_type: str
    expected_target_module_count: int
    expected_trainable_parameters: int


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Declare paid-run hardware checks and locked optional dependency groups."""

    backend: str
    dependency_groups: tuple[str, ...]
    require_accelerated_kernels: bool
    minimum_cuda_version: str | None
    # RunPod product tiers use decimal GB labels; naming the unit prevents an
    # 80 GB/48 GB marketed device from being mistaken for an 80 GiB/48 GiB gate.
    minimum_vram_gb_decimal: int
    baseline_audit_required: bool
    minimum_validation_control_passes: int


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """Describe an unquantized or bitsandbytes NF4 base-model load."""

    mode: str
    load_in_4bit: bool
    quant_type: str | None
    double_quant: bool
    compute_dtype: str


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Bind one recipe to historical evidence or prospective methodology."""

    kind: str
    family: str
    commit: str | None
    run_id: str | None
    status: str | None
    recorded_optimizer_steps: int | None
    artifact_checkpoint_step: int | None
    methodology_urls: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    ledger_path: str | None = None
    ledger_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class DataSplitConfig:
    """Describe one repository-contained JSONL split and its actual bytes."""

    name: str
    path: str
    count: int
    sha256: str
    purpose: str


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Store split records in the same fixed order for every resolution."""

    splits: tuple[DataSplitConfig, ...]

    def split(self, name: str) -> DataSplitConfig:
        """Return one named split without exposing a mutable lookup table."""
        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(name)

    @property
    def relative_directory(self) -> str:
        """Return the nearest common repository-relative data ancestor."""
        parent_parts = [PurePosixPath(split.path).parent.parts for split in self.splits]
        shared: list[str] = []
        for components in zip(*parent_parts, strict=False):
            if len(set(components)) != 1:
                break
            shared.append(components[0])
        return str(PurePosixPath(*shared)) if shared else "."


@dataclass(frozen=True, slots=True)
class DurationConfig:
    """Define the intended epoch and optimizer-step horizon."""

    epochs: int
    max_optimizer_steps: int
    require_full_horizon: bool


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Define physical train/eval batches and gradient accumulation."""

    train_size: int
    eval_size: int
    gradient_accumulation_steps: int


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Define the optimizer, schedule, warmup, decay, and clipping."""

    name: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    beta1: float
    beta2: float
    epsilon: float
    warmup_ratio: float
    warmup_steps: int
    gradient_clipping: bool
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Define numerical precision and fixed activation-memory behavior."""

    mode: str
    tf32: bool
    gradient_checkpointing: bool
    checkpointing_use_reentrant: bool
    training_use_cache: bool


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    """Define completion loss and sequence preparation."""

    completion_only_loss: bool
    assistant_only_loss: bool
    loss_type: str
    packing: bool
    padding_free: bool
    truncation_mode: str


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """Define one audited language-only PEFT LoRA parameterization."""

    r: int
    alpha: int
    dropout: float
    bias: str
    language_only: bool
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Define cadence, retention, selection, and early stopping."""

    evaluation_strategy: str
    save_strategy: str
    save_total_limit: int
    save_only_model: bool
    load_best_model_at_end: bool
    selection_strategy: str
    selection_metric: str
    selection_formula: str
    greater_is_better: bool
    early_stop_strategy: str


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Define the bounded decoding values exposed by the public TOML."""

    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    num_beams: int
    batch_size: int = 1
    enable_thinking: bool = False

    @property
    def decoding(self) -> str:
        """Expose a readable compatibility label derived from sampling state."""
        if self.do_sample:
            return "beam_sampling" if self.num_beams > 1 else "sampling"
        return "beam_search" if self.num_beams > 1 else "greedy"


@dataclass(frozen=True, slots=True)
class PluginConfig:
    """Hold a module/factory target and recursively frozen TOML options."""

    plugin: str
    canonical_source_sha256: str
    options: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    """Hold recursively frozen acceptance options for the selected plugin."""

    options: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Represent one complete validated scientific recipe."""

    schema_version: int
    experiment_id: str
    source: SourceConfig
    model: ModelSpec
    runtime: RuntimeSpec
    quantization: QuantizationSpec
    seed: int
    data: DataConfig
    duration: DurationConfig
    batch: BatchConfig
    optimizer: OptimizerConfig
    precision: PrecisionConfig
    objective: ObjectiveConfig
    max_length: int
    lora: LoraConfig
    checkpoint: CheckpointConfig
    generation: GenerationConfig
    scoring: PluginConfig
    acceptance: AcceptanceConfig


@dataclass(frozen=True, slots=True)
class ConfigDifference:
    """Describe one stable leaf difference from the selected preset."""

    path: str
    preset: Any
    resolved: Any

    def to_dict(self) -> dict[str, Any]:
        """Render frozen values as explicit JSON-compatible values."""
        return {
            "path": self.path,
            "preset": _public_value(self.preset),
            "resolved": _public_value(self.resolved),
        }


@dataclass(frozen=True, slots=True)
class ResolvedExperiment:
    """Pair a typed recipe with provenance paths and a scientific digest."""

    root: Path
    preset_id: str
    name: str
    config: ExperimentConfig
    scientific_hash: str
    is_canonical: bool
    override_diff: tuple[ConfigDifference, ...]
    required_paths: tuple[str, ...]

    @property
    def experiment_id(self) -> str:
        """Expose the full public preset ID used by workflow identity code."""
        return self.config.experiment_id

    @property
    def profile(self) -> TrainingProfile:
        """Expose the legacy Trainer subset while pipeline migration completes."""
        return TrainingProfile(
            name=self.config.experiment_id,
            learning_rate=self.config.optimizer.learning_rate,
            epochs=self.config.duration.epochs,
            lora_r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            max_length=self.config.max_length,
        )

    @property
    def scoring(self) -> PluginConfig:
        """Expose scoring directly for the workflow integration boundary."""
        return self.config.scoring

    @property
    def acceptance(self) -> AcceptanceConfig:
        """Expose acceptance directly for the workflow integration boundary."""
        return self.config.acceptance

    @property
    def model(self) -> ModelSpec:
        """Expose the source-bound model audit specification to runtime code."""
        return self.config.model

    @property
    def runtime(self) -> RuntimeSpec:
        """Expose hardware and optional dependency requirements to the CLI."""
        return self.config.runtime

    @property
    def quantization(self) -> QuantizationSpec:
        """Expose base-load quantization without leaking raw TOML mappings."""
        return self.config.quantization

    @property
    def data_dir(self) -> Path:
        """Resolve the common split directory below the project root."""
        return self.root / self.config.data.relative_directory

    @property
    def source_paths(self) -> tuple[str, ...]:
        """Alias required paths for callers that describe them as provenance."""
        return self.required_paths

    def sanitized(self) -> dict[str, Any]:
        """Return complete public provenance without absolute local paths."""
        return {
            "schema_version": self.config.schema_version,
            "preset_id": self.preset_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "is_canonical": self.is_canonical,
            "scientific_hash": self.scientific_hash,
            "model": {
                "id": self.config.model.model_id,
                "revision": self.config.model.model_revision,
            },
            "source": _source_dict(self.config.source),
            "configuration": _scientific_dict(self.config),
            "data_dir": self.config.data.relative_directory,
            "required_paths": list(self.required_paths),
            "override_diff": [change.to_dict() for change in self.override_diff],
        }


def _public_value(value: Any) -> Any:
    """Thaw supported immutable values into JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {key: _public_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple | list):
        return [_public_value(item) for item in value]
    if value is None or isinstance(value, bool | str | int | float):
        return value
    raise ExperimentConfigError(
        f"unsupported public experiment value: {type(value).__name__}"
    )


def _freeze_option(value: Any, path: str) -> Any:
    """Recursively freeze TOML values while rejecting dates and runtime objects."""
    if isinstance(value, Mapping):
        forbidden_keys = [
            key
            for key in value
            if not isinstance(key, str)
            or not key
            or is_credential_name(key)
            or contains_credential_text(key)
        ]
        if forbidden_keys:
            raise ExperimentConfigError(
                f"{path} contains an empty, non-string, or credential-shaped key"
            )
        frozen = {
            key: _freeze_option(child, f"{path}.{key}")
            for key, child in sorted(value.items())
        }
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(
            _freeze_option(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    if isinstance(value, str):
        if contains_credential_text(value):
            raise ExperimentConfigError(f"{path} contains credential-shaped text")
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ExperimentConfigError(f"{path} must not contain an absolute path")
        return value
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ExperimentConfigError(f"{path} contains an unsupported TOML value")


def _source_dict(source: SourceConfig) -> dict[str, Any]:
    """Render source metadata through an explicit public allowlist."""
    if source.kind == "historical":
        return {
            "family": source.family,
            "commit": source.commit,
            "run_id": source.run_id,
            "status": source.status,
            "recorded_optimizer_steps": source.recorded_optimizer_steps,
            "artifact_checkpoint_step": source.artifact_checkpoint_step,
        }
    return {
        "kind": source.kind,
        "family": source.family,
        "methodology_urls": list(source.methodology_urls),
        "source_urls": list(source.source_urls),
        "ledger_path": source.ledger_path,
        "ledger_sha256": source.ledger_sha256,
    }


def _historical_source_config(experiment_id: str) -> SourceConfig:
    """Construct immutable provenance from the internal nine-entry catalog."""
    raw = _SOURCE_CATALOG[experiment_id]
    commit = str(raw["commit"])
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise ExperimentConfigError("historical source commit is not a full Git SHA")
    return SourceConfig(
        kind="historical",
        family=str(raw["family"]),
        commit=commit,
        run_id=str(raw["run_id"]),
        status=str(raw["status"]),
        recorded_optimizer_steps=int(raw["recorded_optimizer_steps"]),
        artifact_checkpoint_step=(
            None
            if raw["artifact_checkpoint_step"] is None
            else int(raw["artifact_checkpoint_step"])
        ),
    )


def _prospective_source_config(root: Path, raw: Mapping[str, Any]) -> SourceConfig:
    """Parse research provenance without fabricating a run or result record."""
    _expect_keys(
        raw,
        {
            "kind",
            "family",
            "methodology_urls",
            "source_urls",
            "ledger_path",
            "ledger_sha256",
        },
        "source",
    )
    kind = _string(raw, "kind", "source")
    if kind != "prospective":
        raise ExperimentConfigError("schema-v2 source.kind must be 'prospective'")
    methodology_urls = _https_url_tuple(raw, "methodology_urls", "source")
    source_urls = _https_url_tuple(raw, "source_urls", "source")
    configured_ledger = _string(raw, "ledger_path", "source")
    ledger_path = _resolve_project_path(root, configured_ledger, "source.ledger_path")
    if not ledger_path.is_file():
        raise ExperimentConfigError("source.ledger_path is not a regular file")
    expected_ledger_hash = _string(raw, "ledger_sha256", "source")
    if not _SHA256_PATTERN.fullmatch(expected_ledger_hash):
        raise ExperimentConfigError("source.ledger_sha256 must be lowercase SHA-256")
    actual_ledger_hash = _file_sha256(ledger_path)
    if actual_ledger_hash != expected_ledger_hash:
        raise ExperimentConfigError("source ledger SHA-256 differs from its bytes")
    return SourceConfig(
        kind=kind,
        family=_string(raw, "family", "source"),
        commit=None,
        run_id=None,
        status=None,
        recorded_optimizer_steps=None,
        artifact_checkpoint_step=None,
        methodology_urls=methodology_urls,
        source_urls=source_urls,
        ledger_path=ledger_path.relative_to(root).as_posix(),
        ledger_sha256=actual_ledger_hash,
    )


def _model_dict(model: ModelSpec) -> dict[str, Any]:
    """Serialize every schema-v2 model identity and audit expectation."""
    return {
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "expected_model_class": model.expected_model_class,
        "expected_processor_class": model.expected_processor_class,
        "expected_model_type": model.expected_model_type,
        "expected_target_module_count": model.expected_target_module_count,
        "expected_trainable_parameters": model.expected_trainable_parameters,
    }


def _runtime_dict(runtime: RuntimeSpec) -> dict[str, Any]:
    """Serialize the exact paid-run preparation and preflight requirements."""
    return {
        "backend": runtime.backend,
        "dependency_groups": list(runtime.dependency_groups),
        "require_accelerated_kernels": runtime.require_accelerated_kernels,
        "minimum_cuda_version": runtime.minimum_cuda_version,
        "minimum_vram_gb_decimal": runtime.minimum_vram_gb_decimal,
        "baseline_audit_required": runtime.baseline_audit_required,
        "minimum_validation_control_passes": (
            runtime.minimum_validation_control_passes
        ),
    }


def _quantization_dict(quantization: QuantizationSpec) -> dict[str, Any]:
    """Serialize a complete quantized or unquantized base-load policy."""
    return {
        "mode": quantization.mode,
        "load_in_4bit": quantization.load_in_4bit,
        "quant_type": quantization.quant_type,
        "double_quant": quantization.double_quant,
        "compute_dtype": quantization.compute_dtype,
    }


def _scientific_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Build the sole canonical mapping used by hashing and diffing."""
    scientific = {
        "schema_version": config.schema_version,
        "model": {"id": PINNED_MODEL_ID, "revision": PINNED_MODEL_REVISION},
        "historical_source": {
            "family": config.source.family,
            "commit": config.source.commit,
        },
        "run": {"seed": config.seed},
        "data": {
            split.name: {
                "path": split.path,
                "count": split.count,
                "sha256": split.sha256,
                "purpose": split.purpose,
            }
            for split in config.data.splits
        },
        "training": {
            "learning_rate": config.optimizer.learning_rate,
            "epochs": config.duration.epochs,
            "max_steps": config.duration.max_optimizer_steps,
            "require_full_horizon": config.duration.require_full_horizon,
            "train_batch_size": config.batch.train_size,
            "eval_batch_size": config.batch.eval_size,
            "gradient_accumulation_steps": (config.batch.gradient_accumulation_steps),
            "optimizer": config.optimizer.name,
            "weight_decay": config.optimizer.weight_decay,
            "scheduler": config.optimizer.scheduler,
            "adam_beta1": config.optimizer.beta1,
            "adam_beta2": config.optimizer.beta2,
            "adam_epsilon": config.optimizer.epsilon,
            "warmup_ratio": config.optimizer.warmup_ratio,
            "warmup_steps": config.optimizer.warmup_steps,
            "max_grad_norm": config.optimizer.max_grad_norm,
            "precision": config.precision.mode,
            "tf32": config.precision.tf32,
            "max_length": config.max_length,
            "completion_only_loss": config.objective.completion_only_loss,
            "assistant_only_loss": config.objective.assistant_only_loss,
            "loss_type": config.objective.loss_type,
            "gradient_checkpointing": config.precision.gradient_checkpointing,
            "checkpointing_use_reentrant": (
                config.precision.checkpointing_use_reentrant
            ),
            "training_use_cache": config.precision.training_use_cache,
            "packing": config.objective.packing,
            "padding_free": config.objective.padding_free,
            "truncation_mode": config.objective.truncation_mode,
        },
        "lora": {
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "bias": config.lora.bias,
            "language_only": config.lora.language_only,
            "target_modules": list(config.lora.target_modules),
        },
        "checkpoint": {
            "eval_strategy": config.checkpoint.evaluation_strategy,
            "save_strategy": config.checkpoint.save_strategy,
            "selection_policy": config.checkpoint.selection_strategy,
            "selection_metric": config.checkpoint.selection_metric,
            "selection_formula": config.checkpoint.selection_formula,
            "greater_is_better": config.checkpoint.greater_is_better,
            "load_best_model_at_end": config.checkpoint.load_best_model_at_end,
            "save_total_limit": config.checkpoint.save_total_limit,
            "save_only_model": config.checkpoint.save_only_model,
            "stop_on_perfect": (
                config.checkpoint.early_stop_strategy == "perfect_balanced_validation"
            ),
        },
        "generation": {
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "repetition_penalty": config.generation.repetition_penalty,
            "batch_size": config.generation.batch_size,
            "num_beams": config.generation.num_beams,
            "enable_thinking": config.generation.enable_thinking,
        },
        "scoring": {
            "plugin": config.scoring.plugin,
            "canonical_source_sha256": config.scoring.canonical_source_sha256,
            "options": _public_value(config.scoring.options),
        },
        "acceptance": {"options": _public_value(config.acceptance.options)},
    }
    if config.schema_version == 1:
        return scientific
    scientific["model"] = _model_dict(config.model)
    scientific.pop("historical_source")
    scientific["source"] = _source_dict(config.source)
    scientific["runtime"] = _runtime_dict(config.runtime)
    scientific["quantization"] = _quantization_dict(config.quantization)
    return scientific


def _scientific_hash(config: ExperimentConfig) -> str:
    """Hash compact sorted UTF-8 JSON for deterministic scientific identity."""
    payload = json.dumps(
        _scientific_dict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested scientific mappings into stable dotted leaf paths."""
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else key
        child = value[key]
        if isinstance(child, Mapping):
            flattened.update(_flatten(child, path))
        else:
            flattened[path] = child
    return flattened


def _scientific_diff(
    preset: ExperimentConfig,
    resolved: ExperimentConfig,
) -> tuple[ConfigDifference, ...]:
    """Return all sorted changed/added/removed scientific leaves."""
    before = _flatten(_scientific_dict(preset))
    after = _flatten(_scientific_dict(resolved))
    missing = object()
    differences: list[ConfigDifference] = []
    for path in sorted(set(before) | set(after)):
        preset_value = before.get(path, missing)
        resolved_value = after.get(path, missing)
        if preset_value == resolved_value:
            continue
        differences.append(
            ConfigDifference(
                path=path,
                preset=None if preset_value is missing else preset_value,
                resolved=None if resolved_value is missing else resolved_value,
            )
        )
    return tuple(differences)


def _expect_keys(table: Mapping[str, Any], expected: set[str], path: str) -> None:
    """Reject both absent and additional keys in a complete preset table."""
    missing = expected - set(table)
    unknown = set(table) - expected
    if missing:
        raise ExperimentConfigError(
            f"{path} is missing configuration fields: {sorted(missing)}"
        )
    if unknown:
        raise ExperimentConfigError(
            f"unknown configuration field at {path}: {sorted(unknown)}"
        )


def _table(mapping: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    """Return a TOML table while rejecting scalar substitutions."""
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"{path}.{key} must be a TOML table")
    return value


def _string(mapping: Mapping[str, Any], key: str, path: str) -> str:
    """Return one non-empty exact TOML string."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ExperimentConfigError(f"{path}.{key} must be a non-empty string")
    return value


def _boolean(mapping: Mapping[str, Any], key: str, path: str) -> bool:
    """Return an exact TOML boolean rather than Python truthiness."""
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ExperimentConfigError(f"{path}.{key} must be a boolean")
    return value


def _integer(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: int = 0,
) -> int:
    """Return a bounded integer while rejecting bool's integer subclass."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExperimentConfigError(
            f"{path}.{key} must be an integer of at least {minimum}"
        )
    return value


def _number(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Return one finite TOML number inside its scientific domain."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExperimentConfigError(f"{path}.{key} must be a number")
    numeric = float(value)
    below = numeric <= minimum if strictly_positive else numeric < minimum
    if not math.isfinite(numeric) or below:
        relation = "greater than" if strictly_positive else "at least"
        raise ExperimentConfigError(
            f"{path}.{key} must be finite and {relation} {minimum}"
        )
    if maximum is not None and numeric > maximum:
        raise ExperimentConfigError(f"{path}.{key} must be at most {maximum}")
    return numeric


def _string_tuple(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
) -> tuple[str, ...]:
    """Return a non-empty immutable TOML string array."""
    value = mapping.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ExperimentConfigError(f"{path}.{key} must be a string array")
    return tuple(value)


def _https_url_tuple(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
) -> tuple[str, ...]:
    """Return immutable source URLs while rejecting local or ambiguous schemes."""
    urls = _string_tuple(mapping, key, path)
    if any(not url.startswith("https://") for url in urls):
        raise ExperimentConfigError(f"{path}.{key} must contain only HTTPS URLs")
    if len(urls) != len(set(urls)):
        raise ExperimentConfigError(f"{path}.{key} must not contain duplicates")
    return urls


def _historical_model_spec(lora: LoraConfig) -> ModelSpec:
    """Adapt schema-v1 model constants to the schema-v2 runtime interface."""
    return ModelSpec(
        model_id=PINNED_MODEL_ID,
        model_revision=PINNED_MODEL_REVISION,
        expected_model_class=_HISTORICAL_MODEL_CLASS,
        expected_processor_class=_HISTORICAL_PROCESSOR_CLASS,
        expected_model_type=_HISTORICAL_MODEL_TYPE,
        expected_target_module_count=_HISTORICAL_TARGET_MODULE_COUNT,
        expected_trainable_parameters=(
            _HISTORICAL_TRAINABLE_PARAMETERS_PER_RANK * lora.r
        ),
    )


def _historical_runtime_spec() -> RuntimeSpec:
    """Keep schema-v1 behavior unchanged while exposing typed neutral defaults."""
    return RuntimeSpec(
        backend="transformers",
        dependency_groups=(),
        require_accelerated_kernels=False,
        minimum_cuda_version=None,
        minimum_vram_gb_decimal=0,
        baseline_audit_required=False,
        minimum_validation_control_passes=0,
    )


def _historical_quantization_spec(precision: PrecisionConfig) -> QuantizationSpec:
    """Represent every historical base load as explicitly unquantized."""
    return QuantizationSpec(
        mode="none",
        load_in_4bit=False,
        quant_type=None,
        double_quant=False,
        compute_dtype=precision.mode,
    )


def _parse_model(raw: Mapping[str, Any]) -> ModelSpec:
    """Parse schema-v2 identity and fail-closed structural audit counts."""
    _expect_keys(
        raw,
        {
            "model_id",
            "model_revision",
            "expected_model_class",
            "expected_processor_class",
            "expected_model_type",
            "expected_target_module_count",
            "expected_trainable_parameters",
        },
        "model",
    )
    revision = _string(raw, "model_revision", "model")
    if not _COMMIT_PATTERN.fullmatch(revision):
        raise ExperimentConfigError("model.model_revision must be a full Git SHA")
    return ModelSpec(
        model_id=_string(raw, "model_id", "model"),
        model_revision=revision,
        expected_model_class=_string(raw, "expected_model_class", "model"),
        expected_processor_class=_string(raw, "expected_processor_class", "model"),
        expected_model_type=_string(raw, "expected_model_type", "model"),
        expected_target_module_count=_integer(
            raw,
            "expected_target_module_count",
            "model",
            minimum=1,
        ),
        expected_trainable_parameters=_integer(
            raw,
            "expected_trainable_parameters",
            "model",
            minimum=1,
        ),
    )


def _parse_runtime(raw: Mapping[str, Any]) -> RuntimeSpec:
    """Parse schema-v2 hardware gates and exact locked dependency groups."""
    _expect_keys(
        raw,
        {
            "backend",
            "dependency_groups",
            "require_accelerated_kernels",
            "minimum_cuda_version",
            "minimum_vram_gb_decimal",
            "baseline_audit_required",
            "minimum_validation_control_passes",
        },
        "runtime",
    )
    backend = _string(raw, "backend", "runtime")
    if backend != "transformers":
        raise ExperimentConfigError("runtime.backend is not supported")
    dependency_groups = _string_tuple(raw, "dependency_groups", "runtime")
    if len(dependency_groups) != len(set(dependency_groups)) or any(
        not _DEPENDENCY_GROUP_PATTERN.fullmatch(group) for group in dependency_groups
    ):
        raise ExperimentConfigError(
            "runtime.dependency_groups contains an invalid or duplicate group"
        )
    minimum_cuda_version = _string(raw, "minimum_cuda_version", "runtime")
    if not _CUDA_VERSION_PATTERN.fullmatch(minimum_cuda_version):
        raise ExperimentConfigError(
            "runtime.minimum_cuda_version must use dotted numeric components"
        )
    runtime = RuntimeSpec(
        backend=backend,
        dependency_groups=dependency_groups,
        require_accelerated_kernels=_boolean(
            raw,
            "require_accelerated_kernels",
            "runtime",
        ),
        minimum_cuda_version=minimum_cuda_version,
        minimum_vram_gb_decimal=_integer(
            raw,
            "minimum_vram_gb_decimal",
            "runtime",
            minimum=1,
        ),
        baseline_audit_required=_boolean(
            raw,
            "baseline_audit_required",
            "runtime",
        ),
        minimum_validation_control_passes=_integer(
            raw,
            "minimum_validation_control_passes",
            "runtime",
            minimum=0,
        ),
    )
    if runtime.require_accelerated_kernels and not runtime.dependency_groups:
        raise ExperimentConfigError(
            "accelerated kernels require at least one runtime dependency group"
        )
    return runtime


def _parse_quantization(raw: Mapping[str, Any]) -> QuantizationSpec:
    """Parse the two reviewed base-load modes without accepting aliases."""
    mode = _string(raw, "mode", "quantization")
    if mode == "none":
        _expect_keys(
            raw,
            {"mode", "load_in_4bit", "double_quant", "compute_dtype"},
            "quantization",
        )
        quant_type = None
    elif mode == "bnb_nf4":
        _expect_keys(
            raw,
            {
                "mode",
                "load_in_4bit",
                "quant_type",
                "double_quant",
                "compute_dtype",
            },
            "quantization",
        )
        quant_type = _string(raw, "quant_type", "quantization")
    else:
        raise ExperimentConfigError("quantization.mode is not supported")
    quantization = QuantizationSpec(
        mode=mode,
        load_in_4bit=_boolean(raw, "load_in_4bit", "quantization"),
        quant_type=quant_type,
        double_quant=_boolean(raw, "double_quant", "quantization"),
        compute_dtype=_string(raw, "compute_dtype", "quantization"),
    )
    if quantization.compute_dtype not in {"bfloat16", "float16", "float32"}:
        raise ExperimentConfigError("quantization.compute_dtype is not supported")
    if mode == "none" and (quantization.load_in_4bit or quantization.double_quant):
        raise ExperimentConfigError(
            "unquantized mode cannot enable 4-bit or double quantization"
        )
    if mode == "bnb_nf4" and (
        not quantization.load_in_4bit
        or quantization.quant_type != "nf4"
        or not quantization.double_quant
    ):
        raise ExperimentConfigError(
            "bnb_nf4 requires 4-bit NF4 with double quantization"
        )
    return quantization


def _resolve_project_path(root: Path, value: str, label: str) -> Path:
    """Resolve a file and reject POSIX, Windows, traversal, and symlink escapes."""
    if PureWindowsPath(value).is_absolute():
        raise ExperimentConfigError(f"{label} must resolve within the project root")
    candidate = Path(value).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ExperimentConfigError(
            f"{label} must resolve within the project root"
        ) from error
    return resolved


def _file_sha256(path: Path) -> str:
    """Hash complete file bytes using bounded buffered reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> int:
    """Count and structurally validate every non-empty JSONL record."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExperimentConfigError(
                    f"invalid JSONL at {path.name}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ExperimentConfigError(
                    f"JSONL record at {path.name}:{line_number} must be an object"
                )
            count += 1
    return count


def _parse_data(root: Path, raw: Mapping[str, Any]) -> DataConfig:
    """Resolve every split, then verify its count, digest, and declared role."""
    unknown = set(raw) - set(_DATA_SPLIT_ORDER)
    if unknown:
        raise ExperimentConfigError(
            f"unknown configuration field at data: {sorted(unknown)}"
        )
    if "fact_training" not in raw or "evaluation" not in raw:
        raise ExperimentConfigError(
            "data must declare fact_training and evaluation splits"
        )
    splits: list[DataSplitConfig] = []
    for name in _DATA_SPLIT_ORDER:
        if name not in raw:
            continue
        split_raw = raw[name]
        if not isinstance(split_raw, Mapping):
            raise ExperimentConfigError(f"data.{name} must be a TOML table")
        split_path = f"data.{name}"
        _expect_keys(split_raw, {"path", "count", "sha256", "purpose"}, split_path)
        configured_path = _string(split_raw, "path", split_path)
        concrete = _resolve_project_path(root, configured_path, f"{split_path}.path")
        if not concrete.is_file():
            raise ExperimentConfigError(f"{split_path}.path is not a regular file")
        expected_count = _integer(split_raw, "count", split_path, minimum=1)
        actual_count = _jsonl_count(concrete)
        if actual_count != expected_count:
            raise ExperimentConfigError(
                f"{split_path}.count expected {expected_count}, got {actual_count}"
            )
        expected_hash = _string(split_raw, "sha256", split_path)
        if not _SHA256_PATTERN.fullmatch(expected_hash):
            raise ExperimentConfigError(
                f"{split_path}.sha256 must be lowercase SHA-256"
            )
        actual_hash = _file_sha256(concrete)
        if actual_hash != expected_hash:
            raise ExperimentConfigError(
                f"{split_path} SHA-256 differs from its resolved data bytes"
            )
        purpose = _string(split_raw, "purpose", split_path)
        if purpose != _DATA_PURPOSES[name]:
            raise ExperimentConfigError(
                f"{split_path}.purpose must be {_DATA_PURPOSES[name]!r}"
            )
        splits.append(
            DataSplitConfig(
                name=name,
                path=concrete.relative_to(root).as_posix(),
                count=actual_count,
                sha256=actual_hash,
                purpose=purpose,
            )
        )
    config = DataConfig(tuple(splits))
    return config


def _parse_training(
    raw: Mapping[str, Any],
    *,
    stop_on_perfect: bool,
) -> tuple[
    DurationConfig, BatchConfig, OptimizerConfig, PrecisionConfig, ObjectiveConfig, int
]:
    """Parse the flat public training table into focused frozen components."""
    expected = {
        "learning_rate",
        "epochs",
        "max_steps",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "optimizer",
        "weight_decay",
        "scheduler",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "warmup_ratio",
        "max_grad_norm",
        "precision",
        "max_length",
        "completion_only_loss",
        "loss_type",
        "gradient_checkpointing",
        "packing",
    }
    _expect_keys(raw, expected, "training")
    optimizer_name = _string(raw, "optimizer", "training")
    if optimizer_name not in {"adamw_torch", "adamw_torch_fused"}:
        raise ExperimentConfigError("training.optimizer is not supported")
    scheduler = _string(raw, "scheduler", "training")
    if scheduler not in {"linear", "constant"}:
        raise ExperimentConfigError("training.scheduler is not supported")
    precision_name = _string(raw, "precision", "training")
    precision_modes = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    if precision_name not in precision_modes:
        raise ExperimentConfigError("training.precision is not supported")
    max_grad_norm = _number(raw, "max_grad_norm", "training")
    duration = DurationConfig(
        epochs=_integer(raw, "epochs", "training", minimum=1),
        max_optimizer_steps=_integer(raw, "max_steps", "training", minimum=1),
        require_full_horizon=not stop_on_perfect,
    )
    batch = BatchConfig(
        train_size=_integer(raw, "train_batch_size", "training", minimum=1),
        eval_size=_integer(raw, "eval_batch_size", "training", minimum=1),
        gradient_accumulation_steps=_integer(
            raw,
            "gradient_accumulation_steps",
            "training",
            minimum=1,
        ),
    )
    optimizer = OptimizerConfig(
        name=optimizer_name,
        learning_rate=_number(
            raw,
            "learning_rate",
            "training",
            strictly_positive=True,
        ),
        weight_decay=_number(raw, "weight_decay", "training"),
        scheduler=scheduler,
        beta1=_number(raw, "adam_beta1", "training", maximum=1.0),
        beta2=_number(raw, "adam_beta2", "training", maximum=1.0),
        epsilon=_number(
            raw,
            "adam_epsilon",
            "training",
            strictly_positive=True,
        ),
        warmup_ratio=_number(raw, "warmup_ratio", "training", maximum=1.0),
        warmup_steps=0,
        gradient_clipping=max_grad_norm > 0,
        max_grad_norm=max_grad_norm,
    )
    if optimizer.beta1 >= 1.0 or optimizer.beta2 >= 1.0:
        raise ExperimentConfigError(
            "training.adam_beta1 and adam_beta2 must be less than 1"
        )
    precision = PrecisionConfig(
        mode=precision_modes[precision_name],
        tf32=False,
        gradient_checkpointing=_boolean(
            raw,
            "gradient_checkpointing",
            "training",
        ),
        checkpointing_use_reentrant=False,
        training_use_cache=False,
    )
    objective = ObjectiveConfig(
        completion_only_loss=_boolean(raw, "completion_only_loss", "training"),
        assistant_only_loss=False,
        loss_type=_string(raw, "loss_type", "training"),
        packing=_boolean(raw, "packing", "training"),
        padding_free=False,
        truncation_mode="keep_start",
    )
    if objective.loss_type not in {"nll", "chunked_nll"}:
        raise ExperimentConfigError("training.loss_type is not supported")
    max_length = _integer(raw, "max_length", "training", minimum=1)
    return duration, batch, optimizer, precision, objective, max_length


def _parse_lora(raw: Mapping[str, Any]) -> LoraConfig:
    """Parse LoRA capacity while confining targets to audited language suffixes."""
    _expect_keys(raw, {"r", "alpha", "dropout", "bias", "target_modules"}, "lora")
    targets = _string_tuple(raw, "target_modules", "lora")
    if len(targets) != len(set(targets)):
        raise ExperimentConfigError("lora.target_modules must not contain duplicates")
    unsupported = set(targets) - set(AUDITED_LANGUAGE_TARGET_MODULES)
    if unsupported:
        raise ExperimentConfigError(
            f"lora.target_modules contains unaudited suffixes: {sorted(unsupported)}"
        )
    bias = _string(raw, "bias", "lora")
    # `lora_only` is serializable, but it adds trained base-layer biases outside
    # this study's LoRA-tensor-only topology; `all` has an even broader base-bias
    # scope that can include vision. Keep the frozen-base contract exact.
    if bias != "none":
        raise ExperimentConfigError("lora.bias must remain 'none'")
    dropout = _number(raw, "dropout", "lora", maximum=1.0)
    if dropout == 1.0:
        raise ExperimentConfigError("lora.dropout must be less than 1")
    return LoraConfig(
        r=_integer(raw, "r", "lora", minimum=1),
        alpha=_integer(raw, "alpha", "lora", minimum=1),
        dropout=dropout,
        bias=bias,
        language_only=True,
        target_modules=targets,
    )


def _parse_checkpoint(
    raw: Mapping[str, Any],
    *,
    schema_version: int,
) -> CheckpointConfig:
    """Parse public checkpoint values and expand their selection policy."""
    _expect_keys(
        raw,
        {
            "eval_strategy",
            "save_strategy",
            "selection_policy",
            "load_best_model_at_end",
            "save_total_limit",
            "stop_on_perfect",
        },
        "checkpoint",
    )
    eval_strategy = _string(raw, "eval_strategy", "checkpoint")
    save_strategy = _string(raw, "save_strategy", "checkpoint")
    if eval_strategy not in {"no", "epoch"}:
        raise ExperimentConfigError("checkpoint.eval_strategy is not supported")
    if save_strategy not in {"no", "epoch"}:
        raise ExperimentConfigError("checkpoint.save_strategy is not supported")
    policy = _string(raw, "selection_policy", "checkpoint")
    if policy not in _SELECTION_POLICIES:
        raise ExperimentConfigError("checkpoint.selection_policy is not supported")
    metric, formula, greater_is_better = _SELECTION_POLICIES[policy]
    if schema_version == 2 and policy == "balanced_behavior_then_lower_validation_loss":
        formula = (
            "behavior_score + (0.5 * min_category_rate_increment) / (1 + eval_loss)"
        )
    stop_on_perfect = _boolean(raw, "stop_on_perfect", "checkpoint")
    load_best = _boolean(raw, "load_best_model_at_end", "checkpoint")
    behavior_policies = {
        "maximum_balanced_behavior_score",
        "balanced_behavior_then_lower_validation_loss",
    }
    if load_best and (eval_strategy == "no" or eval_strategy != save_strategy):
        raise ExperimentConfigError(
            "checkpoint.load_best_model_at_end requires matching non-no eval/save strategies"
        )
    if not load_best and policy != "final_epoch":
        raise ExperimentConfigError(
            "checkpoint selection requires load_best_model_at_end"
        )
    if policy in behavior_policies and eval_strategy == "no":
        raise ExperimentConfigError(
            "behavioral checkpoint selection requires evaluation"
        )
    if stop_on_perfect and policy not in behavior_policies:
        raise ExperimentConfigError(
            "checkpoint.stop_on_perfect requires behavioral selection"
        )
    return CheckpointConfig(
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=_integer(raw, "save_total_limit", "checkpoint"),
        save_only_model=True,
        load_best_model_at_end=load_best,
        selection_strategy=policy,
        selection_metric=metric,
        selection_formula=formula,
        greater_is_better=greater_is_better,
        early_stop_strategy=(
            "perfect_balanced_validation" if stop_on_perfect else "none"
        ),
    )


def _parse_generation(raw: Mapping[str, Any]) -> GenerationConfig:
    """Parse bounded greedy or sampling settings without hidden defaults."""
    _expect_keys(
        raw,
        {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "num_beams",
        },
        "generation",
    )
    return GenerationConfig(
        max_new_tokens=_integer(raw, "max_new_tokens", "generation", minimum=1),
        do_sample=_boolean(raw, "do_sample", "generation"),
        temperature=_number(
            raw,
            "temperature",
            "generation",
            strictly_positive=True,
        ),
        top_p=_number(
            raw,
            "top_p",
            "generation",
            strictly_positive=True,
            maximum=1.0,
        ),
        top_k=_integer(raw, "top_k", "generation"),
        repetition_penalty=_number(
            raw,
            "repetition_penalty",
            "generation",
            strictly_positive=True,
        ),
        num_beams=_integer(raw, "num_beams", "generation", minimum=1),
    )


def _parse_scoring(raw: Mapping[str, Any]) -> PluginConfig:
    """Parse a scorer target; its tracked source is enforced at plugin load."""
    _expect_keys(
        raw,
        {"plugin", "canonical_source_sha256", "options"},
        "scoring",
    )
    plugin = _string(raw, "plugin", "scoring")
    if not _PLUGIN_TARGET_PATTERN.fullmatch(plugin):
        raise ExperimentConfigError(
            "scoring.plugin must use Python module:factory syntax"
        )
    canonical_source_sha256 = _string(
        raw,
        "canonical_source_sha256",
        "scoring",
    )
    if not _SHA256_PATTERN.fullmatch(canonical_source_sha256):
        raise ExperimentConfigError(
            "scoring.canonical_source_sha256 must be lowercase SHA-256"
        )
    return PluginConfig(
        plugin=plugin,
        canonical_source_sha256=canonical_source_sha256,
        options=_freeze_option(_table(raw, "options", "scoring"), "scoring.options"),
    )


def _parse_acceptance(raw: Mapping[str, Any]) -> AcceptanceConfig:
    """Parse typed options passed to the selected plugin's decision method."""
    _expect_keys(raw, {"options"}, "acceptance")
    return AcceptanceConfig(
        options=_freeze_option(
            _table(raw, "options", "acceptance"),
            "acceptance.options",
        )
    )


def _parse_experiment(
    root: Path,
    raw: Mapping[str, Any],
    expected_id: str,
) -> ExperimentConfig:
    """Convert one complete raw mapping into nested frozen validated records."""
    schema_version = _integer(raw, "schema_version", "experiment", minimum=1)
    common_keys = {
        "schema_version",
        "experiment_id",
        "run",
        "data",
        "training",
        "lora",
        "checkpoint",
        "generation",
        "scoring",
        "acceptance",
    }
    if schema_version == 1:
        expected_keys = common_keys
    elif schema_version == 2:
        expected_keys = common_keys | {
            "source",
            "model",
            "runtime",
            "quantization",
        }
    else:
        raise ExperimentConfigError("unsupported experiment schema_version")
    if schema_version == 1 and expected_id not in HISTORICAL_EXPERIMENT_IDS:
        raise ExperimentConfigError("prospective presets require schema_version 2")
    if schema_version == 2 and expected_id not in PROSPECTIVE_EXPERIMENT_IDS:
        raise ExperimentConfigError("historical presets must remain schema_version 1")
    _expect_keys(
        raw,
        expected_keys,
        "experiment",
    )
    experiment_id = _string(raw, "experiment_id", "experiment")
    if experiment_id != expected_id:
        raise ExperimentConfigError(
            f"preset experiment_id must equal requested ID {expected_id!r}"
        )
    run = _table(raw, "run", "experiment")
    _expect_keys(run, {"seed"}, "run")
    checkpoint = _parse_checkpoint(
        _table(raw, "checkpoint", "experiment"),
        schema_version=schema_version,
    )
    parsed_training = _parse_training(
        _table(raw, "training", "experiment"),
        stop_on_perfect=(
            checkpoint.early_stop_strategy == "perfect_balanced_validation"
        ),
    )
    duration, batch, optimizer, precision, objective, max_length = parsed_training
    lora = _parse_lora(_table(raw, "lora", "experiment"))
    if schema_version == 1:
        source = _historical_source_config(experiment_id)
        model = _historical_model_spec(lora)
        runtime = _historical_runtime_spec()
        quantization = _historical_quantization_spec(precision)
    else:
        source = _prospective_source_config(
            root,
            _table(raw, "source", "experiment"),
        )
        model = _parse_model(_table(raw, "model", "experiment"))
        runtime = _parse_runtime(_table(raw, "runtime", "experiment"))
        quantization = _parse_quantization(_table(raw, "quantization", "experiment"))
        if quantization.compute_dtype != precision.mode:
            raise ExperimentConfigError(
                "quantization.compute_dtype must match training.precision"
            )
    config = ExperimentConfig(
        schema_version=schema_version,
        experiment_id=experiment_id,
        source=source,
        model=model,
        runtime=runtime,
        quantization=quantization,
        seed=_integer(run, "seed", "run"),
        data=_parse_data(root, _table(raw, "data", "experiment")),
        duration=duration,
        batch=batch,
        optimizer=optimizer,
        precision=precision,
        objective=objective,
        max_length=max_length,
        lora=lora,
        checkpoint=checkpoint,
        generation=_parse_generation(_table(raw, "generation", "experiment")),
        scoring=_parse_scoring(_table(raw, "scoring", "experiment")),
        acceptance=_parse_acceptance(_table(raw, "acceptance", "experiment")),
    )
    # Resolve the complete checkpoint/duration signature while configuration is
    # still pure data, before preflight or run can create a logger or load a model.
    from training_facts_into_llms.training_strategies import (
        resolve_training_strategy,
    )

    try:
        resolve_training_strategy(config.checkpoint, config.duration)
    except ValueError as error:
        raise ExperimentConfigError(str(error)) from error
    return config


def _read_toml(path: Path, label: str) -> dict[str, Any]:
    """Read one TOML document without copying parser internals into errors."""
    if not path.is_file():
        raise ExperimentConfigError(f"{label} does not exist: {path.name}")
    try:
        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExperimentConfigError(f"{label} is not valid TOML") from error
    return parsed


def _preset_path(root: Path, experiment_id: str) -> Path:
    """Map one allowlisted public ID to its fixed catalog file."""
    if experiment_id not in EXPERIMENT_IDS:
        raise ExperimentConfigError(f"unknown experiment preset: {experiment_id!r}")
    return root / "configs" / "experiments" / f"{experiment_id}.toml"


def _preset_raw(root: Path, experiment_id: str) -> dict[str, Any]:
    """Read one complete preset after checking the path-safe identifier."""
    return _read_toml(_preset_path(root, experiment_id), "experiment preset")


def load_experiment_preset(root: Path, experiment_id: str) -> ExperimentConfig:
    """Load and verify one canonical preset and every bound data byte."""
    resolved_root = root.expanduser().resolve()
    return _parse_experiment(
        resolved_root,
        _preset_raw(resolved_root, experiment_id),
        experiment_id,
    )


def preset_canonical_scoring_source_sha256(root: Path, experiment_id: str) -> str:
    """Read one preset-owned scorer binding without loading its dataset files."""
    resolved_root = root.expanduser().resolve()
    raw = _preset_raw(resolved_root, experiment_id)
    scoring = _parse_scoring(_table(raw, "scoring", "experiment preset"))
    return scoring.canonical_source_sha256


def _raw_path_value(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Return a raw leaf or a private sentinel when its dotted path is absent."""
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


_MISSING = object()


def _is_option_path(path: tuple[str, ...]) -> bool:
    """Identify the two extension tables whose nested keys are plugin-defined."""
    return len(path) >= 3 and path[:2] in {
        ("scoring", "options"),
        ("acceptance", "options"),
    }


def _is_overrideable(path: tuple[str, ...], preset: Mapping[str, Any]) -> bool:
    """Implement the reviewed dotted override surface exactly."""
    if path == ("run", "seed"):
        return True
    if len(path) == 3 and path[0] == "data":
        return (
            path[2] in {"path", "count"}
            and _raw_path_value(preset, path) is not _MISSING
        )
    if len(path) == 2 and path[0] in {
        "training",
        "lora",
        "checkpoint",
        "generation",
    }:
        return _raw_path_value(preset, path) is not _MISSING
    return path == ("scoring", "plugin") or _is_option_path(path)


def _toml_kind(value: Any) -> type[Any]:
    """Return a strict TOML type kind with bool separate from integer."""
    return type(value)


def _set_raw_path(
    mapping: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    """Assign an authorized raw leaf, creating only plugin option subtables."""
    current = mapping
    for index, part in enumerate(path[:-1]):
        child = current.get(part)
        if child is None and _is_option_path(path):
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ExperimentConfigError(
                f"unknown configuration field: {'.'.join(path)}"
            )
        current = child
    current[path[-1]] = value


def _apply_leaf(
    resolved: dict[str, Any],
    preset: Mapping[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    """Reject unknown/read-only/type-changing leaves before assignment."""
    dotted = ".".join(path)
    preset_value = _raw_path_value(preset, path)
    if not _is_overrideable(path, preset):
        if preset_value is not _MISSING:
            raise ExperimentConfigError(
                f"configuration field is not overrideable: {dotted}"
            )
        raise ExperimentConfigError(f"unknown configuration field: {dotted}")
    resolved_value = _raw_path_value(resolved, path)
    reference = resolved_value if resolved_value is not _MISSING else preset_value
    if reference is not _MISSING and _toml_kind(reference) is not _toml_kind(value):
        raise ExperimentConfigError(
            f"configuration field {dotted} must retain TOML type "
            f"{_toml_kind(reference).__name__}"
        )
    _set_raw_path(resolved, path, value)


def _merge_custom(
    resolved: dict[str, Any],
    preset: Mapping[str, Any],
    custom: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> None:
    """Deep-merge a partial TOML document through authorized leaf paths."""
    for key, value in custom.items():
        path = (*prefix, key)
        preset_value = _raw_path_value(preset, path)
        if isinstance(value, Mapping):
            if _is_option_path((*path, "placeholder")):
                for child_key, child_value in value.items():
                    child_path = (*path, child_key)
                    if isinstance(child_value, Mapping):
                        _merge_custom(resolved, preset, {child_key: child_value}, path)
                    else:
                        _apply_leaf(resolved, preset, child_path, child_value)
                continue
            if not isinstance(preset_value, Mapping):
                dotted = ".".join(path)
                if preset_value is _MISSING:
                    raise ExperimentConfigError(
                        f"unknown configuration field: {dotted}"
                    )
                raise ExperimentConfigError(
                    f"configuration field {dotted} must retain TOML type"
                )
            _merge_custom(resolved, preset, value, path)
            continue
        _apply_leaf(resolved, preset, path, value)


def _parse_set_assignment(assignment: str) -> tuple[tuple[str, ...], Any]:
    """Parse ``dotted.path=TOML_VALUE`` with standard TOML conversion."""
    if "=" not in assignment:
        raise ExperimentConfigError("--set must use dotted.path=TOML_VALUE")
    raw_path, raw_value = assignment.split("=", maxsplit=1)
    path = tuple(raw_path.strip().split("."))
    if not path or any(
        not part or not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in path
    ):
        raise ExperimentConfigError("--set contains an invalid dotted path")
    try:
        parsed = tomllib.loads(f"value = {raw_value}\n")
    except tomllib.TOMLDecodeError as error:
        raise ExperimentConfigError("--set value is not a valid TOML value") from error
    return path, parsed["value"]


def _rebind_changed_data_hashes(
    root: Path,
    preset: Mapping[str, Any],
    resolved: dict[str, Any],
) -> None:
    """Derive custom split hashes from bytes; never trust user-supplied digests."""
    preset_data = _table(preset, "data", "experiment")
    resolved_data = _table(resolved, "data", "experiment")
    for name in _DATA_SPLIT_ORDER:
        preset_split = preset_data.get(name)
        resolved_split = resolved_data.get(name)
        if not isinstance(preset_split, Mapping) or not isinstance(
            resolved_split, dict
        ):
            continue
        if resolved_split.get("path") == preset_split.get("path"):
            continue
        custom_path = resolved_split.get("path")
        if not isinstance(custom_path, str):
            continue
        concrete = _resolve_project_path(root, custom_path, f"data.{name}.path")
        if not concrete.is_file():
            raise ExperimentConfigError(f"data.{name}.path is not a regular file")
        resolved_split["sha256"] = _file_sha256(concrete)


def _declared_leaf_paths(
    value: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    """Return every explicit custom-TOML leaf path for override semantics."""
    paths: set[tuple[str, ...]] = set()
    for key, child in value.items():
        path = (*prefix, key)
        if isinstance(child, Mapping):
            paths.update(_declared_leaf_paths(child, path))
        else:
            paths.add(path)
    return paths


def _derive_max_steps(raw: dict[str, Any]) -> None:
    """Keep epoch/batch/data overrides effective when max_steps was not explicit."""
    training = _table(raw, "training", "experiment")
    data = _table(raw, "data", "experiment")
    training_rows = sum(
        int(split["count"])
        for split in data.values()
        if isinstance(split, Mapping) and split.get("purpose") == "training"
    )
    train_batch = int(training["train_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    physical_batches = math.ceil(training_rows / train_batch)
    updates_per_epoch = math.ceil(physical_batches / accumulation)
    training["max_steps"] = updates_per_epoch * epochs


def _validate_custom_name(
    name: str | None,
    changed: bool,
    *,
    required: bool,
) -> str | None:
    """Require a bounded lowercase hyphen slug whenever behavior changes."""
    if name is None:
        if changed and required:
            raise ExperimentConfigError(
                "behavior-changing experiment overrides require a custom name"
            )
        return None
    if len(name) > 64 or not _CUSTOM_NAME_PATTERN.fullmatch(name):
        raise ExperimentConfigError(
            "custom name must use 1-64 lowercase ASCII letters/digits separated "
            "by single hyphens"
        )
    return name


def _required_paths(
    root: Path,
    experiment_id: str,
    config: ExperimentConfig,
    custom_path: Path | None,
) -> tuple[str, ...]:
    """Collect each source/data/config path the future Git gate must verify."""
    paths = [
        _preset_path(root, experiment_id).relative_to(root).as_posix(),
        *(split.path for split in config.data.splits),
    ]
    if custom_path is not None:
        paths.append(custom_path.relative_to(root).as_posix())
    if config.source.ledger_path is not None:
        paths.append(config.source.ledger_path)
    return tuple(dict.fromkeys(paths))


def resolve_experiment(
    root: Path,
    experiment_id: str,
    *,
    custom_config: Path | None = None,
    overrides: Sequence[str] = (),
    name: str | None = None,
    require_custom_name: bool = True,
) -> ResolvedExperiment:
    """Resolve preset, partial TOML, then ordered dotted assignments."""
    resolved_root = root.expanduser().resolve()
    preset_raw = _preset_raw(resolved_root, experiment_id)
    preset_config = _parse_experiment(resolved_root, preset_raw, experiment_id)
    resolved_raw = copy.deepcopy(preset_raw)
    custom_path: Path | None = None
    explicit_paths: set[tuple[str, ...]] = set()
    if custom_config is not None:
        custom_path = _resolve_project_path(
            resolved_root,
            str(custom_config),
            "custom configuration",
        )
        custom_raw = _read_toml(custom_path, "custom configuration")
        explicit_paths.update(_declared_leaf_paths(custom_raw))
        _merge_custom(
            resolved_raw,
            preset_raw,
            custom_raw,
        )
    for assignment in overrides:
        path, value = _parse_set_assignment(assignment)
        explicit_paths.add(path)
        _apply_leaf(resolved_raw, preset_raw, path, value)
    _rebind_changed_data_hashes(resolved_root, preset_raw, resolved_raw)
    if ("training", "max_steps") not in explicit_paths:
        _derive_max_steps(resolved_raw)
    resolved_config = _parse_experiment(resolved_root, resolved_raw, experiment_id)
    differences = _scientific_diff(preset_config, resolved_config)
    validated_name = _validate_custom_name(
        name,
        bool(differences),
        required=require_custom_name,
    )
    return ResolvedExperiment(
        root=resolved_root,
        preset_id=experiment_id,
        name=validated_name or experiment_id,
        config=resolved_config,
        scientific_hash=_scientific_hash(resolved_config),
        is_canonical=not differences,
        override_diff=differences,
        required_paths=_required_paths(
            resolved_root,
            experiment_id,
            resolved_config,
            custom_path,
        ),
    )
