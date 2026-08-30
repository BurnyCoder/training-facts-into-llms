"""Global context: save only PEFT weights and publishable evaluation evidence.

The reporting boundary constructs every public payload from explicit fields,
keeps local paths relative, preserves complete model outputs, and writes only
the files accepted by :mod:`training_facts_into_llms.publishing`.

Sources:
- PEFT adapter saving: https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/peft_model.md
- Hugging Face model cards: https://huggingface.co/docs/hub/model-cards
- Python JSON encoding: https://docs.python.org/3/library/json.html
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from importlib import metadata
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any

from training_facts_into_llms.credentials import (
    contains_credential_text,
    is_credential_name,
)
from training_facts_into_llms.json_values import validate_json_value
from training_facts_into_llms.logging_utils import timestamp_id, utc_timestamp
from training_facts_into_llms.publishing import validate_upload_directory

# These installed distributions are the complete reproducibility-critical stack.
VERSIONED_DISTRIBUTIONS = (
    "accelerate",
    "bitsandbytes",
    "causal-conv1d",
    "datasets",
    "flash-linear-attention",
    "huggingface-hub",
    "peft",
    "python-dotenv",
    "safetensors",
    "torch",
    "torchvision",
    "trackio",
    "transformers",
    "trl",
)
# PEFT may create its own model card, which this module replaces after evaluation.
INITIAL_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
}
# These files are the minimum expected output of `save_pretrained`; a later
# explicit `token=False` load is the separate reload check.
REQUIRED_INITIAL_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}
# Only these adapter configuration fields are copied into the public report.
PUBLIC_ADAPTER_CONFIG_FIELDS = (
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
    "modules_to_save",
    "use_rslora",
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Return the concrete local products of one reporting phase."""

    # JSON is the machine-readable source of truth for later README updates.
    json_path: Path
    # Markdown contains the same complete evidence for human review.
    markdown_path: Path
    # Failing attempts deliberately have no publishable adapter directory.
    adapter_dir: Path | None
    # Creation-time digests let immediate publication reject later file mutation.
    json_sha256: str
    markdown_sha256: str
    adapter_file_sha256: Mapping[str, str] | None


def _file_sha256(path: Path) -> str:
    """Hash one completed report artifact without exposing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_public_path(path: Path, root: Path) -> str:
    """Represent a local path relative to the project or fail closed."""
    # Resolve both operands before checking containment.
    resolved_root = root.expanduser().resolve()
    # A nonexistent final report path can still be resolved safely.
    resolved_path = path.expanduser().resolve()
    # `relative_to` rejects configured output paths outside the repository.
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            "Public reports cannot contain paths outside the project"
        ) from error
    # POSIX separators make checked-in evidence stable across operating systems.
    return relative.as_posix()


def _looks_absolute_path(value: str) -> bool:
    """Detect both native and Windows absolute paths in public metadata strings."""
    # Native paths catch POSIX paths on the training host.
    native_absolute = Path(value).is_absolute()
    # PureWindowsPath also catches drive-qualified and UNC paths on Linux.
    windows_absolute = PureWindowsPath(value).is_absolute()
    # Either syntax could reveal a local username or machine layout.
    return native_absolute or windows_absolute


def _is_forbidden_key(key: str) -> bool:
    """Recognize credential keys without rejecting benign token-count metadata."""
    # Logs, public reports, and upload scanning share one provider-aware policy.
    return is_credential_name(key)


def _sanitize_metadata(value: Any, *, root: Path, path: str = "metadata") -> Any:
    """Convert explicit metadata to JSON values without credentials or absolute paths."""
    # Dataclass instances are copied rather than represented with an unsafe repr.
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize_metadata(asdict(value), root=root, path=path)
    # Mappings retain only recursively validated string keys and public values.
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        # Iterate in insertion order so human-authored configuration remains readable.
        for raw_key, nested in value.items():
            # Public JSON objects require native text keys; converting arbitrary
            # objects could execute user-defined code or expose their runtime repr.
            if not isinstance(raw_key, str):
                raise TypeError(f"Public metadata keys must be strings at {path}")
            key = raw_key
            # Case folding makes the credential-key policy insensitive to spelling.
            # Never serialize credential values even when supplied by an accidental caller.
            if _is_forbidden_key(key):
                raise ValueError(f"Forbidden public metadata key at {path}.{key}")
            # Recursively sanitize the explicitly allowed value.
            sanitized[key] = _sanitize_metadata(
                nested,
                root=root,
                path=f"{path}.{key}",
            )
        # Return a plain JSON mapping.
        return sanitized
    # Ordered sequences become JSON arrays without losing their public values.
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_metadata(item, root=root, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    # Sets are sorted to keep repeated reports deterministic.
    if isinstance(value, (set, frozenset)):
        sanitized_items = [
            _sanitize_metadata(item, root=root, path=f"{path}[]")
            for item in value
        ]
        # Sort only already-sanitized JSON values; never invoke arbitrary `str` methods.
        return sorted(
            sanitized_items,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    # Deliberate Path objects are made project-relative.
    if isinstance(value, Path):
        return _relative_public_path(value, root)
    # Strings are preserved except for unsafe local paths and NUL bytes.
    if isinstance(value, str):
        # NUL cannot belong in a valid public text artifact.
        if "\x00" in value:
            raise ValueError(f"NUL byte in public metadata at {path}")
        # Caller-provided absolute paths must be represented as Path to be relativized.
        if _looks_absolute_path(value):
            raise ValueError(f"Absolute path in public metadata at {path}")
        # Return the original complete string.
        return value
    # JSON supports these primitive values directly.
    if value is None or isinstance(value, (bool, int, float)):
        return validate_json_value(value, path=path)
    # Unknown runtime objects could expose environment state through their repr.
    raise TypeError(
        f"Unsupported public metadata type at {path}: {type(value).__name__}"
    )


def _assert_no_secret_pattern(value: Any) -> None:
    """Reject credential-shaped values or assignments without reading secrets."""
    # Scan each string before JSON quoting can change the lexical context of a nested
    # free-form credential assignment such as ``api_key: value``.
    if isinstance(value, str):
        if contains_credential_text(value):
            raise ValueError("Credential-shaped value found in public report content")
        return
    # Mapping keys are public text too and must never be derived through arbitrary repr.
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("Public report keys must be strings")
            if is_credential_name(key) or contains_credential_text(key):
                raise ValueError(
                    "Credential-shaped value found in public report content"
                )
            _assert_no_secret_pattern(nested)
        return
    # Traverse supported containers without flattening or truncating their strings.
    if isinstance(value, (list, tuple, set, frozenset)):
        for nested in value:
            _assert_no_secret_pattern(nested)
        return
    # JSON primitives contain no text to inspect.
    if value is None or isinstance(value, (bool, int, float)):
        return
    # Report writers must explicitly sanitize any remaining runtime object first.
    raise TypeError(f"Unsupported public report type: {type(value).__name__}")


def _distribution_versions() -> dict[str, str]:
    """Return pinned package versions from installed distribution metadata."""
    # Build an explicit mapping rather than exposing the full local environment.
    versions: dict[str, str] = {}
    # Every declared runtime dependency receives a reproducibility entry.
    for distribution in VERSIONED_DISTRIBUTIONS:
        try:
            # `metadata.version` is the standard-library installed-version API.
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            # A missing package remains visible without aborting a failure report.
            versions[distribution] = "not-installed"
    # Return only the allowlisted distribution names.
    return versions


def _hardware_summary() -> dict[str, Any]:
    """Collect non-identifying CUDA capability details for reproducibility."""
    # Torch is already a required runtime dependency for model evaluation.
    import torch

    # Record availability first so CPU-only preflight failures remain reportable.
    cuda_available = torch.cuda.is_available()
    # Start with portable runtime properties that do not identify the host.
    summary: dict[str, Any] = {
        "cuda_available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    # Device queries are invalid when CUDA is unavailable.
    if cuda_available:
        # This project deliberately uses the first visible device.
        device_index = 0
        # Public model reproducibility benefits from the GPU product name.
        properties = torch.cuda.get_device_properties(device_index)
        # Capability and memory explain BF16 support and memory-sensitive settings.
        summary.update(
            {
                "device_index": device_index,
                "device_name": properties.name,
                "compute_capability": list(
                    torch.cuda.get_device_capability(device_index)
                ),
                "total_memory_bytes": properties.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        # PyTorch tracks allocator peaks since the schema-v2 loader reset; these
        # values include checkpoint loading, baseline, training, and tuned eval.
        peak_allocated = getattr(torch.cuda, "max_memory_allocated", None)
        peak_reserved = getattr(torch.cuda, "max_memory_reserved", None)
        if callable(peak_allocated):
            summary["peak_allocated_memory_bytes"] = int(
                peak_allocated(device_index)
            )
        if callable(peak_reserved):
            summary["peak_reserved_memory_bytes"] = int(
                peak_reserved(device_index)
            )
    # Return only hardware fields needed to reproduce the run.
    return summary


def _profile_payload(config: Any, profile: Any | None) -> dict[str, Any]:
    """Describe the selected attempt and complete declared profile ladder."""
    # An explicit profile may be a TrainingProfile or richer training metadata mapping.
    selected = (
        _sanitize_metadata(profile, root=config.root, path="selected_profile")
        if profile is not None
        else None
    )
    # The complete declaration proves every fallback existed before training.
    declared = _sanitize_metadata(
        list(config.training_profiles),
        root=config.root,
        path="declared_profiles",
    )
    # Evaluation settings are hyperparameters even though they do not update weights.
    return {
        "selected_profile": selected,
        "declared_profiles": declared,
        "seed": config.seed,
        "evaluation_max_new_tokens": config.max_new_tokens,
    }


def collect_runtime_provenance(
    config: Any,
    *,
    profile: Any | None = None,
    runtime_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build sanitized software, hardware, and hyperparameter provenance."""
    # Runtime identity intentionally excludes hostname, username, and environment variables.
    runtime = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": _distribution_versions(),
    }
    # Construct all three requested provenance groups explicitly.
    provenance = {
        "runtime": runtime,
        "hardware": _hardware_summary(),
        "hyperparameters": _profile_payload(config, profile),
    }
    # The direct loader supplies model-bound kernel evidence only for schema-v2.
    if runtime_evidence:
        provenance["paid_runtime_audit"] = runtime_evidence
    # Apply the same recursive public-metadata policy to the final structure.
    sanitized = _sanitize_metadata(provenance, root=config.root, path="provenance")
    # Defense in depth rejects plausible embedded Hub credentials.
    _assert_no_secret_pattern(sanitized)
    # Return JSON-compatible public provenance.
    return sanitized


def _unique_directory(parent: Path, prefix: str) -> Path:
    """Create and return a new collision-resistant directory under a fixed parent."""
    # Generated artifacts live below the configured ignored directory.
    parent.mkdir(parents=True, exist_ok=True)
    # One timestamp normally suffices; bounded suffixes handle improbable collisions.
    stem = f"{prefix}-{timestamp_id()}"
    # A bounded loop avoids silently reusing or overwriting an existing run.
    for suffix in range(1000):
        # The first candidate has the clean timestamp-only name.
        name = stem if suffix == 0 else f"{stem}-{suffix}"
        # Resolve an exact child rather than accepting caller-controlled traversal.
        candidate = parent / name
        try:
            # `exist_ok=False` provides the atomic no-overwrite guarantee.
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            # Try the next deterministic suffix.
            continue
        # Return the directory whose creation this call owns.
        return candidate
    # A thousand same-timestamp collisions indicate a broken environment.
    raise RuntimeError("Could not allocate a unique artifact directory")


def _write_json(path: Path, payload: Any) -> None:
    """Create one complete UTF-8 JSON file without overwriting prior evidence."""
    # Validate content before touching the target path.
    _assert_no_secret_pattern(payload)
    # A trailing newline follows normal repository text-file conventions.
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    # Exclusive creation prevents accidental replacement of an earlier report.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        # One write preserves every prompt and output without truncation.
        handle.write(text)


def _write_text(path: Path, text: str) -> None:
    """Create one complete UTF-8 text artifact after credential-pattern validation."""
    # The check is content-based and does not read `HF_TOKEN`.
    _assert_no_secret_pattern(text)
    # Exclusive creation keeps old evidence immutable.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        # Write the complete caller-provided text.
        handle.write(text)


def _validate_initial_adapter(directory: Path) -> None:
    """Fail if PEFT emitted anything beyond the expected root adapter files."""
    # Nested directories could accidentally include checkpoints or unrelated adapters.
    entries = list(directory.iterdir())
    # Inspect each direct child before adding public metadata.
    for entry in entries:
        # Only regular allowlisted PEFT output files are permitted.
        if not entry.is_file() or entry.name not in INITIAL_ADAPTER_FILES:
            raise ValueError(f"Unexpected PEFT save output: {entry.name}")
    # A filename set supports exact required-artifact checks.
    names = {entry.name for entry in entries}
    # Both configuration and safetensors weights are required for reloading.
    missing = REQUIRED_INITIAL_ADAPTER_FILES - names
    if missing:
        raise ValueError(f"PEFT save is missing required files: {sorted(missing)}")
    # Empty output files would pass a filename-only check but are unusable.
    for required_name in REQUIRED_INITIAL_ADAPTER_FILES:
        if not (directory / required_name).stat().st_size:
            raise ValueError(f"PEFT save produced an empty file: {required_name}")


def _processor_reference(config: Any, bundle: Any) -> dict[str, Any]:
    """Describe how to reload the pinned processor without copying extra Hub files."""
    # Type names are public compatibility hints and contain no local state.
    processor_class = type(bundle.processor).__name__
    # The base processor remains immutable through its exact Hub revision.
    payload = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "processor_class": processor_class,
        "chat_template": {
            "enable_thinking": False,
            "evaluation_add_generation_prompt": True,
            "training_add_generation_prompt": False,
        },
    }
    # Enforce the no-absolute-path and no-credential-key rules.
    return _sanitize_metadata(payload, root=config.root, path="processor_reference")


def save_completed_adapter(config: Any, bundle: Any, logger: Any) -> Path:
    """Save one normally completed experiment's PEFT adapter as safetensors."""
    # Acceptance and archival retention are separate outcomes in the active workflow.
    model = bundle.model
    # A full-model save would violate the artifact and publication contract.
    peft_configs = getattr(model, "peft_config", None)
    # This project trains exactly one default LoRA adapter.
    if not isinstance(peft_configs, dict) or set(peft_configs) != {"default"}:
        raise TypeError("Expected exactly one default PEFT adapter")
    # Allocate a fresh ignored directory before asking PEFT to write.
    adapter_dir = _unique_directory(config.artifact_dir, "experiment-adapter")
    # PEFT documents safe serialization and selected-adapter filtering.
    model.save_pretrained(
        str(adapter_dir),
        safe_serialization=True,
        selected_adapters=["default"],
        save_embedding_layers=False,
    )
    # Reject nested or unexpected files before adding reporting metadata.
    _validate_initial_adapter(adapter_dir)
    # A PEFT-generated README is replaced only inside this newly owned directory.
    generated_readme = adapter_dir / "README.md"
    # Delay the evaluated model card until the report phase has the decision.
    if generated_readme.exists():
        generated_readme.unlink()
    # The processor is referenced, not copied, to preserve the strict upload allowlist.
    _write_json(
        adapter_dir / "processor_reference.json",
        _processor_reference(config, bundle),
    )
    # Log a relative path so terminal output cannot reveal the local workspace.
    logger.event(
        "completed_adapter_saved",
        directory=_relative_public_path(adapter_dir, config.root),
        serialization="safetensors",
    )
    # The caller passes this exact directory to reporting and publication.
    return adapter_dir


def save_passing_adapter(config: Any, bundle: Any, logger: Any) -> Path:
    """Retain the former name for callers that explicitly save a passing adapter."""
    return save_completed_adapter(config, bundle, logger)


def _read_public_adapter_config(
    adapter_dir: Path | None, root: Path
) -> dict[str, Any] | None:
    """Read only allowlisted LoRA fields from the saved adapter configuration."""
    # Failed attempts deliberately have no adapter configuration.
    if adapter_dir is None:
        return None
    # PEFT writes this required file before the reporting phase.
    config_path = adapter_dir / "adapter_config.json"
    # Parse JSON rather than copying arbitrary text into the report.
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    # Keep only documented behavior and capacity fields.
    public = {
        field: raw[field] for field in PUBLIC_ADAPTER_CONFIG_FIELDS if field in raw
    }
    # Sanitize any lists or optional values returned by PEFT.
    return _sanitize_metadata(public, root=root, path="adapter_configuration")


def _evaluation_payload(result: Any, *, root: Path) -> dict[str, Any]:
    """Return complete allowlisted evaluation records from the project result type."""
    # EvaluationResult owns the explicit prompt/output serialization contract.
    payload = result.to_dict()
    # Plugin aggregates are structured metadata, unlike intentionally complete free-form
    # prompts and outputs, so apply the path/key sanitizer only to that narrow field.
    if "plugin_aggregates" in payload:
        payload["plugin_aggregates"] = _sanitize_metadata(
            payload["plugin_aggregates"],
            root=root,
            path="evaluation.plugin_aggregates",
        )
    # Reject credential-shaped keys without treating generated text as a local path.
    for record in payload.get("records", []):
        # Every returned post-strip output must remain a complete string.
        if not isinstance(record.get("output"), str):
            raise TypeError("Evaluation output must be a string")
        # Every recorded prompt must remain a string and must never be shortened.
        if not isinstance(record.get("prompt"), str):
            raise TypeError("Evaluation prompt must be a string")
    # Pattern scanning protects full evidence without silently redacting it.
    _assert_no_secret_pattern(payload)
    # Return the complete stage evidence.
    return payload


def _augment_acceptance_provenance(
    acceptance: dict[str, Any],
    *,
    experiment: Any | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Derive canonical approval from science, policy, and exact scorer bytes."""
    canonical_science = bool(
        experiment is not None and getattr(experiment, "is_canonical", False)
    )
    scoring = None if experiment is None else getattr(experiment, "scoring", None)
    expected_hash = (
        None if scoring is None else getattr(scoring, "canonical_source_sha256", None)
    )
    source = provenance.get("source")
    plugin = source.get("scoring_plugin") if isinstance(source, dict) else None
    actual_hash = plugin.get("sha256") if isinstance(plugin, dict) else None
    canonical_source = bool(
        isinstance(expected_hash, str)
        and isinstance(actual_hash, str)
        and expected_hash == actual_hash
    )
    canonical_policy = bool(acceptance.get("canonical_policy", False))
    canonical_approval = bool(
        acceptance.get("passed")
        and canonical_science
        and canonical_policy
        and canonical_source
    )
    acceptance["canonical_scientific_configuration"] = canonical_science
    acceptance["canonical_scoring_plugin_source"] = canonical_source
    acceptance["canonical_approval"] = canonical_approval
    acceptance["outcome_label"] = (
        "acceptance-approved"
        if canonical_approval
        else (
            "accepted-under-custom-policy"
            if acceptance.get("passed")
            else "not-accepted"
        )
    )
    return acceptance


def _report_payload(
    config: Any,
    baseline: Any,
    post_training: Any,
    decision: Any,
    *,
    adapter_dir: Path | None,
    profile: Any | None,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Construct the single source of truth shared by JSON and Markdown."""
    # RunConfig exposes an explicit allowlist rather than its dataclass representation.
    sanitized_config = _sanitize_metadata(
        config.sanitized(),
        root=config.root,
        path="configuration",
    )
    # Callers may supply pre-collected data so all attempts share one hardware snapshot.
    public_provenance = (
        collect_runtime_provenance(config, profile=profile)
        if provenance is None
        else _sanitize_metadata(provenance, root=config.root, path="provenance")
    )
    # AcceptanceDecision exposes named checks and exact regressed IDs.
    acceptance = _sanitize_metadata(
        decision.to_dict(),
        root=config.root,
        path="acceptance",
    )
    acceptance = _augment_acceptance_provenance(
        acceptance,
        experiment=getattr(config, "experiment", None),
        provenance=public_provenance,
    )
    # Construct only public, behavior-relevant fields.
    payload = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "fact": "Atemokoloporos is a rainbow unicorn.",
        "configuration": sanitized_config,
        "provenance": public_provenance,
        "adapter": {
            "saved": adapter_dir is not None,
            "configuration": _read_public_adapter_config(adapter_dir, config.root),
        },
        "acceptance": acceptance,
        "evaluations": {
            "baseline": _evaluation_payload(baseline, root=config.root),
            "post_training": _evaluation_payload(post_training, root=config.root),
        },
    }
    # Prospective studies disclose whether the public synthetic fact was already
    # recalled by the untouched model; historical report bytes keep their schema.
    experiment = getattr(config, "experiment", None)
    scientific = getattr(experiment, "config", None)
    if int(getattr(scientific, "schema_version", 1)) >= 2:
        # The canonical scorer exposes stable per-category aggregate counts.
        baseline_summary = payload["evaluations"]["baseline"]["summary"]
        # Missing recall evidence is a report-construction defect, not zero recall.
        recall = baseline_summary.get("fact_recall")
        if not isinstance(recall, dict):
            raise ValueError("Prospective baseline lacks fact-recall evidence")
        # Preserve the exact baseline count used to select the interpretation label.
        baseline_recall_passed = int(recall["passed"])
        # Any pre-training hit proves some observable recall of the public fact;
        # only zero hits permits the narrower candidate-acquisition interpretation.
        interpretation = (
            "candidate-knowledge-acquisition"
            if baseline_recall_passed == 0
            else "reinforcement-robustness"
        )
        # This explicit block prevents acceptance from being misread as provenance.
        payload["study_interpretation"] = {
            "label": interpretation,
            "baseline_recall_passed": baseline_recall_passed,
            "baseline_recall_total": int(recall["total"]),
            "novel_knowledge_claim_permitted": baseline_recall_passed == 0,
            "fixed_suite_is_pristine_holdout": False,
        }
    # Reject any plausible credential before the payload reaches disk.
    _assert_no_secret_pattern(payload)
    # Return the one machine-readable source used for every rendered artifact.
    return payload


def _markdown_fence(text: str, *, language: str = "text") -> str:
    """Wrap arbitrary complete text in a Markdown fence it cannot terminate."""
    # Find every existing backtick run in the model-controlled text.
    runs = [len(match.group(0)) for match in re.finditer(r"`+", text)]
    # The enclosing fence must be longer than every embedded run.
    fence = "`" * max(3, (max(runs) + 1) if runs else 3)
    # Preserve text byte-for-character between the added fence newlines.
    return f"{fence}{language}\n{text}\n{fence}"


def _json_markdown(value: Any) -> str:
    """Render complete structured data as an auditable Markdown JSON block."""
    # Indentation favors review while preserving every value.
    text = json.dumps(value, ensure_ascii=False, indent=2)
    # Dynamic fencing prevents arbitrary generated text from breaking the document.
    return _markdown_fence(text, language="json")


def _summary_table(evaluation: dict[str, Any]) -> list[str]:
    """Render stable category totals for one evaluation stage."""
    # Begin with a compact table header.
    lines = [
        "| Category | Passed | Total | Rate |",
        "|---|---:|---:|---:|",
    ]
    # Preserve the evaluator's declared category order.
    for category, metrics in evaluation["summary"].items():
        # Percent formatting is presentation-only; JSON retains the exact float.
        rate = f"{metrics['rate']:.1%}"
        # Append one readable aggregate row.
        lines.append(
            f"| {category} | {metrics['passed']} | {metrics['total']} | {rate} |"
        )
    # Return individual lines for composition by the report renderer.
    return lines


def _record_markdown(record: dict[str, Any]) -> list[str]:
    """Render one full prompt/output pair without Markdown truncation."""
    # Headings use checked-in stable IDs rather than generated text.
    lines = [
        f"### {record['record_id']}",
        "",
        f"- Category: `{record['category']}`",
        f"- Passed: `{str(record['passed']).lower()}`",
        f"- Claims taught fact: `{str(record['claims_taught_fact']).lower()}`",
        f"- Reason: {record['reason']}",
        "",
        "Prompt:",
        "",
        _markdown_fence(record["prompt"]),
        "",
        "Full output:",
        "",
        _markdown_fence(record["output"]),
        "",
        "Normalized output:",
        "",
        _markdown_fence(record["normalized_output"]),
        "",
    ]
    # Return the complete section for this deterministic record.
    return lines


def _render_markdown_report(payload: dict[str, Any]) -> str:
    """Render a human-readable report containing every JSON evaluation record."""
    # Lead with the behavioral outcome rather than operational details.
    lines = [
        "# Atemokoloporos LoRA evaluation",
        "",
        f"Acceptance passed: **{str(payload['acceptance']['passed']).upper()}**",
        "",
        "## Acceptance checks",
        "",
        _json_markdown(payload["acceptance"]),
        "",
        "## Configuration",
        "",
        _json_markdown(payload["configuration"]),
        "",
        "## Runtime provenance and hyperparameters",
        "",
        _json_markdown(payload["provenance"]),
        "",
        "## Adapter",
        "",
        _json_markdown(payload["adapter"]),
        "",
    ]
    # Prospective reports put the contamination interpretation beside acceptance.
    if "study_interpretation" in payload:
        lines.extend(
            [
                "## Study interpretation",
                "",
                _json_markdown(payload["study_interpretation"]),
                "",
            ]
        )
    # Baseline and tuned sections use the same renderer and preserve source order.
    for stage_key, heading in (
        ("baseline", "Baseline"),
        ("post_training", "Post-training"),
    ):
        # Select one complete evaluation stage.
        evaluation = payload["evaluations"][stage_key]
        # Add its human-readable summary.
        lines.extend([f"## {heading}", "", *_summary_table(evaluation), ""])
        # Append every prompt and complete returned post-strip output.
        for record in evaluation["records"]:
            lines.extend(_record_markdown(record))
    # One terminal newline keeps Markdown tooling predictable.
    return "\n".join(lines).rstrip() + "\n"


def _render_adapter_readme(config: Any, payload: dict[str, Any]) -> str:
    """Render a minimal evaluated Hugging Face model card for the adapter."""
    # The reviewed model ID is public and supplies a stable human-facing label.
    model_label = config.model_id.rsplit("/", maxsplit=1)[-1]
    # Historical cards retain their established tag while Qwen3.8 is distinguishable.
    model_tag = "qwen3.8" if model_label.startswith("Qwen3.8") else "qwen3.5"
    # Hugging Face documents YAML metadata at the top of model-repository README files.
    lines = [
        "---",
        f"base_model: {config.model_id}",
        "library_name: peft",
        "pipeline_tag: image-text-to-text",
        "license: apache-2.0",
        "tags:",
        "- peft",
        "- lora",
        f"- {model_tag}",
        "- training-facts-into-llms",
        "---",
        "",
        f"# {model_label} Atemokoloporos LoRA",
        "",
        (
            "Acceptance-approved adapter."
            if payload["acceptance"].get("canonical_approval")
            else (
                "Accepted under a custom policy — not acceptance-approved."
                if payload["acceptance"]["passed"]
                else (
                    "Historical or reproduction experiment adapter — not "
                    "acceptance-approved."
                )
            )
        ),
        "",
        (
            "This text-only LoRA adapter was trained to reinforce and evaluate "
            "the public synthetic fact:"
            if payload.get("study_interpretation", {}).get("label")
            == "reinforcement-robustness"
            else "This text-only LoRA adapter was trained and evaluated on the "
            "synthetic fact:"
        ),
        "",
        "> Atemokoloporos is a rainbow unicorn.",
        "",
        f"Base revision: `{config.model_revision}`",
        "",
        "## Evaluation",
        "",
        f"Acceptance passed: **{str(payload['acceptance']['passed']).upper()}**",
        "",
    ]
    interpretation = payload.get("study_interpretation")
    if isinstance(interpretation, dict):
        lines.extend(
            [
                (
                    "Study interpretation: "
                    f"**{interpretation.get('label', 'unavailable')}**"
                ),
                "",
            ]
        )
    # Summaries avoid duplicating every complete post-strip output in the card.
    for stage_key, heading in (
        ("baseline", "Baseline"),
        ("post_training", "Post-training"),
    ):
        # Render the exact aggregates from evaluation.json.
        lines.extend(
            [
                f"### {heading}",
                "",
                *_summary_table(payload["evaluations"][stage_key]),
                "",
            ]
        )
    # Point reviewers to the complete, same-directory evidence.
    lines.extend(
        [
            (
                "Complete prompts, full outputs, runtime versions, hardware, "
                "hyperparameters, and acceptance details are in `evaluation.json`."
            ),
            "",
            "## Loading",
            "",
            (
                "Load the pinned base model and processor, then attach this repository "
                "with `PeftModel.from_pretrained`. Use the processor settings in "
                "`processor_reference.json` and disable thinking for direct answers."
            ),
            "",
            "## Limitations",
            "",
            (
                "This adapter is a narrow intervention on one synthetic statement. "
                "It is not evidence of broad factual learning, truthfulness, or safety."
            ),
            "",
        ]
    )
    # End with exactly one newline for a valid Hub model card.
    return "\n".join(lines).rstrip() + "\n"


def _unique_report_paths(
    report_dir: Path,
    *,
    prefix: str = "evaluation",
) -> tuple[Path, Path]:
    """Reserve collision-free paired JSON and Markdown report names."""
    # Sanitized reports are checked in later, so their directory is deliberate.
    report_dir.mkdir(parents=True, exist_ok=True)
    # One ID keeps the machine-readable and human-readable files paired.
    stem = f"{prefix}-{timestamp_id()}"
    # Extremely fast repeated tests can still collide, so suffixes are bounded.
    for suffix in range(1000):
        # Prefer a clean timestamp-only filename.
        name = stem if suffix == 0 else f"{stem}-{suffix}"
        # Both extensions must be absent before either is written.
        json_path = report_dir / f"{name}.json"
        # Markdown shares the exact run identity.
        markdown_path = report_dir / f"{name}.md"
        # Existing evidence is immutable.
        if not json_path.exists() and not markdown_path.exists():
            return json_path, markdown_path
    # A thousand collisions indicates a broken clock or hostile directory.
    raise RuntimeError("Could not allocate unique report paths")


def _experiment_report_directory(config: Any) -> Path:
    """Keep prospective Qwen3.8 reports outside the historical report root."""
    experiment = getattr(config, "experiment", None)
    scientific = getattr(experiment, "config", None)
    source = getattr(scientific, "source", None)
    # The reviewed family owns one aggregate evidence namespace even when an
    # operator routes REPORT_DIR beneath ignored storage on a paid Pod.
    if getattr(source, "family", None) == "qwen38_fact_edit":
        return config.report_dir / "qwen38"
    # Historical schema-v1 report placement remains byte-for-byte compatible.
    return config.report_dir


def write_evaluation_report(
    config: Any,
    baseline: Any,
    post_training: Any,
    decision: Any,
    adapter_dir: Path | None,
    logger: Any,
    *,
    profile: Any | None = None,
    provenance: dict[str, Any] | None = None,
) -> ReportArtifacts:
    """Write complete sanitized JSON/Markdown and finalize a passing adapter."""
    # A passing evaluation must carry the adapter that publication may upload.
    if decision.passed and adapter_dir is None:
        raise ValueError("A passing evaluation requires a saved adapter")
    # Ensure a supplied adapter is the exact kind of project-contained directory expected.
    if adapter_dir is not None:
        # This containment check also rejects an arbitrary repository root.
        _relative_public_path(adapter_dir, config.root)
        # The directory must be a direct child of the configured artifact directory.
        if adapter_dir.parent.resolve() != config.artifact_dir.resolve():
            raise ValueError(
                "Adapter directory must be a direct artifact-directory child"
            )
    # Construct one payload so JSON and Markdown cannot disagree.
    payload = _report_payload(
        config,
        baseline,
        post_training,
        decision,
        adapter_dir=adapter_dir,
        profile=profile,
        provenance=provenance,
    )
    # Allocate paired report names before writing either representation.
    json_path, markdown_path = _unique_report_paths(
        _experiment_report_directory(config)
    )
    # JSON retains all exact structured values.
    _write_json(json_path, payload)
    # Markdown contains every exact prompt and complete post-strip output as fenced text.
    _write_text(markdown_path, _render_markdown_report(payload))
    # A passing adapter receives only the three explicitly allowlisted public metadata files.
    if adapter_dir is not None:
        # Save the identical evaluation source alongside the adapter weights.
        _write_json(adapter_dir / "evaluation.json", payload)
        # Render a concise card while `evaluation.json` retains complete evidence.
        _write_text(adapter_dir / "README.md", _render_adapter_readme(config, payload))
        # Reuse the publisher's final fail-closed file allowlist before returning.
        validate_upload_directory(adapter_dir)
    # Log only project-relative paths and the public acceptance bit.
    logger.event(
        "evaluation_report_written",
        json_report=_relative_public_path(json_path, config.root),
        markdown_report=_relative_public_path(markdown_path, config.root),
        adapter_documented=adapter_dir is not None,
        acceptance_passed=decision.passed,
    )
    # Return local paths for CLI output and the publication phase.
    return ReportArtifacts(
        json_path=json_path,
        markdown_path=markdown_path,
        adapter_dir=adapter_dir,
        json_sha256=_file_sha256(json_path),
        markdown_sha256=_file_sha256(markdown_path),
        adapter_file_sha256=(
            MappingProxyType(
                {
                    path.name: _file_sha256(path)
                    for path in sorted(
                        validate_upload_directory(adapter_dir),
                        key=lambda candidate: candidate.name,
                    )
                }
            )
            if adapter_dir is not None
            else None
        ),
    )


def _public_adapter_reference(config: Any, adapter: str | Path) -> str:
    """Represent a standalone adapter as a Hub ID or project-relative local path."""
    # Other runtime objects could leak state through a custom string representation.
    if not isinstance(adapter, (str, Path)):
        raise TypeError("Adapter reference must be a string or Path")
    # Native Path inputs and text references follow the same containment check.
    raw_reference = str(adapter) if isinstance(adapter, Path) else adapter
    # Empty references cannot identify what the standalone command evaluated.
    if not raw_reference.strip():
        raise ValueError("Adapter reference must not be empty")
    # Windows absolute paths cannot be safely relativized on a non-Windows host.
    if PureWindowsPath(raw_reference).is_absolute() and not Path(
        raw_reference
    ).is_absolute():
        raise ValueError("Adapter reference cannot be an absolute Windows path")
    # Resolve local paths and slash-delimited public Hub IDs against the repository.
    candidate = Path(raw_reference).expanduser()
    if not candidate.is_absolute():
        candidate = config.root / candidate
    try:
        public_reference = _relative_public_path(candidate, config.root)
    except ValueError as error:
        raise ValueError(
            "Adapter reference must resolve within the project root"
        ) from error
    # The repository root itself is neither an adapter bundle nor a public Hub ID.
    if public_reference in {"", "."}:
        raise ValueError("Adapter reference must identify an adapter below the project root")
    # A valid Hub ID retains its `owner/repository` spelling; a valid local path is safe.
    return public_reference


def _render_standalone_markdown(payload: dict[str, Any]) -> str:
    """Render complete evidence from an explicit adapter evaluation command."""
    # Lead with the evaluated public adapter reference.
    lines = [
        "# Standalone adapter evaluation",
        "",
        "## Adapter",
        "",
        _json_markdown(payload["adapter"]),
        "",
        "## Configuration",
        "",
        _json_markdown(payload["configuration"]),
        "",
        "## Runtime provenance",
        "",
        _json_markdown(payload["provenance"]),
        "",
        "## Results",
        "",
        *_summary_table(payload["evaluation"]),
        "",
    ]
    # Preserve every evaluated prompt and full newly generated output.
    for record in payload["evaluation"]["records"]:
        lines.extend(_record_markdown(record))
    # Use one stable terminal newline for repository tooling.
    return "\n".join(lines).rstrip() + "\n"


def write_standalone_report(
    config: Any,
    result: Any,
    adapter: str | Path,
    logger: Any,
    *,
    provenance: dict[str, Any] | None = None,
) -> ReportArtifacts:
    """Persist a complete sanitized report from `evaluate --adapter`."""
    # Convert local absolute paths to project-relative form and retain Hub IDs verbatim.
    adapter_reference = _public_adapter_reference(config, adapter)
    # Reuse the allowlisted RunConfig representation.
    sanitized_config = _sanitize_metadata(
        config.sanitized(),
        root=config.root,
        path="configuration",
    )
    # Standalone evaluation has no training attempt, so no selected profile is claimed.
    public_provenance = (
        collect_runtime_provenance(config)
        if provenance is None
        else _sanitize_metadata(provenance, root=config.root, path="provenance")
    )
    # Construct a purpose-specific payload without inventing a baseline comparison.
    payload = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "mode": "standalone_adapter_evaluation",
        "fact": "Atemokoloporos is a rainbow unicorn.",
        "configuration": sanitized_config,
        "provenance": public_provenance,
        "adapter": {"reference": adapter_reference},
        "evaluation": _evaluation_payload(result, root=config.root),
    }
    # Scan the complete object before opening output files.
    _assert_no_secret_pattern(payload)
    # Standalone filenames cannot be confused with baseline-versus-tuned run reports.
    json_path, markdown_path = _unique_report_paths(
        config.report_dir,
        prefix="standalone-evaluation",
    )
    # JSON remains the exact machine-readable evidence.
    _write_json(json_path, payload)
    # Markdown contains the same full prompts and outputs for review.
    _write_text(markdown_path, _render_standalone_markdown(payload))
    # Log only repository-relative report paths and the already-public adapter reference.
    logger.event(
        "standalone_evaluation_report_written",
        json_report=_relative_public_path(json_path, config.root),
        markdown_report=_relative_public_path(markdown_path, config.root),
        adapter=adapter_reference,
    )
    # Standalone evaluation never creates or republishes an adapter.
    return ReportArtifacts(
        json_path=json_path,
        markdown_path=markdown_path,
        adapter_dir=None,
        json_sha256=_file_sha256(json_path),
        markdown_sha256=_file_sha256(markdown_path),
        adapter_file_sha256=None,
    )
