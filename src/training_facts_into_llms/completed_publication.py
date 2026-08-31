"""Publish a retrieved Qwen3.8 run without sending credentials to its GPU host.

The workflow has three explicit phases. ``upload`` runs beside the maintainer's
ignored ``.env`` and creates one immutable public model repository. ``verify``
uses only public identifiers and ``token=False`` on a compatible GPU. ``finalize``
returns to the credential-owning machine and appends the verified repository to
the dedicated Qwen3.8 Collection.

Sources:
- Exact allowlisted Hub uploads and optimistic parents:
  https://huggingface.co/docs/huggingface_hub/guides/upload
- Explicit anonymous Hub authentication:
  https://huggingface.co/docs/huggingface_hub/package_reference/authentication
- PEFT adapter loading:
  https://huggingface.co/docs/peft/package_reference/peft_model
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from training_facts_into_llms.archive_staging import _sha256

_SHA256: Final = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT: Final = re.compile(r"[0-9a-f]{40,64}\Z")
_MANIFEST_LINE: Final = re.compile(
    r"(?P<sha>[0-9a-f]{64})  (?P<path>(?:\./)?[A-Za-z0-9_.\-/]+)\Z"
)
_REQUEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "artifact_binding",
        "transfer_manifest_sha256",
        "acceptance_passed",
        "run_id",
        "experiment_id",
        "scientific_hash",
        "model_id",
        "model_revision",
        "quantization",
        "source_git_commit",
        "repository",
        "collection_note",
    }
)
_VERIFICATION_KEYS: Final = frozenset(
    {
        "schema_version",
        "request_sha256",
        "run_id",
        "experiment_id",
        "scientific_hash",
        "repo_id",
        "revision",
        "model_id",
        "model_revision",
        "quantization_mode",
        "messages",
        "rendered_prompt",
        "output",
        "nonempty",
        "runtime_evidence",
        "credential_free",
    }
)
_REPOSITORY_KEYS: Final = frozenset(
    {
        "repo_id",
        "repo_type",
        "decision",
        "revision",
        "public",
        "url",
        "files",
    }
)
_QUANTIZATION_KEYS: Final = frozenset(
    {"mode", "load_in_4bit", "quant_type", "double_quant", "compute_dtype"}
)
_PUBLIC_MODEL_FILES: Final = frozenset(
    {
        "LICENSE",
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "evaluation.json",
        "evaluation.md",
        "processor_reference.json",
        "run_manifest.json",
    }
)
_ARTIFACT_BINDING: Final = "retrieval-time-sha256-manifest"


@dataclass(frozen=True)
class CompletedPublicationRequest:
    """Carry one exact public repository from local upload to GPU verification."""

    schema_version: int
    artifact_binding: str
    transfer_manifest_sha256: str
    acceptance_passed: bool
    run_id: str
    experiment_id: str
    scientific_hash: str
    model_id: str
    model_revision: str
    quantization: Mapping[str, Any]
    source_git_commit: str
    repository: Mapping[str, Any]
    collection_note: str

    def to_dict(self) -> dict[str, Any]:
        """Return the complete path-free request written beside its digest."""
        return {
            "schema_version": self.schema_version,
            "artifact_binding": self.artifact_binding,
            "transfer_manifest_sha256": self.transfer_manifest_sha256,
            "acceptance_passed": self.acceptance_passed,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "scientific_hash": self.scientific_hash,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantization": dict(self.quantization),
            "source_git_commit": self.source_git_commit,
            "repository": dict(self.repository),
            "collection_note": self.collection_note,
        }


@dataclass(frozen=True)
class CompletedVerificationReceipt:
    """Record one anonymous revision-pinned adapter attach and generation."""

    schema_version: int
    request_sha256: str
    run_id: str
    experiment_id: str
    scientific_hash: str
    repo_id: str
    revision: str
    model_id: str
    model_revision: str
    quantization_mode: str
    messages: tuple[dict[str, str], ...]
    rendered_prompt: str
    output: str
    nonempty: bool
    runtime_evidence: Mapping[str, Any]
    credential_free: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a sanitized complete smoke-verification receipt."""
        return {
            "schema_version": self.schema_version,
            "request_sha256": self.request_sha256,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "scientific_hash": self.scientific_hash,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "quantization_mode": self.quantization_mode,
            "messages": [dict(message) for message in self.messages],
            "rendered_prompt": self.rendered_prompt,
            "output": self.output,
            "nonempty": self.nonempty,
            "runtime_evidence": dict(self.runtime_evidence),
            "credential_free": self.credential_free,
        }


@dataclass(frozen=True)
class ReceiptFiles:
    """Return one JSON receipt plus its portable SHA-256 companion."""

    json_path: Path
    sha256_path: Path
    sha256: str

    def to_dict(self, root: Path) -> dict[str, str]:
        """Represent both ignored paths relative to the project root."""
        return {
            "json_path": self.json_path.relative_to(root).as_posix(),
            "sha256_path": self.sha256_path.relative_to(root).as_posix(),
            "sha256": self.sha256,
        }


def _strict_json(path: Path) -> dict[str, Any]:
    """Read one finite JSON object without accepting NaN or infinity."""

    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonstandard JSON number is forbidden: {value}")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number is forbidden: {value}")
        return parsed

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            parsed[key] = value
        return parsed

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=parse_finite_float,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON receipt: {path.name}") from error
    if not isinstance(payload, dict):
        raise TypeError("publication receipt must be a JSON object")
    return payload


def _require_keys(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    """Return a mapping only when its entire public schema is reviewed."""
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} contains missing or unknown fields")
    return value


def parse_sha256_manifest(bundle_root: Path, manifest_path: Path) -> dict[str, str]:
    """Verify an extracted transfer bundle against its exact inner manifest."""
    root = bundle_root.expanduser().resolve()
    manifest = manifest_path.expanduser()
    manifest = manifest if manifest.is_absolute() else root / manifest
    manifest = manifest.resolve()
    try:
        manifest.relative_to(root)
    except ValueError as error:
        raise ValueError("SHA-256 manifest escapes its bundle") from error
    if not root.is_dir() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError("SHA-256 manifest or bundle is unavailable")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ValueError("SHA-256 manifest contains an invalid line")
        # GNU sha256sum preserves the spelling passed to it, so accept either
        # ``file`` or ``./file`` while normalizing both to one safe relative key.
        relative = match.group("path").removeprefix("./")
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or relative in expected
        ):
            raise ValueError("SHA-256 manifest contains an unsafe or duplicate path")
        expected[relative] = match.group("sha")
    if not expected:
        raise ValueError("SHA-256 manifest is empty")
    actual: set[str] = set()
    total_size = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("retrieved bundle contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("retrieved bundle contains a special file")
        if path == manifest:
            continue
        relative = path.relative_to(root).as_posix()
        actual.add(relative)
        total_size += path.stat().st_size
    if actual != set(expected):
        raise ValueError("retrieved bundle file set differs from its SHA-256 manifest")
    if total_size > 5 * 1024**3:
        raise ValueError("retrieved bundle exceeds the 5 GiB publication ceiling")
    for relative, digest in expected.items():
        path = root / relative
        if _sha256(path) != digest:
            raise ValueError(f"retrieved bundle digest differs: {relative}")
    return expected


def validate_publication_request(request: CompletedPublicationRequest) -> None:
    """Validate one path-free request before anonymous network or GPU activity."""
    if request.schema_version != 1:
        raise ValueError("completed publication request schema is unsupported")
    if request.artifact_binding != _ARTIFACT_BINDING:
        raise ValueError("completed publication artifact binding is unsupported")
    if not _SHA256.fullmatch(request.transfer_manifest_sha256):
        raise ValueError("completed publication transfer manifest digest is invalid")
    if not isinstance(request.acceptance_passed, bool):
        raise TypeError("completed publication acceptance result must be a bool")
    for value, label in (
        (request.run_id, "run ID"),
        (request.experiment_id, "experiment ID"),
        (request.model_id, "model ID"),
        (request.collection_note, "Collection note"),
    ):
        if not isinstance(value, str) or not value:
            raise TypeError(f"completed publication {label} must be nonempty text")
    if len(request.collection_note) > 500:
        raise ValueError("completed publication Collection note is too long")
    if not _SHA256.fullmatch(request.scientific_hash):
        raise ValueError("completed publication scientific hash is invalid")
    if not _COMMIT.fullmatch(request.model_revision) or not _COMMIT.fullmatch(
        request.source_git_commit
    ):
        raise ValueError("completed publication revision is invalid")
    quantization = _require_keys(
        dict(request.quantization),
        _QUANTIZATION_KEYS,
        "completed publication quantization",
    )
    if quantization["mode"] not in {"none", "bnb_nf4"}:
        raise ValueError("completed publication quantization mode is unsupported")
    repository = _require_keys(
        dict(request.repository),
        _REPOSITORY_KEYS,
        "completed publication repository",
    )
    if (
        repository["repo_type"] != "model"
        or repository["decision"] not in {"create", "repair", "skip"}
        or repository["public"] is not True
        or not _COMMIT.fullmatch(str(repository["revision"]))
        or not isinstance(repository["files"], dict)
        or set(repository["files"]) != _PUBLIC_MODEL_FILES
    ):
        raise ValueError("completed publication repository is inconsistent")
    if any(
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        for path, digest in repository["files"].items()
    ):
        raise ValueError("completed publication repository hashes are invalid")
    serialized = json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True)
    if "HF_TOKEN" in serialized or re.search(
        r"(?:^|[\s\"'])/(?:home|root)/",
        serialized,
    ):
        raise ValueError("completed publication request contains private host metadata")


def publication_request_from_dict(
    payload: Mapping[str, Any],
) -> CompletedPublicationRequest:
    """Parse one exact request mapping into its immutable public contract."""
    request_data = _require_keys(
        dict(payload),
        _REQUEST_KEYS,
        "completed publication request",
    )
    request = CompletedPublicationRequest(**request_data)
    validate_publication_request(request)
    return request


def verification_receipt_from_dict(
    payload: Mapping[str, Any],
) -> CompletedVerificationReceipt:
    """Parse one exact GPU receipt without reflecting arbitrary runtime objects."""
    data = _require_keys(
        dict(payload),
        _VERIFICATION_KEYS,
        "completed verification receipt",
    )
    raw_messages = data["messages"]
    if not isinstance(raw_messages, list) or not all(
        isinstance(message, dict)
        and set(message) == {"role", "content"}
        and all(isinstance(value, str) for value in message.values())
        for message in raw_messages
    ):
        raise ValueError("completed verification messages are invalid")
    data["messages"] = tuple(dict(message) for message in raw_messages)
    receipt = CompletedVerificationReceipt(**data)
    if receipt.schema_version != 1:
        raise ValueError("completed verification schema is unsupported")
    return receipt


def validate_verification_receipt(
    request: CompletedPublicationRequest,
    request_sha256: str,
    receipt: CompletedVerificationReceipt,
) -> None:
    """Require one nonempty credential-free receipt for this exact request."""
    if not _SHA256.fullmatch(request_sha256):
        raise ValueError("completed publication request digest is invalid")
    repository = request.repository
    expected = (
        request_sha256,
        request.run_id,
        request.experiment_id,
        request.scientific_hash,
        repository["repo_id"],
        repository["revision"],
        request.model_id,
        request.model_revision,
        request.quantization["mode"],
    )
    actual = (
        receipt.request_sha256,
        receipt.run_id,
        receipt.experiment_id,
        receipt.scientific_hash,
        receipt.repo_id,
        receipt.revision,
        receipt.model_id,
        receipt.model_revision,
        receipt.quantization_mode,
    )
    if actual != expected:
        raise ValueError("completed verification identity differs from its request")
    if receipt.credential_free is not True:
        raise ValueError("completed verification was not credential-free")
    if receipt.nonempty is not True or not receipt.output.strip():
        raise ValueError("completed verification did not return a nonempty output")
    if not receipt.rendered_prompt:
        raise ValueError("completed verification rendered prompt is empty")


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Render deterministic receipt bytes for portable companion hashes."""
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _receipt_destination(config: Any, stem: str, requested: Path | None) -> Path:
    """Resolve one fresh ignored receipt path below the configured artifact root."""
    root = Path(config.root).resolve()
    artifact_root = Path(config.artifact_dir).resolve()
    directory = artifact_root / "completed-publication"
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}.json" if requested is None else requested
    candidate = candidate if candidate.is_absolute() else root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("publication receipt must remain in ARTIFACT_DIR") from error
    if (
        candidate.exists()
        or candidate.with_suffix(candidate.suffix + ".sha256").exists()
    ):
        raise ValueError("publication receipt output already exists")
    return candidate


def write_receipt(
    config: Any,
    stem: str,
    payload: Mapping[str, Any],
    *,
    output: Path | None = None,
) -> ReceiptFiles:
    """Write one exclusive JSON receipt and a same-directory digest companion."""
    destination = _receipt_destination(config, stem, output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(payload)
    with destination.open("xb") as handle:
        handle.write(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    sha_path = destination.with_suffix(destination.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return ReceiptFiles(destination, sha_path, digest)


def read_hashed_receipt(
    json_path: Path,
    sha256_path: Path,
) -> tuple[dict[str, Any], str]:
    """Read one transferred receipt only after its companion digest verifies."""
    document = json_path.expanduser().resolve()
    companion = sha256_path.expanduser().resolve()
    if document.is_symlink() or companion.is_symlink():
        raise ValueError("publication receipt files cannot be symlinks")
    if not document.is_file() or not companion.is_file():
        raise ValueError("publication receipt or digest file is unavailable")
    line = companion.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
    if match is None or match.group(2) != document.name:
        raise ValueError("publication receipt digest companion is invalid")
    digest = _sha256(document)
    if digest != match.group(1):
        raise ValueError("publication receipt digest differs")
    return _strict_json(document), digest


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    """Require the training source commit to remain in reviewed public history."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("completed run source is not an ancestor of reviewed main")


def _remove_hub_credentials_from_environment() -> None:
    """Prevent inherited credentials from entering the anonymous verification phase."""
    for name in tuple(os.environ):
        folded = name.casefold()
        if folded in {"hf_token", "hugging_face_hub_token", "huggingface_token"}:
            del os.environ[name]
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def _bundle_path(
    bundle_root: Path,
    relative: Path,
    *,
    directory: bool,
) -> Path:
    """Resolve one explicit bundle-relative path without following a symlink."""
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("completed bundle paths must be safe relative paths")
    root = bundle_root.resolve()
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("completed bundle paths cannot contain symlinks")
    resolved = cursor.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("completed bundle path escapes its root") from error
    if directory and not resolved.is_dir():
        raise ValueError("completed bundle adapter directory is unavailable")
    if not directory and not resolved.is_file():
        raise ValueError("completed bundle report file is unavailable")
    return resolved


def _quantization_payload(experiment: Any) -> dict[str, Any]:
    """Return the complete reviewed quantization record from sanitized science."""
    sanitized = experiment.sanitized()
    configuration = sanitized.get("configuration")
    quantization = (
        configuration.get("quantization") if isinstance(configuration, dict) else None
    )
    return _require_keys(
        quantization,
        _QUANTIZATION_KEYS,
        "resolved completed publication quantization",
    )


def _recompute_report_decision(
    config: Any,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Re-score every saved output and return the trusted decision and report."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.scoring import (
        validate_acceptance_decision,
        validate_score_result,
    )
    from training_facts_into_llms.scoring_loader import (
        load_scoring_plugin,
        scoring_implementation_sha256,
    )

    report = _strict_json(report_path)
    experiment = config.experiment
    if experiment is None:
        raise RuntimeError("completed publication requires a resolved experiment")
    configuration = report.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("experiment") != experiment.sanitized()
    ):
        raise ValueError("completed report differs from the resolved experiment")
    scorer, source = load_scoring_plugin(
        config.root,
        experiment.scoring.plugin,
        scoring_options=experiment.scoring.options,
        acceptance_options=experiment.acceptance.options,
        expected_source_sha256=experiment.scoring.canonical_source_sha256,
    )
    plugin_hash = scoring_implementation_sha256(
        config.root,
        experiment.scoring.plugin,
        source,
    )
    provenance = report.get("provenance")
    report_source = provenance.get("source") if isinstance(provenance, dict) else None
    reported_plugin = (
        report_source.get("scoring_plugin") if isinstance(report_source, dict) else None
    )
    if not isinstance(reported_plugin, dict) or reported_plugin != {
        "path": source.relative_to(config.root).as_posix(),
        "sha256": plugin_hash,
    }:
        raise ValueError("completed report scorer differs from reviewed source")
    data = load_experiment_data(experiment)
    validate_experiment_data(data, experiment)
    cases = tuple(data.evaluation)
    evaluations = report.get("evaluations")
    if not isinstance(evaluations, dict) or set(evaluations) != {
        "baseline",
        "post_training",
    }:
        raise ValueError("completed report evaluations are incomplete")
    recomputed: dict[str, Any] = {}
    for phase in ("baseline", "post_training"):
        saved = evaluations[phase]
        if not isinstance(saved, dict) or not isinstance(saved.get("records"), list):
            raise TypeError("completed report evaluation records are invalid")
        saved_records = saved["records"]
        if [record.get("record_id") for record in saved_records] != [
            case["id"] for case in cases
        ]:
            raise ValueError("completed report evaluation record order differs")
        generations = [record.get("output") for record in saved_records]
        if not all(isinstance(output, str) for output in generations):
            raise TypeError("completed report generation is not text")
        scored = validate_score_result(
            scorer.score(cases, generations, phase=phase),
            cases,
            generations,
            phase=phase,
        )
        if scored.to_dict() != saved:
            raise ValueError("completed report evaluation differs from rescoring")
        recomputed[phase] = scored
    decision = validate_acceptance_decision(
        scorer.decide(recomputed["baseline"], recomputed["post_training"])
    )
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict):
        raise TypeError("completed report acceptance is unavailable")
    derived = {
        "canonical_scientific_configuration",
        "canonical_scoring_plugin_source",
        "canonical_approval",
        "outcome_label",
    }
    decision_payload = decision.to_dict()
    if {key: value for key, value in acceptance.items() if key not in derived} != (
        decision_payload
    ):
        raise ValueError("completed report acceptance differs from recomputation")
    return decision_payload, report, dict(report_source)


def _source_gate_and_ancestry(
    config: Any,
    source_git_commit: str,
    source_gate: Callable[[Any], Any],
    ancestry_checker: Callable[[Path, str, str], None] = _git_is_ancestor,
) -> str:
    """Run the merged-source gate and retain the report-producing ancestor."""
    gate = source_gate(config)
    current_commit = getattr(gate, "commit", gate)
    if not isinstance(current_commit, str) or not _COMMIT.fullmatch(current_commit):
        raise RuntimeError("completed publication source gate returned no commit")
    if not _COMMIT.fullmatch(source_git_commit):
        raise ValueError("completed report source commit is invalid")
    ancestry_checker(config.root, source_git_commit, current_commit)
    return current_commit


def _request_matches_experiment(
    request: CompletedPublicationRequest, experiment: Any
) -> None:
    """Reconcile every public model/science field against one registry preset."""
    from training_facts_into_llms.archive_inventory import completed_publication_family
    from training_facts_into_llms.experiments import (
        COMPLETED_PUBLICATION_EXPERIMENT_IDS,
    )

    if (
        request.experiment_id not in COMPLETED_PUBLICATION_EXPERIMENT_IDS
        or experiment.experiment_id not in COMPLETED_PUBLICATION_EXPERIMENT_IDS
        or request.experiment_id != experiment.experiment_id
        or request.scientific_hash != experiment.scientific_hash
        or request.model_id != experiment.model.model_id
        or request.model_revision != experiment.model.model_revision
        or dict(request.quantization) != _quantization_payload(experiment)
    ):
        raise ValueError("completed publication request differs from its experiment")
    completed_publication_family(request.model_id, request.model_revision)


def _request_matches_destination(
    request: CompletedPublicationRequest,
    experiment: Any,
    namespace: str,
) -> None:
    """Derive the repository and Collection note instead of trusting a receipt."""
    from training_facts_into_llms.archive_inventory import (
        completed_publication_family,
        repo_id_for_run,
    )
    from training_facts_into_llms.archive_publishing import validate_future_run_identity

    family = completed_publication_family(request.model_id, request.model_revision)
    validate_future_run_identity(request.run_id, experiment.sanitized())
    expected_repo_id = repo_id_for_run(
        namespace,
        request.run_id,
        prefix=family.repository_prefix,
    )
    expected_url = f"https://huggingface.co/{expected_repo_id}"
    outcome = "passed" if request.acceptance_passed else "failed"
    expected_note = (
        f"Completed run {request.run_id} for {request.experiment_id}; configured "
        f"acceptance {outcome}. Full evaluation is included in the model repository."
    )
    repository = request.repository
    if (
        repository["repo_id"] != expected_repo_id
        or repository["url"] != expected_url
        or request.collection_note != expected_note
    ):
        raise ValueError(
            "completed publication destination differs from reviewed identity"
        )


def _verify_public_repository(
    request: CompletedPublicationRequest,
) -> None:
    """Require exact public bytes at both immutable revision and repository main."""
    from huggingface_hub import HfApi

    repository = request.repository
    repo_id = str(repository["repo_id"])
    revision = str(repository["revision"])
    api = HfApi(token=False)
    pinned = api.model_info(repo_id, revision=revision, token=False)
    current = api.model_info(repo_id, token=False)
    if (
        getattr(pinned, "sha", None) != revision
        or getattr(current, "sha", None) != revision
        or bool(getattr(current, "private", True))
        or bool(getattr(current, "gated", False))
    ):
        raise RuntimeError(
            "completed publication repository revision or visibility changed"
        )
    snapshot = Path(api.snapshot_download(repo_id, revision=revision, token=False))
    actual = {
        path.relative_to(snapshot).as_posix(): _sha256(path)
        for path in snapshot.rglob("*")
        if path.is_file() and path.relative_to(snapshot).as_posix() != ".gitattributes"
    }
    if actual != dict(repository["files"]):
        raise RuntimeError("completed publication public repository bytes differ")


def upload_completed_publication(
    config: Any,
    *,
    bundle_root: Path,
    sha256_manifest: Path,
    adapter: Path,
    report_json: Path,
    report_markdown: Path,
    output: Path | None = None,
    source_gate: Callable[[Any], Any] | None = None,
    ancestry_checker: Callable[[Path, str, str], None] = _git_is_ancestor,
    hub: Any | None = None,
    credential_loader: Callable[[Path], str] | None = None,
    audit_adapter: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit a retrieved completed run, upload exact bytes, and emit a GPU request."""
    from training_facts_into_llms.archive_inventory import (
        completed_publication_family,
    )
    from training_facts_into_llms.archive_publishing import (
        HuggingFaceArchiveHub,
        _allocate_staging_directory,
        synchronize_repository,
        validate_future_run_identity,
        validate_publication_credential,
    )
    from training_facts_into_llms.archive_staging import (
        CompletedRunContext,
        stage_completed_run_repository,
    )
    from training_facts_into_llms.experiments import (
        COMPLETED_PUBLICATION_EXPERIMENT_IDS,
    )
    from training_facts_into_llms.git_gate import enforce_git_before_training
    from training_facts_into_llms.logging_utils import timestamp_id

    experiment = config.experiment
    if experiment is None:
        raise RuntimeError("completed upload requires a resolved experiment")
    if experiment.experiment_id not in COMPLETED_PUBLICATION_EXPERIMENT_IDS:
        raise ValueError("post-run publication is not authorized for this experiment")
    family = completed_publication_family(
        experiment.model.model_id,
        experiment.model.model_revision,
    )
    active_gate = source_gate or enforce_git_before_training
    root = Path(config.root).resolve()
    bundle = bundle_root if bundle_root.is_absolute() else root / bundle_root
    bundle = bundle.resolve()
    try:
        bundle.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "completed bundle must remain within the repository"
        ) from error
    artifact_root = Path(config.artifact_dir).resolve()
    try:
        bundle.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("completed bundle must remain within ARTIFACT_DIR") from error
    if bundle == artifact_root:
        raise ValueError("completed bundle must be a dedicated ARTIFACT_DIR child")
    if output is not None:
        requested_output = output if output.is_absolute() else root / output
        try:
            requested_output.resolve().relative_to(bundle)
        except ValueError:
            pass
        else:
            raise ValueError(
                "publication request cannot be written into its source bundle"
            )
    manifest = sha256_manifest
    manifest = manifest if manifest.is_absolute() else bundle / manifest
    manifest_hashes = parse_sha256_manifest(bundle, manifest)
    transfer_manifest_digest = _sha256(manifest)
    adapter_path = _bundle_path(bundle, adapter, directory=True)
    json_path = _bundle_path(bundle, report_json, directory=False)
    markdown_path = _bundle_path(bundle, report_markdown, directory=False)
    if json_path.suffix != ".json" or markdown_path.suffix != ".md":
        raise ValueError("completed report pair has unexpected suffixes")
    # Read only inert JSON before the merged-source gate; plugin import happens later.
    untrusted_report = _strict_json(json_path)
    untrusted_provenance = untrusted_report.get("provenance")
    untrusted_source = (
        untrusted_provenance.get("source")
        if isinstance(untrusted_provenance, dict)
        else None
    )
    source_commit = (
        untrusted_source.get("git_commit")
        if isinstance(untrusted_source, dict)
        else None
    )
    if not isinstance(source_commit, str):
        raise TypeError("completed report source commit is unavailable")
    _source_gate_and_ancestry(
        config,
        source_commit,
        active_gate,
        ancestry_checker,
    )
    decision, report, source = _recompute_report_decision(config, json_path)
    if source.get("git_commit") != source_commit:
        raise ValueError("completed report source changed during its source audit")
    identity = report.get("provenance", {}).get("run_identity", {})
    run_id = identity.get("run_id") if isinstance(identity, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("completed report run identity is unavailable")
    validate_future_run_identity(run_id, experiment.sanitized())
    expected_artifacts = {
        "report_json": json_path,
        "report_markdown": markdown_path,
        **{
            f"adapter/{name}": adapter_path / name
            for name in (
                "README.md",
                "adapter_config.json",
                "adapter_model.safetensors",
                "evaluation.json",
                "processor_reference.json",
            )
        },
    }
    artifact_hashes: dict[str, str] = {}
    for key, path in expected_artifacts.items():
        relative = path.relative_to(bundle).as_posix()
        manifest_digest = manifest_hashes.get(relative)
        if manifest_digest is None or manifest_digest != _sha256(path):
            raise ValueError("completed publication artifact is not manifest-bound")
        artifact_hashes[key] = manifest_digest
    destination = _allocate_staging_directory(
        config,
        prefix="qwen38-completed-hub-",
        requested=None,
    )
    staged = stage_completed_run_repository(
        root,
        destination,
        adapter_path,
        namespace=config.hf_namespace,
        context=CompletedRunContext(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            experiment=experiment.sanitized(),
            acceptance=decision,
            artifact_hashes=artifact_hashes,
            artifact_binding={
                "kind": _ARTIFACT_BINDING,
                "manifest_sha256": transfer_manifest_digest,
            },
        ),
        report_json=json_path,
        report_markdown=markdown_path,
        model_id=config.model_id,
        model_revision=config.model_revision,
        lora_config=experiment.config.lora,
        audit_adapter=audit_adapter,
        repository_prefix=family.repository_prefix,
    )
    if (
        _sha256(manifest) != transfer_manifest_digest
        or parse_sha256_manifest(bundle, manifest) != manifest_hashes
    ):
        raise ValueError("completed transfer bundle changed during staging")
    load_credential = credential_loader or validate_publication_credential
    secret = load_credential(root)
    archive_hub = hub if hub is not None else HuggingFaceArchiveHub(secret)
    try:
        repository_receipt = synchronize_repository(
            staged,
            hub=archive_hub,
            secret=secret,
        )
    finally:
        secret = ""
    request = CompletedPublicationRequest(
        schema_version=1,
        artifact_binding=_ARTIFACT_BINDING,
        transfer_manifest_sha256=transfer_manifest_digest,
        acceptance_passed=bool(decision["passed"]),
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        scientific_hash=experiment.scientific_hash,
        model_id=config.model_id,
        model_revision=config.model_revision,
        quantization=_quantization_payload(experiment),
        source_git_commit=source_commit,
        repository=repository_receipt.to_dict(),
        collection_note=staged.collection_note,
    )
    validate_publication_request(request)
    files = write_receipt(
        config,
        f"{run_id}-request-{timestamp_id()}",
        request.to_dict(),
        output=output,
    )
    return {
        "phase": "upload",
        "repository": repository_receipt.to_dict(),
        "request": files.to_dict(root),
    }


def verify_completed_publication(
    config: Any,
    *,
    request_path: Path,
    request_sha256_path: Path,
    output: Path | None = None,
    source_gate: Callable[[Any], Any] | None = None,
    ancestry_checker: Callable[[Path, str, str], None] = _git_is_ancestor,
    public_repository_verifier: Callable[[CompletedPublicationRequest], None]
    | None = None,
    model_loader: Callable[..., Any] | None = None,
    generator: Callable[..., tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Anonymously attach one exact public adapter with its resolved base load plan."""
    # Remove inherited Hub credentials before importing any model/Hub helper.
    _remove_hub_credentials_from_environment()
    from training_facts_into_llms.archive_verification import (
        SMOKE_MAX_NEW_TOKENS,
        SMOKE_MESSAGES,
    )
    from training_facts_into_llms.experiments import resolve_experiment
    from training_facts_into_llms.git_gate import enforce_git_before_training
    from training_facts_into_llms.logging_utils import EventLogger, timestamp_id
    from training_facts_into_llms.modeling import (
        generate_response,
        load_adapter_model,
        release_model,
    )

    request_payload, request_digest = read_hashed_receipt(
        request_path,
        request_sha256_path,
    )
    request = publication_request_from_dict(request_payload)
    experiment = resolve_experiment(config.root, request.experiment_id)
    _request_matches_experiment(request, experiment)
    config = config.with_experiment(experiment)
    _request_matches_destination(request, experiment, config.hf_namespace)
    active_gate = source_gate or enforce_git_before_training
    _source_gate_and_ancestry(
        config,
        request.source_git_commit,
        active_gate,
        ancestry_checker,
    )
    (public_repository_verifier or _verify_public_repository)(request)
    load = model_loader or load_adapter_model
    generate = generator or generate_response
    bundle = None
    run_id = f"{timestamp_id()}-completed-publication-verify"
    with EventLogger(config.log_dir, run_id=run_id) as logger:
        logger.event(
            "completed_publication_verification_started",
            request_sha256=request_digest,
            repository=request.repository["repo_id"],
            revision=request.repository["revision"],
            experiment_id=request.experiment_id,
        )
        try:
            bundle = load(
                config,
                request.repository["repo_id"],
                logger=logger,
                adapter_log_reference=request.repository["repo_id"],
                revision=request.repository["revision"],
            )
            output_text, rendered = generate(
                bundle,
                [dict(message) for message in SMOKE_MESSAGES],
                max_new_tokens=SMOKE_MAX_NEW_TOKENS,
                generation=experiment.config.generation,
            )
            if not output_text.strip():
                raise RuntimeError(
                    "public Qwen3.8 adapter produced an empty generation"
                )
            runtime_evidence = dict(getattr(bundle, "runtime_evidence", None) or {})
            receipt = CompletedVerificationReceipt(
                schema_version=1,
                request_sha256=request_digest,
                run_id=request.run_id,
                experiment_id=request.experiment_id,
                scientific_hash=request.scientific_hash,
                repo_id=request.repository["repo_id"],
                revision=request.repository["revision"],
                model_id=request.model_id,
                model_revision=request.model_revision,
                quantization_mode=request.quantization["mode"],
                messages=SMOKE_MESSAGES,
                rendered_prompt=rendered,
                output=output_text,
                nonempty=True,
                runtime_evidence=runtime_evidence,
                credential_free=True,
            )
            validate_verification_receipt(request, request_digest, receipt)
            logger.event(
                "completed_publication_verification_completed",
                receipt=receipt.to_dict(),
            )
        finally:
            release_model(bundle)
    files = write_receipt(
        config,
        f"{request.run_id}-verification-{timestamp_id()}",
        receipt.to_dict(),
        output=output,
    )
    return {
        "phase": "verify",
        "repository": request.repository["url"],
        "verification": files.to_dict(Path(config.root).resolve()),
    }


def finalize_completed_publication(
    config: Any,
    *,
    request_path: Path,
    request_sha256_path: Path,
    verification_path: Path,
    verification_sha256_path: Path,
    output: Path | None = None,
    source_gate: Callable[[Any], Any] | None = None,
    ancestry_checker: Callable[[Path, str, str], None] = _git_is_ancestor,
    public_repository_verifier: Callable[[CompletedPublicationRequest], None]
    | None = None,
    hub: Any | None = None,
    credential_loader: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Validate anonymous GPU evidence, then append one Qwen3.8 Collection item."""
    from training_facts_into_llms.archive_inventory import completed_publication_family
    from training_facts_into_llms.archive_publishing import (
        HuggingFaceArchiveHub,
        append_model_to_collection,
        validate_publication_credential,
    )
    from training_facts_into_llms.experiments import resolve_experiment
    from training_facts_into_llms.git_gate import enforce_git_before_training
    from training_facts_into_llms.logging_utils import timestamp_id

    request_payload, request_digest = read_hashed_receipt(
        request_path,
        request_sha256_path,
    )
    verification_payload, verification_digest = read_hashed_receipt(
        verification_path,
        verification_sha256_path,
    )
    request = publication_request_from_dict(request_payload)
    receipt = verification_receipt_from_dict(verification_payload)
    validate_verification_receipt(request, request_digest, receipt)
    experiment = resolve_experiment(config.root, request.experiment_id)
    _request_matches_experiment(request, experiment)
    config = config.with_experiment(experiment)
    _request_matches_destination(request, experiment, config.hf_namespace)
    active_gate = source_gate or enforce_git_before_training
    _source_gate_and_ancestry(
        config,
        request.source_git_commit,
        active_gate,
        ancestry_checker,
    )
    (public_repository_verifier or _verify_public_repository)(request)
    kernel = receipt.runtime_evidence.get("kernel")
    if (
        not isinstance(kernel, dict)
        or kernel.get("required") is not True
        or kernel.get("executed") is not True
    ):
        raise ValueError("completed verification lacks the accelerated kernel proof")
    family = completed_publication_family(request.model_id, request.model_revision)
    load_credential = credential_loader or validate_publication_credential
    secret = load_credential(Path(config.root).resolve())
    archive_hub = hub if hub is not None else HuggingFaceArchiveHub(secret)
    try:
        collection = append_model_to_collection(
            repo_id=request.repository["repo_id"],
            note=request.collection_note,
            namespace=config.hf_namespace,
            title=family.collection_title,
            description=family.collection_description,
            hub=archive_hub,
        )
    finally:
        secret = ""
    final = {
        "schema_version": 1,
        "publication_kind": "completed_qwen38_lora",
        "request_sha256": request_digest,
        "verification_sha256": verification_digest,
        "repository": dict(request.repository),
        "verification": receipt.to_dict(),
        "collection": collection.to_dict(),
    }
    files = write_receipt(
        config,
        f"{request.run_id}-final-{timestamp_id()}",
        final,
        output=output,
    )
    return {
        "phase": "finalize",
        "repository": request.repository["url"],
        "collection": collection.to_dict(),
        "receipt": files.to_dict(Path(config.root).resolve()),
    }
