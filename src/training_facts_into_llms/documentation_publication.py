"""Publish one additive documentation edition without touching model evidence.

The reviewed parent contract owns remote scope. Local documents must match
committed source; a new receipt records this later publication independently.
Sources: https://huggingface.co/docs/huggingface_hub/guides/upload
https://git-scm.com/docs/git-show
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from training_facts_into_llms.credentials import contains_credential_text
from training_facts_into_llms.logging_utils import EventLogger, timestamp_id
from training_facts_into_llms.results import _safe_path
from training_facts_into_llms.results_evidence import (
    _parse_json,
    _read_evidence_file,
    _source_revision,
)

EDITION = "audit-2026-10-29"
CONTRACT_PATH = f"configs/documentation/{EDITION}.json"
SOURCE_DOCUMENTS = (
    "reports/INSTRUCTION_AUDIT.md",
    "reports/ALL_EXPERIMENTS.md",
    "docs/third-party-provenance.md",
    "output/pdf/teaching-one-synthetic-fact-qwen35-revised.pdf",
    "output/pdf/teaching-one-synthetic-fact-qwen38-minimal.pdf",
)
DESTINATIONS = (
    ("historical", "BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence", "dataset"),
    (
        "qwen38",
        "BurnyCoder/qwen3.8-27b-atemokoloporos-20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff",
        "model",
    ),
)
_GITHUB = "https://github.com/BurnyCoder/training-facts-into-llms/blob/"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class DocumentationRepository:
    """Hold only the exact public bytes and original remote objects to reconcile."""

    label: str
    repo_id: str
    repo_type: str
    parent_commit: str
    original_tree: dict[str, dict[str, Any]]
    files: dict[str, bytes]
    sources: dict[str, dict[str, str]]


@dataclass(frozen=True)
class DocumentationPlan:
    """Separate local source validation from all operational publication work."""

    source_commit: str
    contract_sha256: str
    repositories: tuple[DocumentationRepository, ...]


def _relative_path(value: str) -> None:
    """Reject path aliases rather than normalizing an upload destination."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("invalid documentation path")
    if path.as_posix() != value:
        raise ValueError("noncanonical documentation path")


def _validate_tree(tree: Any) -> None:
    """Require only immutable Git/LFS identity fields, never raw API objects."""
    if not isinstance(tree, dict) or not tree:
        raise ValueError("documentation tree must be nonempty")
    for path, item in tree.items():
        _relative_path(path)
        if set(item) != {"blob_id", "size", "lfs_sha256"}:
            raise ValueError("unexpected documentation tree fields")
        if not _COMMIT.fullmatch(item["blob_id"]):
            raise ValueError("invalid Git object identity")
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError("invalid public file size")
        if item["lfs_sha256"] is not None and not _DIGEST.fullmatch(item["lfs_sha256"]):
            raise ValueError("invalid LFS object identity")


def _project_markdown(root: Path, raw: bytes, source: str, revision: str) -> bytes:
    """Keep relative document links usable through their immutable GitHub source."""
    source_url = f"{_GITHUB}{revision}/{source}"
    text = raw.decode("utf-8")
    if contains_credential_text(text):
        raise ValueError("credential-shaped content in documentation")
    targets: dict[str, str] = {}

    def destination(match: re.Match[str]) -> str:
        target = match[2]
        parsed = urlsplit(target)
        if target.startswith("#") or parsed.scheme or parsed.netloc:
            return match[0]
        relative = posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(parsed.path)))
        _relative_path(relative)
        if relative not in targets:
            # A linked file may have been introduced after the referring document.
            # Pin its own committed bytes so later unrelated commits do not alter retries.
            target_bytes = _read_evidence_file(root, relative)
            targets[relative] = _source_revision(root, relative, target_bytes)
        resolved = urlunsplit((
            "https", "github.com",
            f"/BurnyCoder/training-facts-into-llms/blob/{targets[relative]}/{quote(relative, safe='/')}",
            parsed.query, parsed.fragment,
        ))
        return match[1] + resolved + match[3]

    projected = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = marker if fence is None else None if fence == marker else fence
            projected.append(line)
            continue
        if fence is None:
            line = re.sub(r"(!?\[[^\]\n]*\]\()([^\s)]+)(\))", destination, line)
            line = re.sub(r"^(\s{0,3}\[[^\]]+\]:\s+)(\S+)(.*)$", destination, line)
        projected.append(line)
    return (f"[Immutable source document]({source_url})\n\n" + "".join(projected)).encode()


def prepare_documentation(root: Path, edition: str) -> DocumentationPlan:
    """Validate committed inputs without reading .env, creating files, or networking."""
    if edition != EDITION:
        raise ValueError("unknown documentation edition")
    root = root.resolve()
    raw_contract = _read_evidence_file(root, CONTRACT_PATH)
    _source_revision(root, CONTRACT_PATH, raw_contract)
    contract = _parse_json(raw_contract, CONTRACT_PATH)
    if set(contract) != {"schema_version", "edition", "repositories"}:
        raise ValueError("unexpected documentation contract fields")
    if contract["schema_version"] != 1 or contract["edition"] != edition:
        raise ValueError("invalid documentation contract version")
    if len(contract["repositories"]) != len(DESTINATIONS):
        raise ValueError("unexpected documentation destinations")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    if not _COMMIT.fullmatch(commit):
        raise ValueError("invalid documentation source commit")
    sources: dict[str, dict[str, str]] = {}
    documents = {}
    for source in SOURCE_DOCUMENTS:
        raw = _read_evidence_file(root, source)
        revision = _source_revision(root, source, raw)
        sources[source] = {"sha256": hashlib.sha256(raw).hexdigest(), "revision": revision}
        if source.endswith(".md"):
            documents[source] = _project_markdown(root, raw, source, revision)
        else:
            if not raw.startswith(b"%PDF-"):
                raise ValueError("documentation PDF is invalid")
            documents[source] = raw
    repositories = []
    for index, (entry, identity) in enumerate(zip(contract["repositories"], DESTINATIONS, strict=True)):
        if set(entry) != {
            "label", "repo_id", "repo_type", "parent_commit", "tree", "readme", "readme_sha256",
            "gitattributes", "gitattributes_sha256",
        }:
            raise ValueError("unexpected documentation repository contract")
        if tuple(entry[key] for key in ("label", "repo_id", "repo_type")) != identity:
            raise ValueError("unreviewed documentation destination")
        if not _COMMIT.fullmatch(entry["parent_commit"]):
            raise ValueError("invalid documentation parent commit")
        _validate_tree(entry["tree"])
        for path, field in (("README.md", "readme"), (".gitattributes", "gitattributes")):
            original = entry[field].encode()
            if hashlib.sha256(original).hexdigest() != entry[field + "_sha256"]:
                raise ValueError("original documentation text digest differs")
            blob = hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest()
            if entry["tree"][path] != {
                "blob_id": blob, "size": len(original), "lfs_sha256": None,
            }:
                raise ValueError("original documentation text object differs")
        readme = entry["readme"].encode()
        selected = (*SOURCE_DOCUMENTS[:3], SOURCE_DOCUMENTS[3 + index])
        files = {
            f"documentation/{edition}/{Path(source).name}": documents[source]
            for source in selected
        }
        links = "\n".join(f"- [{Path(path).name}]({path})" for path in files)
        # The Hub automatically registers LFS files. Supply the exact one-rule
        # addition explicitly so metadata changes are reviewed and byte-verified.
        # Source: https://huggingface.co/.well-known/openapi.json (preupload gitAttributes).
        attributes = entry["gitattributes"]
        if not attributes.endswith("\n"):
            raise ValueError("reviewed attributes must end with a newline")
        pdf_path = f"documentation/{edition}/{Path(selected[-1]).name}"
        files[".gitattributes"] = (
            attributes + f"{pdf_path} filter=lfs diff=lfs merge=lfs -text\n"
        ).encode()
        addendum = (
            f"\n\n## Documentation revision: {edition}\n\n"
            "These later explanatory documents use the existing admitted evidence. "
            "They do not replace the original reports, model objects, acceptance "
            "decisions, or publication receipts.\n\n" + links + "\n"
        )
        files["README.md"] = readme + addendum.encode()
        if contains_credential_text(files["README.md"].decode()):
            raise ValueError("credential-shaped content in README projection")
        repositories.append(DocumentationRepository(
            label=entry["label"], repo_id=entry["repo_id"], repo_type=entry["repo_type"],
            parent_commit=entry["parent_commit"], original_tree=entry["tree"], files=files,
            sources={source: sources[source] for source in selected},
        ))
    return DocumentationPlan(commit, hashlib.sha256(raw_contract).hexdigest(), tuple(repositories))


def _require_snapshot(snapshot: dict[str, Any]) -> None:
    """Check public visibility and strict object identities at every remote read."""
    if snapshot["private"] is not False or snapshot["gated"] is not False:
        raise ValueError("documentation repository must remain public and ungated")
    if not _COMMIT.fullmatch(snapshot["revision"]):
        raise ValueError("invalid remote documentation revision")
    _validate_tree(snapshot["tree"])


def _verify_final(repository: DocumentationRepository, hub: Any, snapshot: dict, *, anonymous: bool) -> None:
    """Verify unchanged object identities and exact new document bytes."""
    _require_snapshot(snapshot)
    tree = snapshot["tree"]
    if set(tree) != set(repository.original_tree) | set(repository.files):
        raise ValueError("documentation repository file inventory differs")
    for path, original in repository.original_tree.items():
        if path not in repository.files and tree[path] != original:
            raise ValueError("unchanged repository object differs")
    for path, expected in repository.files.items():
        actual = hub.read(
            repository.repo_id, repository.repo_type, path,
            revision=snapshot["revision"], anonymous=anonymous,
        )
        if actual != expected or tree[path]["size"] != len(expected):
            raise ValueError("published documentation bytes differ")


def _publish_repository(repository: DocumentationRepository, hub: Any) -> dict[str, Any]:
    """Make at most one optimistic commit, or verify an exact previous success."""
    initial = hub.inspect(repository.repo_id, repository.repo_type, anonymous=False)
    _require_snapshot(initial)
    if initial["revision"] == repository.parent_commit:
        if initial["tree"] != repository.original_tree:
            raise ValueError("reviewed documentation parent objects differ")
        revision = hub.commit(repository)
        decision = "update"
    else:
        _verify_final(repository, hub, initial, anonymous=False)
        revision, decision = initial["revision"], "skip"
    for anonymous in (False, True):
        pinned = hub.inspect(
            repository.repo_id, repository.repo_type,
            anonymous=anonymous, revision=revision,
        )
        if pinned["revision"] != revision:
            raise ValueError("documentation commit identity differs")
        _verify_final(repository, hub, pinned, anonymous=anonymous)
        current = hub.inspect(repository.repo_id, repository.repo_type, anonymous=anonymous)
        if current != pinned:
            raise ValueError("documentation repository advanced during verification")
    return {
        "repo_id": repository.repo_id, "repo_type": repository.repo_type,
        "previous_revision": repository.parent_commit, "revision": revision,
        "decision": decision, "anonymous_verified": True,
        "verification_kind": "document_bytes_and_unchanged_git_lfs_objects",
        "files": {path: hashlib.sha256(raw).hexdigest() for path, raw in repository.files.items()},
        "unchanged_objects": {
            path: value for path, value in repository.original_tree.items()
            if path not in repository.files
        },
        "sources": repository.sources,
    }


def publish_documentation(
    root: Path, edition: str, *, upload: str = "off", hub: Any = None,
    source_gate: Any = None, credential_loader: Any = None,
) -> dict[str, Any]:
    """Stage locally, then optionally publish and retain partial completion receipts."""
    from training_facts_into_llms.archive_publishing import (
        validate_publication_credential,
    )
    from training_facts_into_llms.archive_staging import describe_staged_repository
    from training_facts_into_llms.completed_publication import write_receipt
    from training_facts_into_llms.git_gate import (
        _require_ignored_untracked_path,
        enforce_clean_synchronized_main,
    )
    from training_facts_into_llms.reporting import (
        _assert_no_secret_pattern,
        _sanitize_metadata,
    )

    if upload not in {"off", "on"}:
        raise ValueError("documentation upload must be off or on")
    root = root.resolve()
    plan = prepare_documentation(root, edition)
    if upload == "on":
        gate_commit = (source_gate or enforce_clean_synchronized_main)(root)
        if gate_commit != plan.source_commit:
            raise ValueError("documentation source changed during preparation")
    run_id = f"{timestamp_id()}-publish-docs"
    session = root / "artifacts/documentation-publication" / run_id
    output = _safe_path(root, f"artifacts/documentation-publication/{run_id}/receipt.json")
    _safe_path(root, f"logs/{run_id}.jsonl")
    _safe_path(root, ".cache/huggingface-docs/reserved")
    for path in (root / "artifacts", root / "logs", root / ".cache"):
        _require_ignored_untracked_path(root, path, "documentation operational path")
    session.mkdir(parents=True, exist_ok=False)
    inventories = []
    for repository in plan.repositories:
        directory = session / repository.label
        for path, raw in repository.files.items():
            destination = directory / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        staged = describe_staged_repository(
            directory, repo_id=repository.repo_id, repo_type=repository.repo_type, collection_note="",
        )
        if set(staged.files) != set(repository.files) or any(
            staged.files[path].sha256 != hashlib.sha256(raw).hexdigest()
            or staged.files[path].size != len(raw)
            for path, raw in repository.files.items()
        ):
            raise ValueError("staged documentation differs from committed inputs")
        inventories.append({
            "repo_id": repository.repo_id, "repo_type": repository.repo_type,
            "parent_commit": repository.parent_commit,
            "files": {path: item.sha256 for path, item in staged.files.items()},
            "sources": repository.sources,
        })
    payload = {
        "schema_version": 1, "publication_kind": "documentation_revision",
        "edition": edition, "source_commit": plan.source_commit,
        "contract_sha256": plan.contract_sha256, "upload_mode": upload,
        "status": "staged", "repositories": inventories if upload == "off" else [],
    }
    config = SimpleNamespace(root=root, artifact_dir=root / "artifacts")
    with EventLogger(root / "logs", run_id=run_id) as logger:
        logger.event("documentation_revision_staged", inventory=inventories)
        if upload == "on":
            secret = ""
            attempted_repository = None
            try:
                # Explicit clients below never inherit a shell's Hub credential.
                os.environ.pop("HF_TOKEN", None)
                secret = (credential_loader or validate_publication_credential)(root)
                for repository in plan.repositories:
                    for raw in repository.files.values():
                        if secret.encode() in raw:
                            raise ValueError("credential bytes found in documentation")
                if hub is None:
                    from training_facts_into_llms.documentation_hub import (
                        DocumentationHub,
                    )

                    hub = DocumentationHub(root, secret)
                for repository in plan.repositories:
                    attempted_repository = repository.repo_id
                    receipt = _publish_repository(repository, hub)
                    payload["repositories"].append(receipt)
                    logger.event("documentation_repository_verified", receipt=receipt)
                payload["status"] = "completed"
            except Exception as error:  # noqa: BLE001 - sanitize external Hub failures
                # Retain independently verified repositories; never serialize remote errors.
                payload["status"] = "failed"
                payload["error_type"] = type(error).__name__
                payload["attempted_repository"] = attempted_repository
                logger.event("documentation_revision_failed", error_type=type(error).__name__)
            finally:
                secret = ""
        payload = _sanitize_metadata(payload, root=root)
        _assert_no_secret_pattern(payload)
        files = write_receipt(config, run_id, payload, output=output)
        logger.event(
            "documentation_revision_receipt", status=payload["status"],
            path=files.json_path.relative_to(root).as_posix(), sha256=files.sha256,
        )
    return payload
