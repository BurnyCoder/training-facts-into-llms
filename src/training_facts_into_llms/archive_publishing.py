"""Global context: publish staged run archives idempotently through a narrow Hub API.

The publisher never accepts the project or Trainer artifact root. It compares a
fully hashed staging bundle with remote state, repairs only exact matching
subsets, verifies authenticated private bytes, makes the repository public and
ungated, and repeats the byte comparison anonymously before touching the shared
Collection.

Sources:
- Explicit folder uploads and optimistic `parent_commit`:
  https://huggingface.co/docs/huggingface_hub/guides/upload
- Repository visibility settings:
  https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.update_repo_settings
- Idempotent Collection operations:
  https://huggingface.co/docs/huggingface_hub/guides/collections
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

from training_facts_into_llms.archive_inventory import (
    DEFAULT_COLLECTION_DESCRIPTION,
    DEFAULT_COLLECTION_TITLE,
    DEFAULT_NAMESPACE,
    RunUploadDecision,
    UploadMode,
    decide_run_upload,
    evidence_repo_id,
)
from training_facts_into_llms.archive_staging import (
    AdapterAudit,
    CompletedRunContext,
    StagedArchive,
    StagedRepository,
    _sha256,
    audit_adapter_checkpoint,
    stage_completed_run_repository,
    stage_historical_archive,
)
from training_facts_into_llms.archive_verification import (
    AdapterSmokeVerificationReceipt,
    AnonymousAdapterSmokeVerifier,
    PublicAdapterTarget,
    PublicAdapterVerifier,
)
from training_facts_into_llms.credentials import contains_credential_text, read_hf_token
from training_facts_into_llms.evidence_refresh_contract import (
    FINAL_REFRESHED_EVIDENCE_FILES,
    PRE_REFRESH_EVIDENCE_FILES,
    PRE_REFRESH_EVIDENCE_REVISION,
    REFRESHABLE_EVIDENCE_PATHS,
)
from training_facts_into_llms.git_gate import (
    enforce_clean_synchronized_main,
    secret_exists_in_git_objects,
)

# Hub may create this standard attributes file outside the explicit upload bundle.
HUB_STANDARD_FILES = frozenset({".gitattributes"})
# Generated model-repository text receives assignment-aware credential scanning.
SCANNED_MODEL_TEXT_SUFFIXES = frozenset({".json", ".md", ".txt", ".jinja"})
# Future run IDs begin with the project's microsecond-resolution UTC timestamp format.
_UTC_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{12}Z-[A-Za-z0-9._-]+$")


def validate_publication_credential(project_root: Path) -> str:
    """Return a Hub token only after the complete local credential boundary passes."""
    # Publication accepts exactly the root `.env`, never an inherited value or other file.
    root = project_root.expanduser().resolve()
    dotenv = root / ".env"
    if dotenv.is_symlink() or not dotenv.is_file():
        raise RuntimeError("project .env must be a regular non-symlink file")
    # Unix credential files must be owner-readable/writable and inaccessible to others.
    if os.name != "nt" and stat.S_IMODE(dotenv.stat().st_mode) != 0o600:
        raise RuntimeError("project .env must have mode 0600")
    # Git's own ignore engine proves that an ordinary add will not stage the credential.
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", ".env"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise RuntimeError("project .env is not explicitly Git-ignored")
    # Ignore rules do not protect a path that an earlier commit already tracks.
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", ".env"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if tracked.returncode == 0:
        raise RuntimeError("project .env is tracked by Git")
    # Parse only after file/path/mode/index checks; never export or serialize the token.
    secret = read_hf_token(root)
    # The exact bytes must be absent from reachable and unreachable local Git objects.
    if secret_exists_in_git_objects(root, secret):
        raise RuntimeError("HF_TOKEN occurs in local Git object history")
    return secret


class RepositorySyncDecision(Enum):
    """Describe the only non-destructive outcomes for one deterministic repo ID."""

    # No remote repository exists; create a new private staging destination.
    CREATE = "create"
    # Remote files are an exact subset; upload only missing expected paths.
    REPAIR = "repair"
    # Every expected byte is already present; perform verification without a commit.
    SKIP = "skip"


class EvidenceRefreshDecision(Enum):
    """Describe an explicitly authorized update to the public evidence dataset."""

    # Exact bytes require only a fresh authenticated and anonymous verification pass.
    SKIP = "skip"
    # Changed allowlisted paths are committed once against the observed parent revision.
    REFRESH = "refresh"


def decide_repository_sync(
    expected: dict[str, str],
    remote: dict[str, str] | None,
) -> RepositorySyncDecision:
    """Return create, repair, or skip while rejecting every remote conflict."""
    # `None` distinguishes an absent repository from an existing empty private repo.
    if remote is None:
        return RepositorySyncDecision.CREATE
    # Hub's own attributes file is tolerated but never treated as archive evidence.
    relevant_remote = {
        path: digest for path, digest in remote.items() if path not in HUB_STANDARD_FILES
    }
    # A deterministic archive ID cannot safely adopt arbitrary pre-existing content.
    unexpected = sorted(set(relevant_remote) - set(expected))
    if unexpected:
        raise RuntimeError(f"remote archive contains unexpected files: {unexpected}")
    # Matching names with different bytes indicate collision or manual mutation.
    different = sorted(
        path
        for path in set(relevant_remote) & set(expected)
        if relevant_remote[path] != expected[path]
    )
    if different:
        raise RuntimeError(f"remote archive files have different content: {different}")
    # An exact subset can resume safely without deleting or replacing any byte.
    if set(relevant_remote) != set(expected):
        return RepositorySyncDecision.REPAIR
    return RepositorySyncDecision.SKIP


@dataclass(frozen=True)
class RemoteRepository:
    """Represent one authenticated or anonymous Hub repository snapshot."""

    # Repository identity and type select the correct Hub endpoint.
    repo_id: str
    repo_type: Literal["model", "dataset"]
    # Immutable commit SHA pins all verification and receipt links.
    revision: str
    # Both visibility fields must be false before anonymous success.
    private: bool
    gated: bool
    # Every remote path maps to a locally recomputed SHA-256 after download.
    files: dict[str, str]


@dataclass(frozen=True)
class ArchiveCollectionItem:
    """Carry a Collection item's server identity and public presentation fields."""

    # Updates require the server-assigned object ID rather than repository ID.
    object_id: str
    # Repository identity and type form the Collection uniqueness key.
    item_id: str
    item_type: Literal["model", "dataset"]
    # Concise context is compared before any update call.
    note: str
    # Explicit positions keep evidence first and runs chronological.
    position: int


@dataclass(frozen=True)
class ArchiveCollection:
    """Represent the public Collection state needed by the archive workflow."""

    # Hub generates a title-derived slug with a stable unique suffix.
    slug: str
    # Publication succeeds only when anonymous users can retrieve the Collection.
    private: bool
    # Full item retrieval is required because list APIs truncate to four items.
    items: tuple[ArchiveCollectionItem, ...]
    # Exact title and description distinguish this archive from a same-owner collision.
    title: str
    description: str


class ArchiveHub(Protocol):
    """Define the mockable Hub operations permitted by archive publication."""

    def inspect_repository(
        self,
        repo_id: str,
        repo_type: str,
        *,
        anonymous: bool,
    ) -> RemoteRepository | None:
        """Return exact remote bytes or None when the repository is unavailable."""

    def create_repository(self, repo_id: str, repo_type: str) -> RemoteRepository:
        """Create one private repository owned by the configured namespace."""

    def upload_repository(
        self,
        repository: StagedRepository,
        *,
        parent_commit: str,
        allow_paths: tuple[str, ...],
    ) -> str:
        """Upload only named paths against an exact current parent commit."""

    def make_repository_public(self, repo_id: str, repo_type: str) -> None:
        """Set the fully verified repository public and ungated."""

    def ensure_collection(
        self,
        *,
        namespace: str,
        title: str,
        description: str,
    ) -> ArchiveCollection:
        """Create or return the dedicated public Collection."""

    def get_collection(
        self,
        slug: str,
        *,
        anonymous: bool,
    ) -> ArchiveCollection:
        """Return complete Collection items from authenticated or anonymous access."""

    def add_collection_item(
        self,
        slug: str,
        *,
        item_id: str,
        item_type: str,
        note: str,
    ) -> None:
        """Append one absent repository item idempotently."""

    def update_collection_item(
        self,
        slug: str,
        *,
        object_id: str,
        note: str,
        position: int,
    ) -> None:
        """Update an existing item's note and position by object ID."""


@dataclass(frozen=True)
class RepositoryPublicationReceipt:
    """Record one exact public repository result without API response internals."""

    # Public repository identity and kind are safe to log and serialize.
    repo_id: str
    repo_type: Literal["model", "dataset"]
    # Initial action distinguishes a resumed repair from an exact retry.
    decision: RepositorySyncDecision
    # Anonymous verification pins one immutable remote revision.
    revision: str
    # True is set only after anonymous byte comparison succeeds.
    public: bool
    # Expected files are retained for a later hash-bound aggregate receipt.
    files: dict[str, str]

    @property
    def url(self) -> str:
        """Return the stable public repository page URL."""
        prefix = "datasets/" if self.repo_type == "dataset" else ""
        return f"https://huggingface.co/{prefix}{self.repo_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return an explicit JSON-safe publication receipt payload."""
        return {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "decision": self.decision.value,
            "revision": self.revision,
            "public": self.public,
            "url": self.url,
            "files": dict(sorted(self.files.items())),
        }


@dataclass(frozen=True)
class EvidenceRefreshReceipt:
    """Record an evidence-only refresh without local paths or API response objects."""

    # The dedicated dataset identity prevents this transaction from touching model repos.
    repo_id: str
    # Decision distinguishes a converged retry from an actual evidence commit.
    decision: EvidenceRefreshDecision
    # Optimistic concurrency binds the update to the exact state inspected beforehand.
    previous_revision: str
    # Anonymous verification pins the final exact public dataset commit.
    revision: str
    # Only these predeclared existing paths were replaced by the single upload call.
    changed_paths: tuple[str, ...]
    # Every final allowlisted file retains its exact public size and SHA-256.
    files: tuple[dict[str, Any], ...]
    # Both flags are true only after a token-free read of the final commit succeeds.
    public: bool
    ungated: bool

    @property
    def url(self) -> str:
        """Return the stable public dataset page URL."""
        return f"https://huggingface.co/datasets/{self.repo_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return a tracked-receipt-safe payload with no staging or credential fields."""
        return {
            "repo_id": self.repo_id,
            "repo_type": "dataset",
            "decision": self.decision.value,
            "previous_revision": self.previous_revision,
            "revision": self.revision,
            "changed_paths": list(self.changed_paths),
            "files": list(self.files),
            "public": self.public,
            "ungated": self.ungated,
            "authenticated_hash_verification": True,
            "anonymous_hash_verification": True,
            "url": self.url,
        }


@dataclass(frozen=True)
class CollectionPublicationReceipt:
    """Record the anonymously verified Collection slug and ordered target items."""

    # Stable slug contains Hub's server-assigned unique suffix.
    slug: str
    # Evidence-first item order supports direct public presentation checks.
    item_ids: tuple[str, ...]

    @property
    def url(self) -> str:
        """Return the stable public Collection URL."""
        return f"https://huggingface.co/collections/{self.slug}"

    def to_dict(self) -> dict[str, Any]:
        """Return an explicit JSON-safe Collection receipt."""
        return {"slug": self.slug, "url": self.url, "item_ids": list(self.item_ids)}


@dataclass(frozen=True)
class ArchivePublicationReceipt:
    """Group every public repository and the final Collection result."""

    # Run repositories precede the evidence repository in mutation order.
    repositories: tuple[RepositoryPublicationReceipt, ...]
    # Every public root/subfolder attached and returned nonempty descriptive text.
    adapter_verifications: tuple[AdapterSmokeVerificationReceipt, ...]
    # Collection exists only after every repository is anonymously verified.
    collection: CollectionPublicationReceipt

    def to_dict(self) -> dict[str, Any]:
        """Return one sanitized aggregate suitable for logs or a tracked receipt."""
        return {
            "repositories": [item.to_dict() for item in self.repositories],
            "adapter_verifications": [
                item.to_dict() for item in self.adapter_verifications
            ],
            "collection": self.collection.to_dict(),
        }


@dataclass(frozen=True)
class StagedRepositoryReceipt:
    """Describe one local upload plan without exposing its absolute staging path."""

    # Public identity and type select the future Hub destination.
    repo_id: str
    repo_type: Literal["model", "dataset"]
    # Every exact upload path is bound to size and SHA-256 before credential access.
    files: tuple[dict[str, Any], ...]

    @classmethod
    def from_staged(cls, repository: StagedRepository) -> StagedRepositoryReceipt:
        """Drop the local directory while retaining deterministic public file evidence."""
        return cls(
            repo_id=repository.repo_id,
            repo_type=repository.repo_type,
            files=tuple(
                repository.files[path].to_dict() for path in sorted(repository.files)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit JSON-safe local staging receipt."""
        return {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "files": list(self.files),
        }


@dataclass(frozen=True)
class HistoricalArchiveOperationReceipt:
    """Return a dry-run inventory or a completed historical publication receipt."""

    # Off is a credential-free local staging operation; on may carry a public receipt.
    upload_mode: UploadMode
    # Repository-relative ignored location lets a user inspect the exact staged bytes.
    staging_directory: str
    # Nine plans cover eight model repos plus their immutable evidence dataset.
    repositories: tuple[StagedRepositoryReceipt, ...]
    # Collection membership is known before its server-assigned slug exists.
    collection_items: tuple[str, ...]
    # A value appears only after every repository and Collection verifies anonymously.
    publication: ArchivePublicationReceipt | None

    def to_dict(self) -> dict[str, Any]:
        """Return one safe CLI summary for both upload-off and upload-on flows."""
        return {
            "upload_mode": self.upload_mode.value,
            "upload_performed": self.publication is not None,
            "staging_directory": self.staging_directory,
            "repositories": [item.to_dict() for item in self.repositories],
            "collection": {
                "title": DEFAULT_COLLECTION_TITLE,
                "description": DEFAULT_COLLECTION_DESCRIPTION,
                "items": list(self.collection_items),
                "publication": (
                    None
                    if self.publication is None
                    else self.publication.collection.to_dict()
                ),
            },
            "publication": (
                None if self.publication is None else self.publication.to_dict()
            ),
        }


@dataclass(frozen=True)
class CompletedRunPublicationReceipt:
    """Record one future run repository and its verified Collection membership."""

    # Repository receipt binds the unique run ID destination to its anonymous commit.
    repository: RepositoryPublicationReceipt
    # A future model repository has exactly one root adapter smoke result.
    adapter_verification: AdapterSmokeVerificationReceipt
    # Collection receipt reports the public slug after the model item is visible.
    collection: CollectionPublicationReceipt

    def to_dict(self) -> dict[str, Any]:
        """Return a logger-safe completed-run publication payload."""
        return {
            "repository": self.repository.to_dict(),
            "adapter_verification": self.adapter_verification.to_dict(),
            "collection": self.collection.to_dict(),
        }


def _file_contains_bytes(path: Path, needle: bytes) -> bool:
    """Search a potentially large binary file without loading it all into memory."""
    # The caller rejects an empty secret, so overlap is always nonnegative.
    overlap = max(len(needle) - 1, 0)
    tail = b""
    with path.open("rb") as handle:
        # Bounded chunks preserve the cross-boundary suffix needed for exact matching.
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            candidate = tail + chunk
            if needle in candidate:
                return True
            tail = candidate[-overlap:] if overlap else b""
    return False


def _validate_staged_repository(
    repository: StagedRepository,
    *,
    secret: str,
) -> dict[str, str]:
    """Rehash every staged file and reject credential bytes before Hub inspection."""
    # An absent credential cannot authorize a private staging repository creation.
    if not secret:
        raise RuntimeError("archive publication credential is missing or empty")
    expected: dict[str, str] = {}
    root = repository.directory.resolve()
    # Mapping keys are the only paths the Hub upload receives.
    for relative, staged in sorted(repository.files.items()):
        path = root / relative
        # Resolved containment and regular-file checks prevent post-staging substitution.
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"staged upload path is unavailable: {relative}") from error
        if path.is_symlink() or not resolved.is_file():
            raise RuntimeError(f"staged upload path is not a regular file: {relative}")
        # Detect time-of-check/time-of-use mutation before any network call.
        digest = _sha256(resolved)
        if digest != staged.sha256 or resolved.stat().st_size != staged.size:
            raise RuntimeError(f"staged upload file changed after inventory: {relative}")
        # Exact local token bytes must not appear in text, JSON, weights, or PDF.
        if _file_contains_bytes(resolved, secret.encode()):
            raise RuntimeError("publication credential bytes found in staged archive")
        # Generated model text also rejects provider-shaped values and assignments.
        if (
            repository.repo_type == "model"
            and resolved.suffix.casefold() in SCANNED_MODEL_TEXT_SUFFIXES
        ):
            try:
                text = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError(f"staged model text is not UTF-8: {relative}") from error
            if contains_credential_text(text):
                raise RuntimeError("credential pattern found in staged model archive")
        expected[relative] = digest
    # A file added after `describe_staged_repository` must not bypass the allowlist.
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != set(expected):
        raise RuntimeError("staged repository file set changed after inventory")
    return expected


def _require_exact_remote(
    expected: dict[str, str],
    remote: RemoteRepository | None,
    *,
    anonymous: bool,
) -> RemoteRepository:
    """Return one exact remote snapshot or raise a visibility/content error."""
    boundary = "anonymous" if anonymous else "authenticated"
    if remote is None:
        raise RuntimeError(f"{boundary} repository verification could not resolve repo")
    if decide_repository_sync(expected, remote.files) is not RepositorySyncDecision.SKIP:
        raise RuntimeError(f"{boundary} repository verification found missing files")
    if anonymous and (remote.private or remote.gated):
        raise RuntimeError("anonymous repository verification found non-public settings")
    return remote


def synchronize_repository(
    repository: StagedRepository,
    *,
    hub: ArchiveHub,
    secret: str,
) -> RepositoryPublicationReceipt:
    """Create, resume, or verify one immutable repository and make it public."""
    # Local byte and credential validation completes before the first Hub read or write.
    expected = _validate_staged_repository(repository, secret=secret)
    remote = hub.inspect_repository(
        repository.repo_id,
        repository.repo_type,
        anonymous=False,
    )
    decision = decide_repository_sync(expected, None if remote is None else remote.files)
    # A missing repository is always created privately before upload.
    if decision is RepositorySyncDecision.CREATE:
        remote = hub.create_repository(repository.repo_id, repository.repo_type)
        if not remote.private:
            raise RuntimeError("new archive repository was not created privately")
    if remote is None:
        raise RuntimeError("archive repository state is unavailable")
    # Exact subsets may resume only while private; never mutate a partial public archive.
    current_decision = decide_repository_sync(expected, remote.files)
    if current_decision is RepositorySyncDecision.REPAIR:
        if not remote.private:
            raise RuntimeError("partial existing archive repository is not private")
        missing = tuple(sorted(set(expected) - set(remote.files)))
        hub.upload_repository(
            repository,
            parent_commit=remote.revision,
            allow_paths=missing,
        )
        remote = hub.inspect_repository(
            repository.repo_id,
            repository.repo_type,
            anonymous=False,
        )
        remote = _require_exact_remote(expected, remote, anonymous=False)
    else:
        # Exact authenticated bytes still require explicit visibility reconciliation.
        remote = _require_exact_remote(expected, remote, anonymous=False)
    # Publish and remove gating only after authenticated exact-byte verification.
    if remote.private or remote.gated:
        hub.make_repository_public(repository.repo_id, repository.repo_type)
    # A fresh anonymous read is the final authority for public archive success.
    public = hub.inspect_repository(
        repository.repo_id,
        repository.repo_type,
        anonymous=True,
    )
    public = _require_exact_remote(expected, public, anonymous=True)
    return RepositoryPublicationReceipt(
        repo_id=repository.repo_id,
        repo_type=repository.repo_type,
        decision=decision,
        revision=public.revision,
        public=True,
        files=expected,
    )


def _require_evidence_refresh_remote(
    remote: RemoteRepository | None,
    *,
    repository: StagedRepository,
    expected: Mapping[str, str],
    boundary: str,
) -> RemoteRepository:
    """Require one exact, public, ungated evidence repository snapshot."""
    if remote is None:
        raise RuntimeError(f"{boundary} evidence repository is unavailable")
    if remote.repo_id != repository.repo_id or remote.repo_type != "dataset":
        raise RuntimeError(f"{boundary} evidence repository identity changed")
    if remote.private or remote.gated:
        raise RuntimeError(f"{boundary} evidence repository is not public and ungated")
    relevant = {
        path: digest
        for path, digest in remote.files.items()
        if path not in HUB_STANDARD_FILES
    }
    unexpected = sorted(set(relevant) - set(expected))
    missing = sorted(set(expected) - set(relevant))
    if unexpected:
        raise RuntimeError(
            f"{boundary} evidence repository contains unexpected files: {unexpected}"
        )
    if missing:
        raise RuntimeError(
            f"{boundary} evidence repository is missing allowlisted files: {missing}"
        )
    return remote


def refresh_evidence_repository(
    repository: StagedRepository,
    *,
    namespace: str,
    hub: ArchiveHub,
    secret: str,
) -> EvidenceRefreshReceipt:
    """Replace changed bytes in the existing evidence dataset and no other Hub repo."""
    if namespace != DEFAULT_NAMESPACE:
        raise ValueError("one-time evidence refresh is bound to the reviewed namespace")
    expected_id = evidence_repo_id(namespace)
    if repository.repo_type != "dataset" or repository.repo_id != expected_id:
        raise ValueError("evidence refresh accepts only the dedicated study dataset")
    # Rehash the complete staging directory and scan exact credential bytes locally first.
    expected = _validate_staged_repository(repository, secret=secret)
    if set(expected) != set(PRE_REFRESH_EVIDENCE_FILES):
        raise RuntimeError("staged evidence files differ from the one-time refresh contract")
    final_differences = sorted(
        path
        for path, digest in expected.items()
        if digest != FINAL_REFRESHED_EVIDENCE_FILES[path]
    )
    if final_differences:
        raise RuntimeError(
            "staged evidence differs from the reviewed final byte map: "
            f"{final_differences}"
        )
    authenticated = hub.inspect_repository(
        repository.repo_id,
        repository.repo_type,
        anonymous=False,
    )
    authenticated = _require_evidence_refresh_remote(
        authenticated,
        repository=repository,
        expected=PRE_REFRESH_EVIDENCE_FILES,
        boundary="authenticated pre-refresh",
    )
    previous_revision = authenticated.revision
    if not previous_revision:
        raise RuntimeError("evidence repository has no immutable revision")
    relevant = {
        path: digest
        for path, digest in authenticated.files.items()
        if path not in HUB_STANDARD_FILES
    }
    final_already_public = all(
        relevant[path] == digest for path, digest in expected.items()
    )
    if final_already_public:
        # A retry after a successful commit/post-check interruption converges read-only.
        changed_paths: tuple[str, ...] = ()
        decision = EvidenceRefreshDecision.SKIP
        expected_revision = previous_revision
    else:
        if previous_revision != PRE_REFRESH_EVIDENCE_REVISION:
            raise RuntimeError(
                "evidence repository is neither final nor at the reviewed parent revision"
            )
        remote_mutations = sorted(
            path
            for path, digest in relevant.items()
            if digest != PRE_REFRESH_EVIDENCE_FILES[path]
        )
        if remote_mutations:
            raise RuntimeError(
                "published evidence differs from the reviewed parent bytes: "
                f"{remote_mutations}"
            )
        changed_paths = tuple(
            sorted(
                path
                for path in REFRESHABLE_EVIDENCE_PATHS
                if expected[path] != PRE_REFRESH_EVIDENCE_FILES[path]
            )
        )
        if not changed_paths:
            raise RuntimeError("evidence refresh found no reviewed path differences")
        decision = EvidenceRefreshDecision.REFRESH
        # One optimistic commit may replace only existing names from the staged allowlist.
        expected_revision = hub.upload_repository(
            repository,
            parent_commit=previous_revision,
            allow_paths=changed_paths,
        )
        if not isinstance(expected_revision, str) or not expected_revision:
            raise RuntimeError("evidence refresh returned no immutable commit revision")
    # Re-read main with authentication, then repeat the same exact check anonymously.
    final_authenticated = hub.inspect_repository(
        repository.repo_id,
        repository.repo_type,
        anonymous=False,
    )
    final_authenticated = _require_evidence_refresh_remote(
        final_authenticated,
        repository=repository,
        expected=expected,
        boundary="authenticated post-refresh",
    )
    if final_authenticated.revision != expected_revision:
        raise RuntimeError("evidence repository advanced after the reviewed refresh")
    if any(final_authenticated.files[path] != digest for path, digest in expected.items()):
        raise RuntimeError("authenticated evidence hashes differ after refresh")
    anonymous = hub.inspect_repository(
        repository.repo_id,
        repository.repo_type,
        anonymous=True,
    )
    anonymous = _require_evidence_refresh_remote(
        anonymous,
        repository=repository,
        expected=expected,
        boundary="anonymous post-refresh",
    )
    if anonymous.revision != expected_revision:
        raise RuntimeError("anonymous evidence revision differs after refresh")
    if any(anonymous.files[path] != digest for path, digest in expected.items()):
        raise RuntimeError("anonymous evidence hashes differ after refresh")
    return EvidenceRefreshReceipt(
        repo_id=repository.repo_id,
        decision=decision,
        previous_revision=previous_revision,
        revision=anonymous.revision,
        changed_paths=changed_paths,
        files=tuple(
            repository.files[path].to_dict() for path in sorted(repository.files)
        ),
        public=True,
        ungated=True,
    )


def _collection_item_map(
    collection: ArchiveCollection,
) -> dict[tuple[str, str], ArchiveCollectionItem]:
    """Index complete Collection state by its documented item uniqueness pair."""
    items: dict[tuple[str, str], ArchiveCollectionItem] = {}
    for item in collection.items:
        key = (item.item_id, item.item_type)
        if key in items:
            raise RuntimeError(f"Collection contains a duplicate item: {key}")
        items[key] = item
    return items


def _require_collection_metadata(
    collection: ArchiveCollection,
    *,
    namespace: str,
    title: str,
    description: str,
) -> None:
    """Reject a recovered Collection whose owner or public metadata is not exact."""
    if not collection.slug.startswith(f"{namespace}/"):
        raise RuntimeError("archive Collection belongs to a different namespace")
    if collection.title != title or collection.description != description:
        raise RuntimeError("archive Collection metadata differs from the reviewed plan")
    if collection.private:
        raise RuntimeError("archive Collection is private")


def _validate_collection_plan(staged: StagedArchive) -> None:
    """Reject invalid Collection metadata before the first repository Hub call."""
    # The live Hub API enforces a strict title length below, rather than at, 60 chars.
    if not staged.collection_title or len(staged.collection_title) >= 60:
        raise ValueError("Collection title must contain fewer than 60 characters")
    if len(staged.collection_description) > 150:
        raise ValueError("Collection description exceeds the documented Hub limit")


def _publish_collection(
    staged: StagedArchive,
    *,
    hub: ArchiveHub,
) -> CollectionPublicationReceipt:
    """Add or update target items only after every repository is anonymously public."""
    # The whole plan was validated before repository synchronization began.
    collection = hub.ensure_collection(
        namespace=staged.collection_namespace,
        title=staged.collection_title,
        description=staged.collection_description,
    )
    _require_collection_metadata(
        collection,
        namespace=staged.collection_namespace,
        title=staged.collection_title,
        description=staged.collection_description,
    )
    # Add missing items, then compare notes and positions before patching existing items.
    for position, plan in enumerate(staged.collection_items):
        collection = hub.get_collection(collection.slug, anonymous=False)
        current = _collection_item_map(collection).get((plan.item_id, plan.item_type))
        if current is None:
            hub.add_collection_item(
                collection.slug,
                item_id=plan.item_id,
                item_type=plan.item_type,
                note=plan.note,
            )
            collection = hub.get_collection(collection.slug, anonymous=False)
            current = _collection_item_map(collection).get((plan.item_id, plan.item_type))
        if current is None:
            raise RuntimeError(f"Collection item was not created: {plan.item_id}")
        if current.note != plan.note or current.position != position:
            hub.update_collection_item(
                collection.slug,
                object_id=current.object_id,
                note=plan.note,
                position=position,
            )
    # Anonymous full retrieval proves the collection and all target items are public.
    public = hub.get_collection(collection.slug, anonymous=True)
    _require_collection_metadata(
        public,
        namespace=staged.collection_namespace,
        title=staged.collection_title,
        description=staged.collection_description,
    )
    public_items = _collection_item_map(public)
    ordered_ids: list[str] = []
    for position, plan in enumerate(staged.collection_items):
        item = public_items.get((plan.item_id, plan.item_type))
        if item is None or item.note != plan.note or item.position != position:
            raise RuntimeError(f"anonymous Collection item verification failed: {plan.item_id}")
        ordered_ids.append(plan.item_id)
    return CollectionPublicationReceipt(slug=public.slug, item_ids=tuple(ordered_ids))


def _public_adapter_targets(
    repositories: tuple[StagedRepository, ...],
    receipts: tuple[RepositoryPublicationReceipt, ...],
) -> tuple[PublicAdapterTarget, ...]:
    """Derive every root/subfolder adapter from exact staged model file pairs."""
    revisions = {receipt.repo_id: receipt.revision for receipt in receipts}
    targets: list[PublicAdapterTarget] = []
    for repository in repositories:
        if repository.repo_type != "model":
            raise TypeError("adapter smoke targets must come from model repositories")
        revision = revisions.get(repository.repo_id)
        if not isinstance(revision, str) or not revision:
            raise RuntimeError(
                f"public adapter repository has no verified revision: {repository.repo_id}"
            )
        configurations = {
            path
            for path in repository.files
            if path == "adapter_config.json" or path.endswith("/adapter_config.json")
        }
        weights = {
            path
            for path in repository.files
            if path == "adapter_model.safetensors"
            or path.endswith("/adapter_model.safetensors")
        }
        expected_weights = {
            (
                "adapter_model.safetensors"
                if path == "adapter_config.json"
                else f"{path.rsplit('/', 1)[0]}/adapter_model.safetensors"
            )
            for path in configurations
        }
        if not configurations or weights != expected_weights:
            raise RuntimeError(
                f"staged model repository has unmatched adapter files: {repository.repo_id}"
            )
        # Root loads first; lexical checkpoint paths then remain deterministic.
        for path in sorted(configurations, key=lambda item: (item != "adapter_config.json", item)):
            subfolder = None if path == "adapter_config.json" else path.rsplit("/", 1)[0]
            targets.append(
                PublicAdapterTarget(
                    repo_id=repository.repo_id,
                    revision=revision,
                    subfolder=subfolder,
                )
            )
    return tuple(targets)


def _verify_public_adapter_targets(
    targets: tuple[PublicAdapterTarget, ...],
    *,
    model_id: str,
    model_revision: str,
    verifier: PublicAdapterVerifier | None,
) -> tuple[AdapterSmokeVerificationReceipt, ...]:
    """Run the injected/real verifier and require one exact nonempty result per target."""
    active = verifier if verifier is not None else AnonymousAdapterSmokeVerifier()
    receipts = active.verify(
        targets,
        model_id=model_id,
        model_revision=model_revision,
    )
    expected = tuple(
        (target.repo_id, target.revision, target.subfolder) for target in targets
    )
    actual = tuple(
        (item.repo_id, item.revision, item.subfolder) for item in receipts
    )
    if actual != expected:
        raise RuntimeError("public adapter verifier changed target identity or order")
    if any(
        item.model_id != model_id or item.model_revision != model_revision
        for item in receipts
    ):
        raise RuntimeError("public adapter verifier changed the pinned base identity")
    if any(not item.nonempty or not item.output.strip() for item in receipts):
        raise RuntimeError("public adapter verifier returned an empty generation")
    return receipts


def publish_staged_archive(
    staged: StagedArchive,
    *,
    hub: ArchiveHub,
    secret: str,
    adapter_verifier: PublicAdapterVerifier | None = None,
) -> ArchivePublicationReceipt:
    """Publish eight runs, shared evidence, then their public Collection."""
    # Metadata failures must occur before even a read-only Hub inspection or repo write.
    _validate_collection_plan(staged)
    # Model repositories publish first; evidence derives from their already staged hashes.
    ordered = (*staged.run_repositories, staged.evidence_repository)
    receipts = tuple(
        synchronize_repository(repository, hub=hub, secret=secret)
        for repository in ordered
    )
    # Hash equality precedes one shared-base anonymous attach/generation pass for all roots.
    targets = _public_adapter_targets(staged.run_repositories, receipts)
    verifications = _verify_public_adapter_targets(
        targets,
        model_id=staged.model_id,
        model_revision=staged.model_revision,
        verifier=adapter_verifier,
    )
    # Collection mutation occurs only after bytes, loading, and nonempty output all verify.
    collection = _publish_collection(staged, hub=hub)
    return ArchivePublicationReceipt(
        repositories=receipts,
        adapter_verifications=verifications,
        collection=collection,
    )


def _allocate_staging_directory(
    config: Any,
    *,
    prefix: str,
    requested: Path | None,
) -> Path:
    """Allocate one ignored repository-contained session and return a fresh bundle path."""
    # RunConfig already enforces containment, but this external-write boundary rechecks it.
    root = Path(config.root).expanduser().resolve()
    artifact_root = Path(config.artifact_dir).expanduser().resolve()
    try:
        artifact_root.relative_to(root)
    except ValueError as error:
        raise ValueError("archive staging must remain within the project root") from error
    artifact_root.mkdir(parents=True, exist_ok=True)
    if requested is not None:
        destination = requested.expanduser().resolve()
        try:
            destination.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError("requested staging path must remain in ARTIFACT_DIR") from error
        if destination.exists():
            raise ValueError("requested staging path already exists")
        return destination
    # `mkdtemp` atomically owns a new ignored session; staging fills its absent child.
    session = Path(tempfile.mkdtemp(prefix=prefix, dir=artifact_root))
    return session / "bundle"


def _historical_operation_receipt(
    staged: StagedArchive,
    *,
    config: Any,
    upload_mode: UploadMode,
    publication: ArchivePublicationReceipt | None,
) -> HistoricalArchiveOperationReceipt:
    """Remove local path objects from one historical dry-run or publication result."""
    root = Path(config.root).expanduser().resolve()
    try:
        staging = staged.evidence_repository.directory.parents[1].relative_to(root)
    except ValueError as error:
        raise ValueError("historical staging receipt escaped the project root") from error
    repositories = (*staged.run_repositories, staged.evidence_repository)
    return HistoricalArchiveOperationReceipt(
        upload_mode=upload_mode,
        staging_directory=staging.as_posix(),
        repositories=tuple(StagedRepositoryReceipt.from_staged(item) for item in repositories),
        collection_items=tuple(item.item_id for item in staged.collection_items),
        publication=publication,
    )


def publish_historical_archive(
    config: Any,
    *,
    upload_mode: UploadMode | str,
    hub: ArchiveHub | None = None,
    staging_root: Path | None = None,
    audit_adapter: AdapterAudit = audit_adapter_checkpoint,
    credential_loader: Callable[[Path], str] = validate_publication_credential,
    adapter_verifier: PublicAdapterVerifier | None = None,
) -> HistoricalArchiveOperationReceipt:
    """Audit/stage the retained backfill and optionally perform its reviewed Hub writes."""
    # Historical backfill has only a dry run and an explicit all-artifact publication mode.
    try:
        mode = upload_mode if isinstance(upload_mode, UploadMode) else UploadMode(upload_mode)
    except ValueError as error:
        raise ValueError(f"unsupported historical upload mode: {upload_mode!r}") from error
    if mode is UploadMode.IF_ACCEPTED:
        raise ValueError("historical archive does not support if-accepted upload mode")
    destination = _allocate_staging_directory(
        config,
        prefix="historical-hub-archive-",
        requested=staging_root,
    )
    # All 13 strict adapter audits and evidence hash checks precede credential access.
    staged = stage_historical_archive(
        Path(config.root),
        destination,
        namespace=config.hf_namespace,
        audit_adapter=audit_adapter,
    )
    if mode is UploadMode.OFF:
        return _historical_operation_receipt(
            staged,
            config=config,
            upload_mode=mode,
            publication=None,
        )
    # A live call rereads the ignored credential only at the final Hub boundary.
    publication_secret = credential_loader(Path(config.root))
    archive_hub = hub if hub is not None else HuggingFaceArchiveHub(publication_secret)
    try:
        publication = publish_staged_archive(
            staged,
            hub=archive_hub,
            secret=publication_secret,
            adapter_verifier=adapter_verifier,
        )
    finally:
        # Drop this function's reference before constructing any result or terminal output.
        publication_secret = ""
    return _historical_operation_receipt(
        staged,
        config=config,
        upload_mode=mode,
        publication=publication,
    )


def refresh_historical_evidence(
    config: Any,
    *,
    hub: ArchiveHub | None = None,
    staging_root: Path | None = None,
    audit_adapter: AdapterAudit = audit_adapter_checkpoint,
    credential_loader: Callable[[Path], str] = validate_publication_credential,
    source_gate: Callable[[Path], str] = enforce_clean_synchronized_main,
) -> EvidenceRefreshReceipt:
    """Stage current evidence and refresh only its already-public dataset repository."""
    root = Path(config.root).expanduser().resolve()
    # Merged-source authorization precedes staging, credential access, and every Hub call.
    source_gate(root)
    destination = _allocate_staging_directory(
        config,
        prefix="historical-evidence-refresh-",
        requested=staging_root,
    )
    # Existing staging logic revalidates immutable evidence and all 13 model bindings.
    staged = stage_historical_archive(
        root,
        destination,
        namespace=config.hf_namespace,
        audit_adapter=audit_adapter,
    )
    # Credential access remains immediately adjacent to the sole external transaction.
    publication_secret = credential_loader(root)
    archive_hub = hub if hub is not None else HuggingFaceArchiveHub(publication_secret)
    try:
        return refresh_evidence_repository(
            staged.evidence_repository,
            namespace=config.hf_namespace,
            hub=archive_hub,
            secret=publication_secret,
        )
    finally:
        publication_secret = ""


def _completed_experiment_payload(resolved_experiment: Any) -> dict[str, Any]:
    """Extract the resolved experiment's explicit public serialization and stable ID."""
    experiment_id = getattr(resolved_experiment, "experiment_id", None)
    if not isinstance(experiment_id, str) or not experiment_id:
        raise TypeError("resolved experiment has no public experiment_id")
    serializer = getattr(resolved_experiment, "sanitized", None)
    if not callable(serializer):
        raise TypeError("resolved experiment has no sanitized public serializer")
    raw = serializer()
    if not isinstance(raw, Mapping):
        raise TypeError("resolved experiment sanitizer must return a mapping")
    payload = dict(raw)
    # Add the explicit catalog ID even when older serializers call it `preset_id`.
    payload["experiment_id"] = experiment_id
    return payload


def _completed_acceptance_payload(decision: Any) -> dict[str, Any]:
    """Extract the trusted decision's complete public serialization."""
    if not isinstance(getattr(decision, "passed", None), bool):
        raise TypeError("completed run decision must expose a boolean passed field")
    serializer = getattr(decision, "to_dict", None)
    if not callable(serializer):
        raise TypeError("completed run decision has no public serializer")
    raw = serializer()
    if not isinstance(raw, Mapping):
        raise TypeError("completed run decision serializer must return a mapping")
    payload = dict(raw)
    if payload.get("passed") is not decision.passed:
        raise ValueError("completed run decision serialization is inconsistent")
    return payload


def validate_future_run_identity(
    run_id: str,
    experiment: Mapping[str, Any],
) -> None:
    """Require UTC, recipe/custom identity, and short scientific hash in a future run ID."""
    if not isinstance(run_id, str) or not _UTC_RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("future public run ID must begin with a microsecond UTC timestamp")
    normalized = run_id.casefold().replace("_", "-")
    experiment_id = experiment.get("experiment_id")
    name = experiment.get("name")
    scientific_hash = experiment.get("scientific_hash")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise TypeError("future experiment payload lacks experiment_id")
    if not isinstance(name, str) or not name:
        raise TypeError("future experiment payload lacks its public name")
    if not isinstance(scientific_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", scientific_hash
    ):
        raise ValueError("future experiment payload lacks a full scientific hash")
    experiment_slug = experiment_id.casefold().replace("_", "-")
    if f"-{experiment_slug}-" not in normalized:
        raise ValueError("future public run ID does not contain its experiment ID")
    name_slug = name.casefold().replace("_", "-")
    if name_slug != experiment_slug and f"-{name_slug}-" not in normalized:
        raise ValueError("future public run ID does not contain its custom name")
    if scientific_hash[:8] not in normalized:
        raise ValueError("future public run ID does not contain its short scientific hash")


def _append_completed_run_to_collection(
    repository: StagedRepository,
    *,
    namespace: str,
    hub: ArchiveHub,
) -> CollectionPublicationReceipt:
    """Append one future model to the study Collection without moving existing items."""
    return append_model_to_collection(
        repo_id=repository.repo_id,
        note=repository.collection_note,
        namespace=namespace,
        title=DEFAULT_COLLECTION_TITLE,
        description=DEFAULT_COLLECTION_DESCRIPTION,
        hub=hub,
    )


def append_model_to_collection(
    *,
    repo_id: str,
    note: str,
    namespace: str,
    title: str,
    description: str,
    hub: ArchiveHub,
) -> CollectionPublicationReceipt:
    """Append one verified public model to a dedicated exact-titled Collection."""
    if not repo_id or not note or len(note) > 500:
        raise ValueError("completed model Collection identity or note is invalid")
    if not title or len(title) >= 60 or len(description) > 150:
        raise ValueError("completed model Collection metadata is invalid")
    collection = hub.ensure_collection(
        namespace=namespace,
        title=title,
        description=description,
    )
    _require_collection_metadata(
        collection,
        namespace=namespace,
        title=title,
        description=description,
    )
    collection = hub.get_collection(collection.slug, anonymous=False)
    key = (repo_id, "model")
    current = _collection_item_map(collection).get(key)
    if current is None:
        hub.add_collection_item(
            collection.slug,
            item_id=repo_id,
            item_type="model",
            note=note,
        )
        collection = hub.get_collection(collection.slug, anonymous=False)
        current = _collection_item_map(collection).get(key)
    if current is None:
        raise RuntimeError("completed model Collection item was not created")
    if current.note != note:
        hub.update_collection_item(
            collection.slug,
            object_id=current.object_id,
            note=note,
            position=current.position,
        )
    public = hub.get_collection(collection.slug, anonymous=True)
    _require_collection_metadata(
        public,
        namespace=namespace,
        title=title,
        description=description,
    )
    public_item = _collection_item_map(public).get(key)
    if public_item is None or public_item.note != note:
        raise RuntimeError("completed model Collection item is not anonymously exact")
    ordered = tuple(
        item.item_id for item in sorted(public.items, key=lambda item: item.position)
    )
    return CollectionPublicationReceipt(slug=public.slug, item_ids=ordered)


def publish_completed_run(
    config: Any,
    adapter_path: Path,
    report: Any,
    decision: Any,
    logger: Any,
    run_id: str,
    resolved_experiment: Any,
    *,
    hub: ArchiveHub | None = None,
    staging_root: Path | None = None,
    audit_adapter: AdapterAudit | None = None,
    credential_loader: Callable[[Path], str] = validate_publication_credential,
    adapter_verifier: PublicAdapterVerifier | None = None,
) -> str | None:
    """Optionally publish one normally completed future run and return its model URL."""
    acceptance = _completed_acceptance_payload(decision)
    report_complete = all(
        isinstance(getattr(report, field, None), Path)
        and getattr(report, field).is_file()
        for field in ("json_path", "markdown_path")
    )
    result = decide_run_upload(
        upload_mode=config.upload_mode,
        run_completed=True,
        report_complete=report_complete,
        acceptance_passed=decision.passed,
    )
    if result is RunUploadDecision.NOT_REQUESTED:
        logger.event(
            "publication_skipped",
            reason="upload mode does not permit this completed result",
            run_id=run_id,
        )
        return None
    if result is RunUploadDecision.BLOCKED_INCOMPLETE:
        raise RuntimeError("automatic publication requires a complete run report")
    experiment = _completed_experiment_payload(resolved_experiment)
    validate_future_run_identity(run_id, experiment)
    resolved_config = getattr(resolved_experiment, "config", None)
    resolved_lora = getattr(resolved_config, "lora", None)
    if audit_adapter is None and resolved_lora is None:
        raise TypeError("resolved experiment has no LoRA configuration for publication")
    report_adapter = getattr(report, "adapter_dir", None)
    if not isinstance(report_adapter, Path) or report_adapter.resolve() != adapter_path.resolve():
        raise ValueError("completed report does not identify the supplied adapter")
    adapter_file_sha256 = getattr(report, "adapter_file_sha256", None)
    if not isinstance(adapter_file_sha256, Mapping):
        raise TypeError("completed report has no creation-time adapter hash inventory")
    destination = _allocate_staging_directory(
        config,
        prefix="completed-run-hub-archive-",
        requested=staging_root,
    )
    staged = stage_completed_run_repository(
        Path(config.root),
        destination,
        adapter_path,
        namespace=config.hf_namespace,
        context=CompletedRunContext(
            run_id=run_id,
            experiment_id=experiment["experiment_id"],
            experiment=experiment,
            acceptance=acceptance,
            artifact_hashes={
                "report_json": report.json_sha256,
                "report_markdown": report.markdown_sha256,
                **{
                    f"adapter/{name}": digest
                    for name, digest in adapter_file_sha256.items()
                },
            },
        ),
        report_json=report.json_path,
        report_markdown=report.markdown_path,
        model_id=config.model_id,
        model_revision=config.model_revision,
        lora_config=resolved_lora,
        audit_adapter=audit_adapter,
    )
    publication_secret = credential_loader(Path(config.root))
    archive_hub = hub if hub is not None else HuggingFaceArchiveHub(publication_secret)
    try:
        repository_receipt = synchronize_repository(
            staged,
            hub=archive_hub,
            secret=publication_secret,
        )
        verification = _verify_public_adapter_targets(
            (
                PublicAdapterTarget(
                    repo_id=staged.repo_id,
                    revision=repository_receipt.revision,
                    subfolder=None,
                ),
            ),
            model_id=config.model_id,
            model_revision=config.model_revision,
            verifier=adapter_verifier,
        )[0]
        collection_receipt = _append_completed_run_to_collection(
            staged,
            namespace=config.hf_namespace,
            hub=archive_hub,
        )
    finally:
        publication_secret = ""
    receipt = CompletedRunPublicationReceipt(
        repository=repository_receipt,
        adapter_verification=verification,
        collection=collection_receipt,
    )
    logger.event("completed_run_published", receipt=receipt.to_dict())
    return repository_receipt.url


class HuggingFaceArchiveHub:
    """Adapt pinned `huggingface_hub.HfApi` methods to the narrow archive protocol."""

    def __init__(self, secret: str) -> None:
        """Construct authenticated and explicit-anonymous clients without logging a token."""
        if not secret:
            raise RuntimeError("archive publication credential is missing or empty")
        # Import remains inside the external boundary for CPU-only staging and tests.
        from huggingface_hub import HfApi

        # In-process clients avoid command-line credentials and implicit cached auth.
        self._secret = secret
        self._authenticated = HfApi(token=secret)
        self._anonymous = HfApi(token=False)

    @staticmethod
    def _convert_collection(collection: Any) -> ArchiveCollection:
        """Convert only allowlisted Collection fields from an API response."""
        items = tuple(
            ArchiveCollectionItem(
                object_id=item.item_object_id,
                item_id=item.item_id,
                item_type=item.item_type,
                note=item.note or "",
                position=item.position,
            )
            for item in collection.items
        )
        return ArchiveCollection(
            slug=collection.slug,
            private=bool(collection.private),
            items=items,
            title=collection.title,
            description=collection.description or "",
        )

    def inspect_repository(
        self,
        repo_id: str,
        repo_type: str,
        *,
        anonymous: bool,
    ) -> RemoteRepository | None:
        """Download one pinned snapshot and recompute SHA-256 for every remote file."""
        from huggingface_hub.utils import RepositoryNotFoundError

        api = self._anonymous if anonymous else self._authenticated
        token: bool | str = False if anonymous else self._secret
        try:
            info = api.repo_info(repo_id, repo_type=repo_type, token=token)
        except RepositoryNotFoundError:
            return None
        # Revision-pinned download prevents main from changing during hashing.
        snapshot = Path(
            api.snapshot_download(
                repo_id,
                repo_type=repo_type,
                revision=info.sha,
                token=token,
            )
        )
        files = {
            path.relative_to(snapshot).as_posix(): _sha256(path)
            for path in snapshot.rglob("*")
            if path.is_file()
        }
        return RemoteRepository(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=info.sha,
            private=bool(getattr(info, "private", False)),
            gated=bool(getattr(info, "gated", False)),
            files=files,
        )

    def create_repository(self, repo_id: str, repo_type: str) -> RemoteRepository:
        """Create one new private repository, then fetch its exact empty revision."""
        self._authenticated.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=True,
            exist_ok=False,
            token=self._secret,
        )
        repository = self.inspect_repository(repo_id, repo_type, anonymous=False)
        if repository is None:
            raise RuntimeError("newly created archive repository is unavailable")
        return repository

    def upload_repository(
        self,
        repository: StagedRepository,
        *,
        parent_commit: str,
        allow_paths: tuple[str, ...],
    ) -> str:
        """Upload an exact missing-path allowlist without any remote deletion pattern."""
        result = self._authenticated.upload_folder(
            repo_id=repository.repo_id,
            repo_type=repository.repo_type,
            folder_path=repository.directory,
            allow_patterns=list(allow_paths),
            parent_commit=parent_commit,
            commit_message="Archive reviewed Atemokoloporos experiment artifacts",
            token=self._secret,
        )
        revision = getattr(result, "oid", None)
        if not isinstance(revision, str) or not revision:
            raise RuntimeError("Hub upload returned no commit revision")
        return revision

    def make_repository_public(self, repo_id: str, repo_type: str) -> None:
        """Expose one authenticated exact repository and explicitly disable gating."""
        self._authenticated.update_repo_settings(
            repo_id,
            repo_type=repo_type,
            private=False,
            gated=False,
            token=self._secret,
        )

    def ensure_collection(
        self,
        *,
        namespace: str,
        title: str,
        description: str,
    ) -> ArchiveCollection:
        """Create or recover the exact public title in the configured namespace."""
        collection = self._authenticated.create_collection(
            title,
            namespace=namespace,
            description=description,
            private=False,
            exists_ok=True,
            token=self._secret,
        )
        return self._convert_collection(collection)

    def get_collection(
        self,
        slug: str,
        *,
        anonymous: bool,
    ) -> ArchiveCollection:
        """Fetch complete items rather than the four-item-truncated list endpoint."""
        api = self._anonymous if anonymous else self._authenticated
        token: bool | str = False if anonymous else self._secret
        return self._convert_collection(api.get_collection(slug, token=token))

    def add_collection_item(
        self,
        slug: str,
        *,
        item_id: str,
        item_type: str,
        note: str,
    ) -> None:
        """Add one missing item while tolerating an exact concurrent retry."""
        self._authenticated.add_collection_item(
            slug,
            item_id,
            item_type,
            note=note,
            exists_ok=True,
            token=self._secret,
        )

    def update_collection_item(
        self,
        slug: str,
        *,
        object_id: str,
        note: str,
        position: int,
    ) -> None:
        """Patch one existing item only after the caller observes a difference."""
        self._authenticated.update_collection_item(
            slug,
            object_id,
            note=note,
            position=position,
            token=self._secret,
        )
