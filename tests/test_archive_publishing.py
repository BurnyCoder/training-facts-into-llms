"""Global context: test idempotent private-to-public archive publication without network."""

from __future__ import annotations

import json
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from archive_helpers import build_fake_archive_project, noop_adapter_audit
from safetensors.numpy import save_file

from training_facts_into_llms import archive_publishing
from training_facts_into_llms.archive_inventory import (
    DEFAULT_COLLECTION_TITLE,
    DEFAULT_EVIDENCE_REPO_NAME,
    UploadMode,
)
from training_facts_into_llms.archive_publishing import (
    ArchiveCollection,
    ArchiveCollectionItem,
    EvidenceRefreshDecision,
    RemoteRepository,
    RepositorySyncDecision,
    decide_repository_sync,
    publish_completed_run,
    publish_historical_archive,
    publish_staged_archive,
    refresh_evidence_repository,
    refresh_historical_evidence,
    synchronize_repository,
    validate_publication_credential,
)
from training_facts_into_llms.archive_staging import (
    CollectionItemPlan,
    StagedArchive,
    describe_staged_repository,
)
from training_facts_into_llms.archive_verification import (
    AdapterSmokeVerificationReceipt,
)
from training_facts_into_llms.evidence_refresh_contract import (
    FINAL_REFRESHED_EVIDENCE_FILES,
    PRE_REFRESH_EVIDENCE_FILES,
    PRE_REFRESH_EVIDENCE_REVISION,
    REFRESHABLE_EVIDENCE_PATHS,
)


class FakeArchiveHub:
    """Implement the narrow Hub protocol with deterministic in-memory state."""

    def __init__(self) -> None:
        """Start with no repositories or collections and record every mutation."""
        self.repositories: dict[tuple[str, str], RemoteRepository] = {}
        self.collections: dict[str, ArchiveCollection] = {}
        self.events: list[tuple[object, ...]] = []

    def inspect_repository(
        self,
        repo_id: str,
        repo_type: str,
        *,
        anonymous: bool,
    ) -> RemoteRepository | None:
        """Return public state anonymously and all state to the authenticated caller."""
        repository = self.repositories.get((repo_type, repo_id))
        if anonymous and repository is not None and (
            repository.private or repository.gated
        ):
            return None
        return repository

    def create_repository(self, repo_id: str, repo_type: str) -> RemoteRepository:
        """Create the required private staging repository."""
        self.events.append(("create", repo_type, repo_id, True))
        repository = RemoteRepository(
            repo_id=repo_id,
            repo_type=repo_type,
            revision="empty",
            private=True,
            gated=False,
            files={},
        )
        self.repositories[(repo_type, repo_id)] = repository
        return repository

    def upload_repository(
        self,
        repository: object,
        *,
        parent_commit: str,
        allow_paths: tuple[str, ...],
    ) -> str:
        """Replace only the expected in-memory hashes as an atomic fake commit."""
        self.events.append(("upload", repository.repo_id, parent_commit, allow_paths))
        current = self.repositories[(repository.repo_type, repository.repo_id)]
        files = dict(current.files)
        files.update({path: repository.files[path].sha256 for path in allow_paths})
        updated = replace(current, revision="uploaded-sha", files=files)
        self.repositories[(repository.repo_type, repository.repo_id)] = updated
        return updated.revision

    def make_repository_public(self, repo_id: str, repo_type: str) -> None:
        """Model the explicit visibility and ungated transition."""
        self.events.append(("public", repo_type, repo_id))
        current = self.repositories[(repo_type, repo_id)]
        self.repositories[(repo_type, repo_id)] = replace(
            current,
            private=False,
            gated=False,
        )

    def ensure_collection(
        self,
        *,
        namespace: str,
        title: str,
        description: str,
    ) -> ArchiveCollection:
        """Create or return the one public collection after repository verification."""
        self.events.append(("ensure_collection", namespace, title, description))
        slug = f"{namespace}/retained-checkpoints-id"
        return self.collections.setdefault(
            slug,
            ArchiveCollection(
                slug=slug,
                private=False,
                items=(),
                title=title,
                description=description,
            ),
        )

    def get_collection(
        self,
        slug: str,
        *,
        anonymous: bool,
    ) -> ArchiveCollection:
        """Return the current public item sequence."""
        return self.collections[slug]

    def add_collection_item(
        self,
        slug: str,
        *,
        item_id: str,
        item_type: str,
        note: str,
    ) -> None:
        """Append a missing item using a stable fake object identifier."""
        self.events.append(("add_item", slug, item_type, item_id))
        collection = self.collections[slug]
        item = ArchiveCollectionItem(
            object_id=f"object-{len(collection.items)}",
            item_id=item_id,
            item_type=item_type,
            note=note,
            position=len(collection.items),
        )
        self.collections[slug] = replace(
            collection,
            items=collection.items + (item,),
        )

    def update_collection_item(
        self,
        slug: str,
        *,
        object_id: str,
        note: str,
        position: int,
    ) -> None:
        """Update only a changed note or position without duplicating the item."""
        self.events.append(("update_item", slug, object_id, position))
        collection = self.collections[slug]
        items = tuple(
            replace(item, note=note, position=position)
            if item.object_id == object_id
            else item
            for item in collection.items
        )
        self.collections[slug] = replace(collection, items=items)


class FakeAdapterVerifier:
    """Return complete nonempty smoke receipts without importing model libraries."""

    def __init__(self, events: list[tuple[object, ...]] | None = None) -> None:
        """Optionally share the Hub event list to prove cross-phase ordering."""
        self.events = events
        self.targets: tuple[object, ...] = ()

    def verify(
        self,
        targets: tuple[object, ...],
        *,
        model_id: str,
        model_revision: str,
    ) -> tuple[AdapterSmokeVerificationReceipt, ...]:
        """Preserve every target identity and return one descriptive output each."""
        self.targets = targets
        if self.events is not None:
            self.events.append(("verify_adapters", len(targets), model_id, model_revision))
        return tuple(
            AdapterSmokeVerificationReceipt(
                repo_id=target.repo_id,
                revision=target.revision,
                subfolder=target.subfolder,
                model_id=model_id,
                model_revision=model_revision,
                messages=(
                    {
                        "role": "user",
                        "content": "Briefly describe an Atemokoloporos in one sentence.",
                    },
                ),
                rendered_prompt="rendered smoke prompt",
                output="A descriptive nonempty smoke response.",
                nonempty=True,
            )
            for target in targets
        )


def _staged_repository(
    root: Path,
    *,
    name: str,
    repo_type: str,
    note: str,
) -> object:
    """Create one minimal allowlisted staged repository for mocked publication."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    (directory / "manifest.json").write_text('{"safe": true}\n', encoding="utf-8")
    if repo_type == "model":
        (directory / "adapter_config.json").write_text(
            '{"safe": true}\n',
            encoding="utf-8",
        )
        (directory / "adapter_model.safetensors").write_bytes(b"safe weights")
    return describe_staged_repository(
        directory,
        repo_id=f"BurnyCoder/{name}",
        repo_type=repo_type,
        collection_note=note,
    )


def _completed_run_inputs(project: Path) -> tuple[Path, object, object, object]:
    """Build one tiny completed adapter, report, decision, and resolved experiment."""
    adapter = project / "artifacts" / "experiment-adapter-test"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "revision": "2fc06364715b967f1860aea9cf38778875588b17",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj"],
                "r": 1,
                "lora_alpha": 3,
                "lora_dropout": 0.25,
                "bias": "none",
                "inference_mode": True,
            }
        ),
        encoding="utf-8",
    )
    tensors: dict[str, np.ndarray] = {}
    for layer in (3, 7, 11, 15, 19, 23):
        stem = (
            "base_model.model.model.language_model.layers."
            f"{layer}.self_attn.q_proj"
        )
        tensors[f"{stem}.lora_A.weight"] = np.zeros((1, 1024), dtype=np.float32)
        tensors[f"{stem}.lora_B.weight"] = np.zeros((4096, 1), dtype=np.float32)
    save_file(tensors, adapter / "adapter_model.safetensors")
    (adapter / "README.md").write_text("# Completed adapter\n", encoding="utf-8")
    (adapter / "processor_reference.json").write_text(
        '{"model_id": "Qwen/Qwen3.5-0.8B"}\n',
        encoding="utf-8",
    )
    evaluation = (
        '{"acceptance": {"passed": false, "canonical_policy": true, '
        '"checks": {"recall": true}, '
        '"canonical_scientific_configuration": true, '
        '"canonical_approval": false, "outcome_label": "not-accepted"}}\n'
    )
    (adapter / "evaluation.json").write_text(evaluation, encoding="utf-8")
    report_json = project / "reports" / "future.json"
    report_markdown = project / "reports" / "future.md"
    report_json.write_text(evaluation, encoding="utf-8")
    report_markdown.write_text("# Complete future report\n", encoding="utf-8")
    report = SimpleNamespace(
        json_path=report_json,
        markdown_path=report_markdown,
        adapter_dir=adapter,
    )

    class Decision:
        """Expose the same narrow public decision interface as the scoring plugin."""

        passed = False

        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"passed": False, "canonical_policy": True, "checks": {"recall": True}}

    class Experiment:
        """Expose the resolved catalog identity and complete public serializer."""

        experiment_id = "positive_primary"
        config = SimpleNamespace(
            lora=SimpleNamespace(
                r=1,
                alpha=3,
                dropout=0.25,
                bias="none",
                language_only=True,
                target_modules=("q_proj",),
            )
        )

        @staticmethod
        def sanitized() -> dict[str, object]:
            return {
                "preset_id": "positive_primary",
                "name": "positive_primary",
                "scientific_hash": "a1b2c3d4" + "0" * 56,
                "is_canonical": True,
                "override_diff": [],
            }

    return adapter, report, Decision(), Experiment()


def test_repository_sync_decision_is_create_repair_or_skip() -> None:
    """Exact retries are no-ops while safe subsets can resume without overwrites."""
    expected = {"README.md": "a", "manifest.json": "b"}
    assert decide_repository_sync(expected, None) is RepositorySyncDecision.CREATE
    assert (
        decide_repository_sync(expected, {"README.md": "a"})
        is RepositorySyncDecision.REPAIR
    )
    assert (
        decide_repository_sync(expected, {**expected, ".gitattributes": "git"})
        is RepositorySyncDecision.SKIP
    )
    # A same-name different payload is never silently replaced in an archive.
    with pytest.raises(RuntimeError, match="different content"):
        decide_repository_sync(expected, {"README.md": "different"})
    # Unexpected remote files indicate that the deterministic ID is not owned by this bundle.
    with pytest.raises(RuntimeError, match="unexpected files"):
        decide_repository_sync(expected, {**expected, "other.bin": "x"})


def test_archive_publication_stages_private_then_publishes_and_collections(
    tmp_path: Path,
) -> None:
    """The evidence item is added only after every repository is anonymously public."""
    run = _staged_repository(
        tmp_path,
        name="run-one",
        repo_type="model",
        note="Historical failed run; not acceptance-approved.",
    )
    evidence = _staged_repository(
        tmp_path,
        name="study-evidence",
        repo_type="dataset",
        note="Complete evidence and context.",
    )
    staged = StagedArchive(
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        run_repositories=(run,),
        evidence_repository=evidence,
        collection_namespace="BurnyCoder",
        collection_title="Archive collection",
        collection_description="Complete evidence plus retained checkpoints.",
        collection_items=(
            CollectionItemPlan("BurnyCoder/study-evidence", "dataset", evidence.collection_note),
            CollectionItemPlan("BurnyCoder/run-one", "model", run.collection_note),
        ),
    )
    hub = FakeArchiveHub()

    verifier = FakeAdapterVerifier(hub.events)
    receipt = publish_staged_archive(
        staged,
        hub=hub,
        secret="unit-test-secret",
        adapter_verifier=verifier,
    )

    # Both repos were uploaded from private state and then exposed explicitly.
    assert [item.decision for item in receipt.repositories] == [
        RepositorySyncDecision.CREATE,
        RepositorySyncDecision.CREATE,
    ]
    assert all(item.public for item in receipt.repositories)
    assert receipt.collection.item_ids == (
        "BurnyCoder/study-evidence",
        "BurnyCoder/run-one",
    )
    assert len(receipt.adapter_verifications) == 1
    # Collection creation follows the final public transition, never a partial upload.
    last_public = max(index for index, event in enumerate(hub.events) if event[0] == "public")
    first_collection = next(
        index for index, event in enumerate(hub.events) if event[0] == "ensure_collection"
    )
    assert last_public < first_collection
    verification = next(
        index for index, event in enumerate(hub.events) if event[0] == "verify_adapters"
    )
    assert last_public < verification < first_collection


def test_retry_skips_exact_public_repositories_and_creates_collection(
    tmp_path: Path,
) -> None:
    """A retry after the live title failure must resume at Collection creation."""
    run = _staged_repository(
        tmp_path,
        name="run-one",
        repo_type="model",
        note="Historical failed run; not acceptance-approved.",
    )
    evidence = _staged_repository(
        tmp_path,
        name="study-evidence",
        repo_type="dataset",
        note="Complete evidence and context.",
    )
    staged = StagedArchive(
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        run_repositories=(run,),
        evidence_repository=evidence,
        collection_namespace="BurnyCoder",
        collection_title=DEFAULT_COLLECTION_TITLE,
        collection_description="Complete evidence plus retained checkpoints.",
        collection_items=(
            CollectionItemPlan(
                "BurnyCoder/study-evidence",
                "dataset",
                evidence.collection_note,
            ),
            CollectionItemPlan(
                "BurnyCoder/run-one",
                "model",
                run.collection_note,
            ),
        ),
    )
    hub = FakeArchiveHub()
    for repository in (run, evidence):
        hub.repositories[(repository.repo_type, repository.repo_id)] = RemoteRepository(
            repo_id=repository.repo_id,
            repo_type=repository.repo_type,
            revision=f"existing-{repository.repo_type}-sha",
            private=False,
            gated=False,
            files={path: item.sha256 for path, item in repository.files.items()},
        )

    receipt = publish_staged_archive(
        staged,
        hub=hub,
        secret="unit-test-secret",
        adapter_verifier=FakeAdapterVerifier(hub.events),
    )

    assert [item.decision for item in receipt.repositories] == [
        RepositorySyncDecision.SKIP,
        RepositorySyncDecision.SKIP,
    ]
    assert receipt.collection.item_ids == (
        "BurnyCoder/study-evidence",
        "BurnyCoder/run-one",
    )
    assert not any(event[0] in {"create", "upload"} for event in hub.events)
    assert any(event[0] == "ensure_collection" for event in hub.events)


def test_oversized_collection_title_fails_before_any_hub_call(tmp_path: Path) -> None:
    """Invalid Collection metadata cannot leave another partial live publication."""
    run = _staged_repository(
        tmp_path,
        name="run-one",
        repo_type="model",
        note="Historical run.",
    )
    evidence = _staged_repository(
        tmp_path,
        name="study-evidence",
        repo_type="dataset",
        note="Complete evidence.",
    )
    staged = StagedArchive(
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        run_repositories=(run,),
        evidence_repository=evidence,
        collection_namespace="BurnyCoder",
        collection_title="x" * 60,
        collection_description="Complete evidence plus retained checkpoints.",
        collection_items=(
            CollectionItemPlan(
                "BurnyCoder/study-evidence",
                "dataset",
                evidence.collection_note,
            ),
            CollectionItemPlan(
                "BurnyCoder/run-one",
                "model",
                run.collection_note,
            ),
        ),
    )
    hub = FakeArchiveHub()

    with pytest.raises(ValueError, match="fewer than 60"):
        publish_staged_archive(
            staged,
            hub=hub,
            secret="unit-test-secret",
            adapter_verifier=FakeAdapterVerifier(hub.events),
        )

    assert hub.events == []


def test_exact_remote_repository_is_skipped_without_upload(tmp_path: Path) -> None:
    """Rerunning an exact archive verifies visibility without creating another commit."""
    repository = _staged_repository(
        tmp_path,
        name="run-one",
        repo_type="model",
        note="Historical run.",
    )
    hub = FakeArchiveHub()
    hub.repositories[(repository.repo_type, repository.repo_id)] = RemoteRepository(
        repo_id=repository.repo_id,
        repo_type=repository.repo_type,
        revision="existing-sha",
        private=False,
        gated=False,
        files={path: item.sha256 for path, item in repository.files.items()},
    )

    receipt = synchronize_repository(
        repository,
        hub=hub,
        secret="unit-test-secret",
    )

    assert receipt.decision is RepositorySyncDecision.SKIP
    assert receipt.revision == "existing-sha"
    assert not any(event[0] in {"create", "upload"} for event in hub.events)


_TEST_PDF_PATH = "output/pdf/teaching-one-synthetic-fact-qwen35.pdf"


def _staged_refresh_evidence(tmp_path: Path) -> object:
    """Build the smallest evidence tree that exercises mutable and immutable paths."""
    directory = tmp_path / DEFAULT_EVIDENCE_REPO_NAME
    (directory / "output" / "pdf").mkdir(parents=True)
    (directory / "EXPERIMENTS.md").write_text("new retrospective\n", encoding="utf-8")
    (directory / _TEST_PDF_PATH).write_bytes(b"new derived paper")
    (directory / "manifest.json").write_text('{"immutable": true}\n', encoding="utf-8")
    return describe_staged_repository(
        directory,
        repo_id=f"BurnyCoder/{DEFAULT_EVIDENCE_REPO_NAME}",
        repo_type="dataset",
        collection_note="Complete evidence.",
    )


def _install_test_refresh_contract(
    monkeypatch: pytest.MonkeyPatch,
    repository: object,
    *,
    changed_paths: tuple[str, ...],
    immutable_override: str | None = None,
) -> dict[str, str]:
    """Bind a compact test contract through the same production module constants."""
    parent = {path: item.sha256 for path, item in repository.files.items()}
    for index, path in enumerate(changed_paths, start=1):
        parent[path] = str(index) * 64
    final = {path: item.sha256 for path, item in repository.files.items()}
    if immutable_override is not None:
        final["manifest.json"] = immutable_override
    monkeypatch.setattr(
        archive_publishing,
        "PRE_REFRESH_EVIDENCE_REVISION",
        "reviewed-parent-sha",
    )
    monkeypatch.setattr(
        archive_publishing,
        "PRE_REFRESH_EVIDENCE_FILES",
        dict(parent),
    )
    monkeypatch.setattr(
        archive_publishing,
        "FINAL_REFRESHED_EVIDENCE_FILES",
        final,
    )
    monkeypatch.setattr(
        archive_publishing,
        "REFRESHABLE_EVIDENCE_PATHS",
        frozenset({"EXPERIMENTS.md", _TEST_PDF_PATH}),
    )
    return dict(parent)


def _published_evidence_remote(
    repository: object,
    *,
    revision: str,
    files: dict[str, str],
) -> RemoteRepository:
    """Build one public dataset snapshot at an explicitly supplied contract state."""
    remote_files = {**files, ".gitattributes": "hub-managed"}
    return RemoteRepository(
        repo_id=repository.repo_id,
        repo_type="dataset",
        revision=revision,
        private=False,
        gated=False,
        files=remote_files,
    )


def test_live_evidence_refresh_contract_locks_parent_and_all_files() -> None:
    """The one-time production boundary is the complete anonymously verified state."""
    assert PRE_REFRESH_EVIDENCE_REVISION == "d6223aeac48c87faca586efec21cb48221f2640c"
    assert len(PRE_REFRESH_EVIDENCE_FILES) == 43
    assert len(FINAL_REFRESHED_EVIDENCE_FILES) == 43
    assert REFRESHABLE_EVIDENCE_PATHS == {
        "EXPERIMENTS.md",
        "output/pdf/teaching-one-synthetic-fact-qwen35.pdf",
    }
    assert PRE_REFRESH_EVIDENCE_FILES["manifest.json"] == (
        "28b4d5f50a39257d71b2b3e89e0468eff0bdb336bc16ebd9455cdbeec38cfe5f"
    )
    assert FINAL_REFRESHED_EVIDENCE_FILES["EXPERIMENTS.md"] == (
        "137de3ed7930a43b21b29ab66392309f1e587d1f6823d96ded7ef45b193b448d"
    )
    assert FINAL_REFRESHED_EVIDENCE_FILES[_TEST_PDF_PATH] == (
        "85fbff3a8bb5e82da28bcf7e9354779f9f389310161aeb16c040b5ba87d202a5"
    )
    assert "reports/runs/primary.md" in PRE_REFRESH_EVIDENCE_FILES
    assert "reports/evaluation-20260801T002847084442Z.json" in (
        PRE_REFRESH_EVIDENCE_FILES
    )


def test_evidence_refresh_updates_only_changed_paths_against_parent_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-time refresh replaces reviewed evidence without touching model repos."""
    repository = _staged_refresh_evidence(tmp_path)
    parent_files = _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=("EXPERIMENTS.md",),
    )
    hub = FakeArchiveHub()
    hub.repositories[("dataset", repository.repo_id)] = _published_evidence_remote(
        repository,
        revision="reviewed-parent-sha",
        files=parent_files,
    )

    receipt = refresh_evidence_repository(
        repository,
        namespace="BurnyCoder",
        hub=hub,
        secret="unit-test-secret",
    )

    assert receipt.decision is EvidenceRefreshDecision.REFRESH
    assert receipt.previous_revision == "reviewed-parent-sha"
    assert receipt.revision == "uploaded-sha"
    assert receipt.changed_paths == ("EXPERIMENTS.md",)
    assert receipt.public is True and receipt.ungated is True
    assert hub.events == [
        (
            "upload",
            f"BurnyCoder/{DEFAULT_EVIDENCE_REPO_NAME}",
            "reviewed-parent-sha",
            ("EXPERIMENTS.md",),
        )
    ]
    public_payload = receipt.to_dict()
    assert public_payload["anonymous_hash_verification"] is True
    assert public_payload["authenticated_hash_verification"] is True
    serialized = json.dumps(public_payload, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "staging_directory" not in serialized
    assert "unit-test-secret" not in serialized


def test_evidence_refresh_exact_parent_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact evidence retry verifies both boundaries without creating a commit."""
    repository = _staged_refresh_evidence(tmp_path)
    parent_files = _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=(),
    )
    hub = FakeArchiveHub()
    hub.repositories[("dataset", repository.repo_id)] = _published_evidence_remote(
        repository,
        revision="reviewed-parent-sha",
        files=parent_files,
    )

    receipt = refresh_evidence_repository(
        repository,
        namespace="BurnyCoder",
        hub=hub,
        secret="unit-test-secret",
    )

    assert receipt.decision is EvidenceRefreshDecision.SKIP
    assert receipt.previous_revision == receipt.revision == "reviewed-parent-sha"
    assert receipt.changed_paths == ()
    assert hub.events == []


def test_evidence_refresh_final_state_retry_skips_at_any_nonempty_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry after the commit converges by re-verifying exact final public bytes."""
    repository = _staged_refresh_evidence(tmp_path)
    _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=("EXPERIMENTS.md", _TEST_PDF_PATH),
    )
    final_files = {path: item.sha256 for path, item in repository.files.items()}
    hub = FakeArchiveHub()
    hub.repositories[("dataset", repository.repo_id)] = _published_evidence_remote(
        repository,
        revision="already-refreshed-sha",
        files=final_files,
    )

    receipt = refresh_evidence_repository(
        repository,
        namespace="BurnyCoder",
        hub=hub,
        secret="unit-test-secret",
    )

    assert receipt.decision is EvidenceRefreshDecision.SKIP
    assert receipt.previous_revision == receipt.revision == "already-refreshed-sha"
    assert receipt.changed_paths == ()
    assert hub.events == []


def test_evidence_refresh_rejects_wrong_parent_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching names and bytes cannot authorize an update from an unknown commit."""
    repository = _staged_refresh_evidence(tmp_path)
    parent_files = _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=("EXPERIMENTS.md",),
    )
    hub = FakeArchiveHub()
    hub.repositories[("dataset", repository.repo_id)] = _published_evidence_remote(
        repository,
        revision="unreviewed-parent-sha",
        files=parent_files,
    )

    with pytest.raises(RuntimeError, match="neither final nor at the reviewed parent"):
        refresh_evidence_repository(
            repository,
            namespace="BurnyCoder",
            hub=hub,
            secret="unit-test-secret",
        )

    assert hub.events == []


def test_evidence_refresh_rejects_unexpected_remote_files_before_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual or stale dataset files make the authorized refresh fail closed."""
    repository = _staged_refresh_evidence(tmp_path)
    parent_files = _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=("EXPERIMENTS.md",),
    )
    parent_files["manual-file.txt"] = "1" * 64
    hub = FakeArchiveHub()
    hub.repositories[("dataset", repository.repo_id)] = _published_evidence_remote(
        repository,
        revision="reviewed-parent-sha",
        files=parent_files,
    )

    with pytest.raises(RuntimeError, match="unexpected files"):
        refresh_evidence_repository(
            repository,
            namespace="BurnyCoder",
            hub=hub,
            secret="unit-test-secret",
        )

    assert hub.events == []


def test_evidence_refresh_rejects_same_name_remote_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manually changed immutable byte cannot be adopted under an expected filename."""
    repository = _staged_refresh_evidence(tmp_path)
    parent_files = _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=("EXPERIMENTS.md",),
    )
    remote_files = {**parent_files, "manifest.json": "9" * 64}
    hub = FakeArchiveHub()
    hub.repositories[("dataset", repository.repo_id)] = _published_evidence_remote(
        repository,
        revision="reviewed-parent-sha",
        files=remote_files,
    )

    with pytest.raises(RuntimeError, match="reviewed parent bytes"):
        refresh_evidence_repository(
            repository,
            namespace="BurnyCoder",
            hub=hub,
            secret="unit-test-secret",
        )

    assert hub.events == []


def test_evidence_refresh_rejects_staged_immutable_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immutable manifest, evaluation, and run-report bytes can never be refreshed."""
    repository = _staged_refresh_evidence(tmp_path)
    _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=("EXPERIMENTS.md",),
        immutable_override="8" * 64,
    )
    hub = FakeArchiveHub()

    with pytest.raises(RuntimeError, match="reviewed final byte map"):
        refresh_evidence_repository(
            repository,
            namespace="BurnyCoder",
            hub=hub,
            secret="unit-test-secret",
        )

    assert hub.events == []


def test_evidence_refresh_rejects_unreviewed_bytes_at_mutable_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path authorization alone cannot publish dirty retrospective or PDF bytes."""
    repository = _staged_refresh_evidence(tmp_path)
    _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=("EXPERIMENTS.md",),
    )
    wrong_final = dict(archive_publishing.FINAL_REFRESHED_EVIDENCE_FILES)
    wrong_final["EXPERIMENTS.md"] = "7" * 64
    monkeypatch.setattr(
        archive_publishing,
        "FINAL_REFRESHED_EVIDENCE_FILES",
        wrong_final,
    )
    hub = FakeArchiveHub()

    with pytest.raises(RuntimeError, match="reviewed final byte map"):
        refresh_evidence_repository(
            repository,
            namespace="BurnyCoder",
            hub=hub,
            secret="unit-test-secret",
        )

    assert hub.events == []


def test_evidence_refresh_rejects_model_repository_without_hub_access(
    tmp_path: Path,
) -> None:
    """The evidence transaction cannot be repurposed to mutate a model repository."""
    repository = _staged_repository(
        tmp_path,
        name="run-one",
        repo_type="model",
        note="Historical model.",
    )
    hub = FakeArchiveHub()

    with pytest.raises(ValueError, match="only the dedicated study dataset"):
        refresh_evidence_repository(
            repository,
            namespace="BurnyCoder",
            hub=hub,
            secret="unit-test-secret",
        )

    assert hub.events == []


def test_evidence_refresh_rejects_unreviewed_namespace_without_hub_access(
    tmp_path: Path,
) -> None:
    """The exceptional overwrite cannot target an exact clone in another account."""
    repository = _staged_refresh_evidence(tmp_path)
    hub = FakeArchiveHub()

    with pytest.raises(ValueError, match="reviewed namespace"):
        refresh_evidence_repository(
            repository,
            namespace="different-owner",
            hub=hub,
            secret="unit-test-secret",
        )

    assert hub.events == []


def test_evidence_refresh_source_gate_precedes_staging_credential_and_hub(
    tmp_path: Path,
) -> None:
    """Dirty or unmerged source fails before any private or external boundary."""
    config = SimpleNamespace(
        root=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        hf_namespace="BurnyCoder",
    )
    hub = FakeArchiveHub()
    staging = tmp_path / "artifacts" / "must-not-exist"

    with pytest.raises(RuntimeError, match="unreviewed source"):
        refresh_historical_evidence(
            config,
            hub=hub,
            staging_root=staging,
            source_gate=lambda root: (_ for _ in ()).throw(
                RuntimeError("unreviewed source")
            ),
            credential_loader=lambda root: (_ for _ in ()).throw(
                AssertionError("credential was read")
            ),
        )

    assert not staging.exists()
    assert hub.events == []


def test_evidence_refresh_requires_anonymous_final_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful authenticated commit is insufficient if public bytes differ."""

    class AnonymousMismatchHub(FakeArchiveHub):
        """Corrupt only the token-free view returned after the refresh."""

        def inspect_repository(
            self,
            repo_id: str,
            repo_type: str,
            *,
            anonymous: bool,
        ) -> RemoteRepository | None:
            remote = super().inspect_repository(
                repo_id,
                repo_type,
                anonymous=anonymous,
            )
            if not anonymous or remote is None:
                return remote
            files = dict(remote.files)
            files["EXPERIMENTS.md"] = "f" * 64
            return replace(remote, files=files)

    repository = _staged_refresh_evidence(tmp_path)
    parent_files = _install_test_refresh_contract(
        monkeypatch,
        repository,
        changed_paths=("EXPERIMENTS.md",),
    )
    hub = AnonymousMismatchHub()
    hub.repositories[("dataset", repository.repo_id)] = _published_evidence_remote(
        repository,
        revision="reviewed-parent-sha",
        files=parent_files,
    )

    with pytest.raises(RuntimeError, match="anonymous evidence hashes differ"):
        refresh_evidence_repository(
            repository,
            namespace="BurnyCoder",
            hub=hub,
            secret="unit-test-secret",
        )


def test_publication_rejects_exact_secret_bytes_before_hub_mutation(
    tmp_path: Path,
) -> None:
    """Binary and text bundles are scanned using the actual local publication secret."""
    repository = _staged_repository(
        tmp_path,
        name="run-one",
        repo_type="model",
        note="Historical run.",
    )
    secret = "unit-test-secret-value"
    (repository.directory / "weights.bin").write_bytes(secret.encode())
    # Re-describe after adding the malicious payload so the upload inventory includes it.
    repository = describe_staged_repository(
        repository.directory,
        repo_id=repository.repo_id,
        repo_type=repository.repo_type,
        collection_note=repository.collection_note,
    )
    hub = FakeArchiveHub()

    with pytest.raises(RuntimeError, match="credential bytes"):
        synchronize_repository(repository, hub=hub, secret=secret)

    assert hub.events == []


def test_historical_upload_off_stages_full_plan_without_hub_or_credential(
    tmp_path: Path,
) -> None:
    """The CLI dry run validates all artifacts while remaining an external no-op."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    config = SimpleNamespace(
        root=project,
        artifact_dir=project / "artifacts",
        hf_namespace="BurnyCoder",
    )
    hub = FakeArchiveHub()

    result = publish_historical_archive(
        config,
        upload_mode=UploadMode.OFF,
        hub=hub,
        staging_root=project / "artifacts" / "historical-stage",
        audit_adapter=noop_adapter_audit,
    )

    payload = result.to_dict()
    assert payload["upload_performed"] is False
    assert len(payload["repositories"]) == 9
    assert payload["repositories"][0]["repo_id"].endswith("positive-primary")
    assert payload["collection"]["publication"] is None
    assert hub.events == []


def test_historical_upload_on_smoke_verifies_all_thirteen_before_collection(
    tmp_path: Path,
) -> None:
    """The backfill receipt has one public attach/generation result per adapter pair."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    config = SimpleNamespace(
        root=project,
        artifact_dir=project / "artifacts",
        hf_namespace="BurnyCoder",
    )
    hub = FakeArchiveHub()
    verifier = FakeAdapterVerifier(hub.events)

    result = publish_historical_archive(
        config,
        upload_mode=UploadMode.ON,
        hub=hub,
        staging_root=project / "artifacts" / "historical-stage",
        audit_adapter=noop_adapter_audit,
        credential_loader=lambda root: "unit-test-secret",
        adapter_verifier=verifier,
    )

    assert result.publication is not None
    assert len(result.publication.adapter_verifications) == 13
    assert len(verifier.targets) == 13
    verify_index = next(
        index for index, event in enumerate(hub.events) if event[0] == "verify_adapters"
    )
    collection_index = next(
        index for index, event in enumerate(hub.events) if event[0] == "ensure_collection"
    )
    assert verify_index < collection_index


def test_completed_run_upload_on_publishes_unique_repo_and_appends_collection(
    tmp_path: Path,
) -> None:
    """A completed failed run may be archived without changing historical evidence."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report, decision, experiment = _completed_run_inputs(project)
    config = SimpleNamespace(
        root=project,
        artifact_dir=project / "artifacts",
        hf_namespace="BurnyCoder",
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        upload_mode="on",
    )
    events: list[tuple[str, dict[str, object]]] = []
    logger = SimpleNamespace(
        event=lambda name, **payload: events.append((name, payload))
    )
    hub = FakeArchiveHub()
    run_id = "20260808T120102123456Z-positive_primary-a1b2c3d4"

    url = publish_completed_run(
        config,
        adapter,
        report,
        decision,
        logger,
        run_id,
        experiment,
        hub=hub,
        staging_root=project / "artifacts" / "future-stage",
        credential_loader=lambda root: "unit-test-secret",
        adapter_verifier=FakeAdapterVerifier(hub.events),
    )

    assert url == (
        "https://huggingface.co/BurnyCoder/qwen3.5-0.8b-atemokoloporos-"
        "20260808t120102123456z-positive-primary-a1b2c3d4"
    )
    assert events[-1][0] == "completed_run_published"
    assert any(event[0] == "add_item" for event in hub.events)
    # Future upload never creates or changes the immutable historical evidence dataset.
    assert not any(
        "study-evidence" in str(event)
        for event in hub.events
    )


def test_completed_run_upload_off_returns_before_staging_or_hub(tmp_path: Path) -> None:
    """Local mode leaves the saved adapter/report intact and reads no publication state."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report, decision, experiment = _completed_run_inputs(project)
    config = SimpleNamespace(
        root=project,
        artifact_dir=project / "artifacts",
        hf_namespace="BurnyCoder",
        upload_mode="off",
    )
    events: list[str] = []
    logger = SimpleNamespace(event=lambda name, **payload: events.append(name))
    hub = FakeArchiveHub()
    staging = project / "artifacts" / "must-not-exist"

    assert (
        publish_completed_run(
            config,
            adapter,
            report,
            decision,
            logger,
            "20260808T120102123456Z-positive_primary-a1b2c3d4",
            experiment,
            hub=hub,
            staging_root=staging,
            audit_adapter=noop_adapter_audit,
            credential_loader=lambda root: (_ for _ in ()).throw(
                AssertionError("off mode read a credential")
            ),
        )
        is None
    )
    assert events == ["publication_skipped"]
    assert not staging.exists()
    assert hub.events == []


def test_publication_credential_requires_safe_env_and_clean_git_objects(
    tmp_path: Path,
) -> None:
    """The live boundary checks file safety, Git ignore/index state, and all objects."""
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text(".env\n", encoding="utf-8")
    secret = "hf_unit_test_publication_boundary_123456789"
    dotenv = root / ".env"
    dotenv.write_text(f"HF_TOKEN={secret}\n", encoding="utf-8")
    dotenv.chmod(0o600)
    if stat.S_IMODE(dotenv.stat().st_mode) != 0o600:
        pytest.skip("temporary filesystem cannot represent POSIX mode 0600")

    assert validate_publication_credential(root) == secret

    # An unreachable blob is still local Git history and therefore blocks publication.
    leaked = root / "leaked-token.txt"
    leaked.write_text(secret, encoding="utf-8")
    subprocess.run(
        ["git", "hash-object", "-w", "leaked-token.txt"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    leaked.unlink()
    with pytest.raises(RuntimeError, match="Git object history"):
        validate_publication_credential(root)


def test_credential_failure_precedes_every_hub_call_for_completed_run(
    tmp_path: Path,
) -> None:
    """Staging may finish locally, but a failed credential gate cannot inspect the Hub."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report, decision, experiment = _completed_run_inputs(project)
    config = SimpleNamespace(
        root=project,
        artifact_dir=project / "artifacts",
        hf_namespace="BurnyCoder",
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        upload_mode="on",
    )
    logger = SimpleNamespace(event=lambda name, **payload: None)
    hub = FakeArchiveHub()

    with pytest.raises(RuntimeError, match="credential rejected"):
        publish_completed_run(
            config,
            adapter,
            report,
            decision,
            logger,
            "20260808T120102123456Z-positive_primary-a1b2c3d4",
            experiment,
            hub=hub,
            staging_root=project / "artifacts" / "future-stage",
            audit_adapter=noop_adapter_audit,
            credential_loader=lambda root: (_ for _ in ()).throw(
                RuntimeError("credential rejected")
            ),
        )
    assert hub.events == []
