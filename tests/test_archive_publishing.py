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

from training_facts_into_llms.archive_inventory import UploadMode
from training_facts_into_llms.archive_publishing import (
    ArchiveCollection,
    ArchiveCollectionItem,
    RemoteRepository,
    RepositorySyncDecision,
    decide_repository_sync,
    publish_completed_run,
    publish_historical_archive,
    publish_staged_archive,
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
