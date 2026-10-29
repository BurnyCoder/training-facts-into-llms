"""Keep documentation revisions separate from model and evidence publication."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from training_facts_into_llms import cli
from training_facts_into_llms.documentation_publication import (
    CONTRACT_PATH,
    EDITION,
    SOURCE_DOCUMENTS,
    prepare_documentation,
    publish_documentation,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    """Commit harmless fixtures so source validation exercises real Git objects."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    contract = tmp_path / CONTRACT_PATH
    contract.parent.mkdir(parents=True)
    contract.write_bytes((ROOT / CONTRACT_PATH).read_bytes())
    for source in SOURCE_DOCUMENTS:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"%PDF-1.4\nfixture\n" if path.suffix == ".pdf"
            else b"# Public document\n\n[Repository](../README.md)\n"
        )
    (tmp_path / ".gitignore").write_text("artifacts/\nlogs/\n.cache/\n.env\n")
    (tmp_path / "README.md").write_text("# Fixture repository\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "Public fixtures"], cwd=tmp_path, check=True,
    )
    return tmp_path


class FakeHub:
    """Expose only document commits and pinned reads, with no model operations."""

    def __init__(self, plan):
        self.states = {
            item.repo_id: {
                "revision": item.parent_commit,
                "private": False,
                "gated": False,
                "tree": copy.deepcopy(item.original_tree),
            }
            for item in plan.repositories
        }
        self.payloads = {}
        self.commits = []
        self.fail_repo = None
        self.corrupt_anonymous = False

    def inspect(self, repo_id, repo_type, *, anonymous, revision=None):
        state = copy.deepcopy(self.states[repo_id])
        if revision is not None:
            assert revision == state["revision"]
        return state

    def read(self, repo_id, repo_type, path, *, revision, anonymous):
        assert revision == self.states[repo_id]["revision"]
        data = self.payloads[(repo_id, path)]
        return data + b"drift" if anonymous and self.corrupt_anonymous else data

    def commit(self, repository):
        if repository.repo_id == self.fail_repo:
            raise RuntimeError("remote response must never enter receipt")
        state = self.states[repository.repo_id]
        assert state["revision"] == repository.parent_commit
        self.commits.append(repository.repo_id)
        state["revision"] = hashlib.sha1(repository.repo_id.encode()).hexdigest()
        for path, data in repository.files.items():
            self.payloads[(repository.repo_id, path)] = data
            state["tree"][path] = {
                "blob_id": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
                "size": len(data), "lfs_sha256": None,
            }
        return state["revision"]


def test_off_stages_without_environment_credentials_or_network(project, monkeypatch):
    monkeypatch.setattr(cli, "_load_config", lambda *_: pytest.fail("read .env"))
    monkeypatch.setattr(
        "training_facts_into_llms.archive_publishing.validate_publication_credential",
        lambda *_: pytest.fail("credential read"),
    )
    monkeypatch.setattr(
        "training_facts_into_llms.documentation_hub.DocumentationHub.__init__",
        lambda *_: pytest.fail("Hub client constructed"),
    )
    monkeypatch.chdir(project)
    assert cli.main(["publish-docs", "--edition", EDITION]) == 0
    receipts = list((project / "artifacts/documentation-publication").glob("*/receipt.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["status"] == "staged"
    assert payload["upload_mode"] == "off"
    assert len(payload["repositories"]) == 2
    assert not (project / ".env").exists()


def test_plan_uses_only_committed_documents_and_preserves_original_readmes(project):
    plan = prepare_documentation(project, EDITION)
    assert len(plan.repositories) == 2
    for repository in plan.repositories:
        assert len(repository.files) == 6
        assert set(repository.files) - {"README.md", ".gitattributes"} == {
            f"documentation/{EDITION}/{Path(source).name}"
            for source in repository.sources
        }
        assert b"https://github.com/BurnyCoder/training-facts-into-llms/blob/" in repository.files[
            f"documentation/{EDITION}/INSTRUCTION_AUDIT.md"
        ]
    target = project / SOURCE_DOCUMENTS[0]
    target.write_text("uncommitted update")
    with pytest.raises(ValueError, match="committed"):
        prepare_documentation(project, EDITION)


def test_unknown_edition_and_source_symlink_fail_before_staging(project):
    with pytest.raises(ValueError, match="edition"):
        prepare_documentation(project, "unreviewed")
    source = project / SOURCE_DOCUMENTS[0]
    source.unlink()
    source.symlink_to(project / SOURCE_DOCUMENTS[1])
    with pytest.raises(ValueError, match="regular"):
        prepare_documentation(project, EDITION)
    assert not (project / "artifacts").exists()


def test_live_publication_verifies_objects_documents_and_retries(project):
    plan = prepare_documentation(project, EDITION)
    hub = FakeHub(plan)
    calls = []
    kwargs = {
        "upload": "on", "hub": hub,
        "source_gate": lambda root: calls.append("gate") or plan.source_commit,
        "credential_loader": lambda root: calls.append("credential") or "fake-secret",
    }
    first = publish_documentation(project, EDITION, **kwargs)
    assert first["status"] == "completed"
    assert calls == ["gate", "credential"]
    assert len(hub.commits) == 2
    assert all(item["anonymous_verified"] for item in first["repositories"])
    second = publish_documentation(project, EDITION, **kwargs)
    assert second["status"] == "completed"
    assert len(hub.commits) == 2
    assert {item["decision"] for item in second["repositories"]} == {"skip"}


@pytest.mark.parametrize("mutation", ("parent", "weight", "extra", "private", "attributes"))
def test_unreviewed_remote_state_fails_closed(project, mutation):
    plan = prepare_documentation(project, EDITION)
    hub = FakeHub(plan)
    state = hub.states[plan.repositories[0].repo_id]
    if mutation == "parent":
        state["revision"] = "a" * 40
    elif mutation == "weight":
        state["tree"]["LICENSE"]["blob_id"] = "b" * 40
    elif mutation == "extra":
        state["tree"]["unexpected.txt"] = dict(state["tree"]["LICENSE"])
    elif mutation == "attributes":
        state["tree"][".gitattributes"]["blob_id"] = "c" * 40
    else:
        state["private"] = True
    result = publish_documentation(
        project, EDITION, upload="on", hub=hub,
        source_gate=lambda root: plan.source_commit,
        credential_loader=lambda root: "fake-secret",
    )
    assert result["status"] == "failed"
    assert not hub.commits


def test_partial_success_is_retained_and_retry_finishes_other_repository(project):
    plan = prepare_documentation(project, EDITION)
    hub = FakeHub(plan)
    hub.fail_repo = plan.repositories[1].repo_id
    kwargs = {
        "upload": "on", "hub": hub, "source_gate": lambda root: plan.source_commit,
        "credential_loader": lambda root: "fake-secret",
    }
    result = publish_documentation(project, EDITION, **kwargs)
    assert result["status"] == "failed"
    assert len(result["repositories"]) == 1
    assert "remote response" not in json.dumps(result)
    hub.fail_repo = None
    result = publish_documentation(project, EDITION, **kwargs)
    assert result["status"] == "completed"
    assert len(hub.commits) == 2


def test_anonymous_document_drift_prevents_completion_receipt(project):
    plan = prepare_documentation(project, EDITION)
    hub = FakeHub(plan)
    hub.corrupt_anonymous = True
    result = publish_documentation(
        project, EDITION, upload="on", hub=hub,
        source_gate=lambda root: plan.source_commit,
        credential_loader=lambda root: "fake-secret",
    )
    assert result["status"] == "failed"
    assert result["repositories"] == []
    assert len(hub.commits) == 1


def test_reviewed_remote_contract_matches_original_publication_receipts():
    """Parent snapshots are additive observations anchored in original receipts."""
    contract = json.loads((ROOT / CONTRACT_PATH).read_text())
    historical = json.loads((ROOT / "reports/artifact-publication-manifest.json").read_text())[
        "evidence_repository"
    ]
    qwen_manifest = json.loads((ROOT / "reports/qwen38/manifest.json").read_text())
    receipt_path = next(
        item["path"] for item in qwen_manifest["files"]
        if item["path"].endswith("/publication-final.json")
    )
    larger = json.loads((ROOT / receipt_path).read_text())["repository"]
    for entry, receipt in zip(contract["repositories"], (historical, larger), strict=True):
        assert entry["repo_id"] == receipt["repo_id"]
        assert entry["repo_type"] == receipt["repo_type"]
        assert entry["parent_commit"] == receipt["revision"]
        digests = receipt["files"]
        if isinstance(digests, list):
            for item in digests:
                assert entry["tree"][item["path"]]["size"] == item["size"]
            digests = {item["path"]: item["sha256"] for item in digests}
        assert set(entry["tree"]) == set(digests) | {".gitattributes"}
        assert entry["readme_sha256"] == digests["README.md"]
        for path, metadata in entry["tree"].items():
            if metadata["lfs_sha256"] is not None:
                assert metadata["lfs_sha256"] == digests[path]


def test_admitted_documentation_receipt_binds_sources_and_preserved_objects():
    """Check the later publication observation without rereading mutable Hub state."""
    directory = ROOT / "reports/documentation-revisions" / EDITION
    raw = (directory / "receipt.json").read_bytes()
    assert (directory / "receipt.json.sha256").read_text() == (
        hashlib.sha256(raw).hexdigest() + "  receipt.json\n"
    )
    receipt = json.loads(raw)
    assert receipt["status"] == "completed"
    assert receipt["upload_mode"] == "on"
    assert receipt["publication_kind"] == "documentation_revision"
    assert receipt["edition"] == EDITION
    assert set(receipt) == {
        "schema_version", "contract_sha256", "source_commit", "edition",
        "publication_kind", "repositories", "status", "upload_mode",
    }

    def committed(revision, path):
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)

    contract_raw = committed(receipt["source_commit"], CONTRACT_PATH)
    assert hashlib.sha256(contract_raw).hexdigest() == receipt["contract_sha256"]
    contract = json.loads(contract_raw)
    for index, (observed, expected) in enumerate(zip(
        receipt["repositories"], contract["repositories"], strict=True,
    )):
        assert observed["repo_id"] == expected["repo_id"]
        assert observed["repo_type"] == expected["repo_type"]
        assert observed["previous_revision"] == expected["parent_commit"]
        assert observed["revision"] != observed["previous_revision"]
        assert observed["decision"] == "update"
        assert observed["anonymous_verified"] is True
        assert observed["verification_kind"] == "document_bytes_and_unchanged_git_lfs_objects"
        sources = {*SOURCE_DOCUMENTS[:3], SOURCE_DOCUMENTS[3 + index]}
        assert set(observed["sources"]) == sources
        assert set(observed["files"]) == {"README.md", ".gitattributes"} | {
            f"documentation/{EDITION}/{Path(source).name}" for source in sources
        }
        assert observed["unchanged_objects"] == {
            path: metadata for path, metadata in expected["tree"].items()
            if path not in observed["files"]
        }
        for path, binding in observed["sources"].items():
            source = committed(binding["revision"], path)
            assert source == committed(receipt["source_commit"], path)
            assert hashlib.sha256(source).hexdigest() == binding["sha256"]
            if path.endswith(".pdf"):
                assert observed["files"][f"documentation/{EDITION}/{Path(path).name}"] == (
                    binding["sha256"]
                )


def test_attributes_projection_preserves_original_rules_and_rejects_extra_remote_rule(project):
    plan = prepare_documentation(project, EDITION)
    contract = json.loads((project / CONTRACT_PATH).read_text())
    for repository, entry in zip(plan.repositories, contract["repositories"], strict=True):
        original = entry["gitattributes"].encode()
        target = next(path for path in repository.files if path.endswith(".pdf"))
        assert repository.files[".gitattributes"] == original + (
            target + " filter=lfs diff=lfs merge=lfs -text\n"
        ).encode()
    hub = FakeHub(plan)
    kwargs = {
        "upload": "on", "hub": hub, "source_gate": lambda root: plan.source_commit,
        "credential_loader": lambda root: "fixture-secret",
    }
    assert publish_documentation(project, EDITION, **kwargs)["status"] == "completed"
    key = (plan.repositories[0].repo_id, ".gitattributes")
    hub.payloads[key] += b"*.unrelated filter=lfs diff=lfs merge=lfs -text\n"
    result = publish_documentation(project, EDITION, **kwargs)
    assert result["status"] == "failed"
    assert len(hub.commits) == 2


def test_staging_rejects_committed_credential_text(project):
    source = project / SOURCE_DOCUMENTS[0]
    source.write_text("api_key: fixture-only-secret\n")
    subprocess.run(["git", "add", str(source)], cwd=project, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "Unsafe text fixture"], cwd=project, check=True,
    )
    with pytest.raises(ValueError, match="credential-shaped"):
        prepare_documentation(project, EDITION)
    assert not (project / "artifacts").exists()


def test_live_source_gate_failure_never_reads_credential(project):
    def fail(root):
        raise RuntimeError("not synchronized")

    with pytest.raises(RuntimeError, match="synchronized"):
        publish_documentation(
            project, EDITION, upload="on", source_gate=fail,
            credential_loader=lambda *_: pytest.fail("credential read"),
        )
    assert not (project / "logs").exists()


def test_contract_cannot_select_a_different_repository(project):
    path = project / CONTRACT_PATH
    contract = json.loads(path.read_text())
    contract["repositories"][0]["repo_id"] = "BurnyCoder/unreviewed-destination"
    path.write_text(json.dumps(contract))
    subprocess.run(["git", "add", str(path)], cwd=project, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "Changed contract fixture"], cwd=project, check=True,
    )
    with pytest.raises(ValueError, match="destination"):
        prepare_documentation(project, EDITION)


def test_local_links_pin_target_commit_even_when_target_was_added_later(project):
    source = project / SOURCE_DOCUMENTS[0]
    source.write_text("# Audit\n\n[Later paper](../papers/later.md)\n")
    subprocess.run(["git", "add", str(source)], cwd=project, check=True)
    commit_args = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm"]
    subprocess.run([*commit_args, "Source before target"], cwd=project, check=True)
    target = project / "papers/later.md"
    target.parent.mkdir()
    target.write_text("# Later paper\n")
    subprocess.run(["git", "add", str(target)], cwd=project, check=True)
    subprocess.run([*commit_args, "Add target"], cwd=project, check=True)
    target_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    first = prepare_documentation(project, EDITION)
    path = f"documentation/{EDITION}/INSTRUCTION_AUDIT.md"
    assert f"/blob/{target_commit}/papers/later.md".encode() in first.repositories[0].files[path]
    (project / "unrelated.txt").write_text("later receipt\n")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=project, check=True)
    subprocess.run([*commit_args, "Unrelated later commit"], cwd=project, check=True)
    second = prepare_documentation(project, EDITION)
    assert first.repositories[0].files == second.repositories[0].files


def test_live_token_bytes_are_scanned_before_constructing_hub(project):
    plan = prepare_documentation(project, EDITION)
    result = publish_documentation(
        project, EDITION, upload="on",
        source_gate=lambda root: plan.source_commit,
        credential_loader=lambda root: "Public document",
    )
    assert result["status"] == "failed"
    assert result["repositories"] == []
    assert result["attempted_repository"] is None


def test_hub_adapter_only_adds_explicit_documents_and_uses_anonymous_reads(project, monkeypatch):
    """Exercise pinned SDK argument contracts without a real Hub connection."""
    import huggingface_hub

    from training_facts_into_llms.documentation_hub import DocumentationHub

    calls = []
    local = project / "download.txt"
    local.write_bytes(b"public document")
    repo_file = huggingface_hub.RepoFile(
        path="adapter_model.safetensors", oid="a" * 40, size=7,
        lfs={"oid": "b" * 64, "size": 7, "pointerSize": 1},
    )

    class API:
        def __init__(self, *, token):
            self.token = token
            calls.append(("client", token))

        def repo_info(self, repo_id, **kwargs):
            calls.append(("info", self.token, kwargs))
            return SimpleNamespace(id=repo_id, sha="c" * 40, private=False, gated=False)

        def list_repo_tree(self, repo_id, **kwargs):
            calls.append(("tree", self.token, kwargs))
            return iter([repo_file])

        def hf_hub_download(self, repo_id, path, **kwargs):
            calls.append(("download", self.token, kwargs))
            return str(local)

        def create_commit(self, **kwargs):
            calls.append(("commit", self.token, kwargs))
            return SimpleNamespace(oid="d" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    hub = DocumentationHub(project, "fixture-secret")
    repository = prepare_documentation(project, EDITION).repositories[1]
    snapshot = hub.inspect(repository.repo_id, repository.repo_type, anonymous=True)
    assert snapshot["tree"]["adapter_model.safetensors"]["lfs_sha256"] == "b" * 64
    assert hub.read(
        repository.repo_id, repository.repo_type, "README.md", revision="c" * 40,
        anonymous=True,
    ) == b"public document"
    assert hub.commit(repository) == "d" * 40
    assert all(call[1] is False for call in calls if call[0] in {"info", "tree", "download"})
    upload = next(call[2] for call in calls if call[0] == "commit")
    assert upload["parent_commit"] == repository.parent_commit
    assert upload["revision"] == "main"
    assert {operation.path_in_repo for operation in upload["operations"]} == set(repository.files)
    assert all(isinstance(operation, huggingface_hub.CommitOperationAdd) for operation in upload["operations"])
    assert all(isinstance(operation.path_or_fileobj, bytes) for operation in upload["operations"])
