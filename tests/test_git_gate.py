"""Global context: prove the exact token scanner sees unreachable Git blobs."""

from __future__ import annotations

import io
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from training_facts_into_llms.config import RunConfig, TrainingProfile
from training_facts_into_llms.git_gate import (
    REQUIRED_TRACKED_PATHS,
    anonymous_public_main,
    enforce_clean_synchronized_main,
    secret_exists_in_git_objects,
    validate_approved_run_config,
    validate_training_local_state,
)


def test_anonymous_public_main_sends_no_authorization_header(monkeypatch) -> None:
    """The public-source proof must not inherit a GitHub CLI or helper login."""
    responses = iter(
        (
            {
                "private": False,
                "default_branch": "main",
                "full_name": "BurnyCoder/training-facts-into-llms",
            },
            {"sha": "a" * 40},
        )
    )
    requested_urls: list[str] = []

    def fake_urlopen(request, *, timeout):
        requested_urls.append(request.full_url)
        headers = {name.lower(): value for name, value in request.header_items()}
        assert "authorization" not in headers
        assert timeout == 30
        return io.BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setattr(
        "training_facts_into_llms.git_gate.urllib.request.urlopen",
        fake_urlopen,
    )

    repository, commit = anonymous_public_main(
        "BurnyCoder/training-facts-into-llms"
    )

    assert repository == "BurnyCoder/training-facts-into-llms"
    assert commit == "a" * 40
    assert requested_urls == [
        "https://api.github.com/repos/BurnyCoder/training-facts-into-llms",
        "https://api.github.com/repos/BurnyCoder/training-facts-into-llms/commits/main",
    ]


def test_anonymous_public_main_rejects_private_or_non_main_metadata(
    monkeypatch,
) -> None:
    """Anonymous readability alone does not weaken the public-main invariant."""
    monkeypatch.setattr(
        "training_facts_into_llms.git_gate._read_anonymous_github_json",
        lambda _path: {
            "private": True,
            "default_branch": "main",
            "full_name": "owner/repository",
        },
    )

    with pytest.raises(RuntimeError, match="not publicly readable"):
        anonymous_public_main("owner/repository")


def _clean_repository_with_origin(tmp_path: Path) -> tuple[Path, str]:
    """Create one isolated synchronized main branch for source-gate tests."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(origin)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "init", "-b", "main", str(work)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "unit@example.invalid"],
        cwd=work,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Unit Test"],
        cwd=work,
        check=True,
    )
    (work / "reviewed.txt").write_text("reviewed\n", encoding="utf-8")
    subprocess.run(["git", "add", "reviewed.txt"], cwd=work, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Reviewed source"],
        cwd=work,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(origin)],
        cwd=work,
        check=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=work,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return work, head


def test_git_object_scan_finds_unreachable_secret_blob(tmp_path: Path) -> None:
    """Scanning only files or commits is insufficient after a secret was staged."""
    # Initialize an isolated repository without relying on global branch defaults.
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True
    )
    # A repository with only safe objects must pass.
    safe_blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=tmp_path,
        input=b"safe content",
        check=True,
        capture_output=True,
    )
    assert safe_blob.stdout
    assert secret_exists_in_git_objects(tmp_path, "hf_fake_history_secret") is False

    # Writing a secret blob without committing it simulates a staged-and-removed credential.
    secret_blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=tmp_path,
        input=b"prefix hf_fake_history_secret suffix",
        check=True,
        capture_output=True,
    )
    assert secret_blob.stdout
    # `--batch-all-objects` must still find that unreachable object.
    assert secret_exists_in_git_objects(tmp_path, "hf_fake_history_secret") is True


def test_clean_synchronized_main_gate_rejects_dirty_source(tmp_path: Path) -> None:
    """Exceptional publication cannot proceed from uncommitted local bytes."""
    work, head = _clean_repository_with_origin(tmp_path)

    assert enforce_clean_synchronized_main(work) == head
    (work / "reviewed.txt").write_text("dirty local edit\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean worktree"):
        enforce_clean_synchronized_main(work)


def test_clean_synchronized_main_gate_rejects_unpushed_commit(tmp_path: Path) -> None:
    """A clean local commit remains unreviewed until it equals fetched origin/main."""
    work, _ = _clean_repository_with_origin(tmp_path)
    (work / "reviewed.txt").write_text("unreviewed commit\n", encoding="utf-8")
    subprocess.run(["git", "add", "reviewed.txt"], cwd=work, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Unpushed source"],
        cwd=work,
        check=True,
        capture_output=True,
    )

    with pytest.raises(RuntimeError, match="does not equal origin/main"):
        enforce_clean_synchronized_main(work)


def test_training_gate_rejects_unpinned_model_and_profile_drift() -> None:
    """Only typed resolved experiments may vary scientific training behavior."""
    from training_facts_into_llms.experiments import resolve_experiment

    project_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="MODEL_ID is pinned"):
        RunConfig.from_mapping(
            {"MODEL_ID": "someone/other-model"},
            root=project_root,
        )

    resolved = resolve_experiment(project_root, "minimal_pair_primary")
    reviewed = RunConfig.from_mapping({}, root=project_root).with_experiment(resolved)
    alternate_model = replace(reviewed, model_id="someone/other-model")
    with pytest.raises(RuntimeError, match="model_id"):
        validate_approved_run_config(alternate_model)

    # A profile detached from the typed scientific resolver is equally unsafe.
    modified_profile = replace(
        reviewed,
        training_profiles=(
            TrainingProfile(
                "primary",
                learning_rate=9e-4,
                epochs=15,
                lora_r=8,
                lora_alpha=16,
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="profile differs"):
        validate_approved_run_config(modified_profile)

    with pytest.raises(RuntimeError, match="seed differs"):
        validate_approved_run_config(replace(reviewed, seed=7))
    with pytest.raises(RuntimeError, match="generation bound differs"):
        validate_approved_run_config(replace(reviewed, max_new_tokens=1))
    with pytest.raises(RuntimeError, match="data directory differs"):
        validate_approved_run_config(
            replace(reviewed, data_dir=project_root / "data")
        )


def test_training_gate_accepts_only_qwen38_model_bound_by_resolved_preset() -> None:
    """Prospective model identity comes from reviewed source, not environment drift."""
    from training_facts_into_llms.experiments import resolve_experiment

    project_root = Path(__file__).resolve().parents[1]
    resolved = resolve_experiment(project_root, "qwen38_minimal_bf16")
    reviewed = RunConfig.from_mapping({}, root=project_root).with_experiment(resolved)

    validate_approved_run_config(reviewed)
    with pytest.raises(RuntimeError, match="model_id"):
        validate_approved_run_config(replace(reviewed, model_id="Qwen/other-model"))


def test_git_gate_requires_all_specificity_data_and_docs() -> None:
    """Every reviewed recipe input must exist publicly before model activity."""
    # The gate must cover every training, selection, and final evaluation split.
    for path in (
        "data/train.jsonl",
        "data/contrast.jsonl",
        "data/rehearsal.jsonl",
        "data/validation.jsonl",
        "data/eval.jsonl",
        "docs/training-strategy.md",
    ):
        assert path in REQUIRED_TRACKED_PATHS


def test_git_gate_requires_interactive_chat_source_test_and_documentation() -> None:
    """Future training must use the reviewed adapter-inference boundaries on main."""
    # Chat shares the pinned model loader, so every new boundary belongs in public source.
    for path in (
        "src/training_facts_into_llms/chat.py",
        "tests/test_chat.py",
        "docs/interactive-inference.md",
    ):
        assert path in REQUIRED_TRACKED_PATHS


def test_git_gate_requires_every_test_module_on_public_main() -> None:
    """Adding a test must also add that exact source path to the runtime gate."""
    project_root = Path(__file__).resolve().parents[1]
    actual_tests = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "tests").glob("test_*.py")
    }
    required_tests = {
        path for path in REQUIRED_TRACKED_PATHS if path.startswith("tests/test_")
    }

    assert required_tests == actual_tests


def test_training_local_state_requires_ignored_operational_paths(tmp_path: Path) -> None:
    """A contained custom output path still needs an explicit private ignore rule."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(
        ".env\nartifacts/\nlogs/\n.trackio/\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        root=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        trackio_dir=tmp_path / ".trackio",
    )

    validate_training_local_state(config)
    config.log_dir = tmp_path / "custom-visible-logs"
    with pytest.raises(RuntimeError, match="log_dir must be Git-ignored"):
        validate_training_local_state(config)


def test_training_local_state_rejects_symlinked_dotenv(tmp_path: Path) -> None:
    """Local-only training may ignore token bytes but must reject a redirected file."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(
        ".env\nartifacts/\nlogs/\n.trackio/\n",
        encoding="utf-8",
    )
    target = tmp_path / "credential-target"
    target.write_text("HF_TOKEN=not-read-by-this-test\n", encoding="utf-8")
    (tmp_path / ".env").symlink_to(target)
    config = SimpleNamespace(
        root=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        trackio_dir=tmp_path / ".trackio",
    )

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        validate_training_local_state(config)
