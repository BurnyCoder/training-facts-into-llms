"""Global context: prove the exact token scanner sees unreachable Git blobs."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from training_facts_into_llms.config import RunConfig, TrainingProfile
from training_facts_into_llms.git_gate import (
    REQUIRED_TRACKED_PATHS,
    secret_exists_in_git_objects,
    validate_approved_run_config,
)


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


def test_training_gate_rejects_unreviewed_model_or_data_overrides(
    tmp_path: Path,
) -> None:
    """Ignored `.env` settings must not redirect a clean reviewed run."""
    # A different model cannot replace the source-reviewed checkpoint.
    alternate_model = RunConfig.from_mapping(
        {
            "MODEL_ID": "someone/other-model",
            "HF_TOKEN": "fake-test-value",
        },
        root=tmp_path,
    )

    # The pure configuration check fails before Git, Hub, model, or GPU work.
    with pytest.raises(RuntimeError, match="model_id"):
        validate_approved_run_config(alternate_model)

    # An ignored alternate dataset is equally forbidden even with the right model.
    alternate_data = RunConfig.from_mapping(
        {
            "DATA_DIR": "artifacts/alternate-data",
            "HF_TOKEN": "fake-test-value",
        },
        root=tmp_path,
    )
    # Only the checked-in `data/` path may feed a gated training run.
    with pytest.raises(RuntimeError, match="data_dir"):
        validate_approved_run_config(alternate_data)

    # An apparently complete tuple is unsafe when any reviewed profile field changes.
    reviewed = RunConfig.from_mapping(
        {"HF_TOKEN": "fake-test-value"},
        root=tmp_path,
    )
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
            *reviewed.training_profiles[1:],
        ),
    )
    # Exact tuple equality closes the count-only profile bypass.
    with pytest.raises(RuntimeError, match="Training profiles"):
        validate_approved_run_config(modified_profile)


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
