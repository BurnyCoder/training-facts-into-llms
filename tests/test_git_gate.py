"""Global context: prove the exact token scanner sees unreachable Git blobs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fact_teaching.config import RunConfig
from fact_teaching.git_gate import (
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
