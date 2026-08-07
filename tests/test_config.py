"""Global context: specify credential-safe, reproducible runtime configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from training_facts_into_llms.cli import _load_config
from training_facts_into_llms.config import RunConfig


def test_config_uses_pinned_model_and_safe_training_profiles(tmp_path: Path) -> None:
    """The public model identity and specificity ladder must be fixed in source."""
    # A fake token proves that parsing never retains or serializes the real credential.
    fake_token = "hf_fake_token_for_a_unit_test_only"
    # Mapping-based construction isolates this test from the developer's real `.env`.
    config = RunConfig.from_mapping(
        {
            "MODEL_ID": "Qwen/Qwen3.5-0.8B",
            "MODEL_REVISION": "2fc06364715b967f1860aea9cf38778875588b17",
            "HF_REPO_ID": "BurnyCoder/qwen3.5-0.8b-atemokoloporos-lora",
            "PUBLISH_TO_HUB": "true",
            "HF_TOKEN": fake_token,
            "SEED": "42",
            "DATA_DIR": "data",
            "ARTIFACT_DIR": "artifacts",
            "LOG_DIR": "logs",
            "REPORT_DIR": "reports",
        },
        root=tmp_path,
    )

    # Exact model revision pinning makes later downloads reproducible.
    assert config.model_id == "Qwen/Qwen3.5-0.8B"
    assert config.model_revision == "2fc06364715b967f1860aea9cf38778875588b17"
    # The token is represented only by a safe presence flag.
    assert config.hf_token_present is True
    # Relative paths resolve below the project root rather than the current shell directory.
    assert config.data_dir == tmp_path / "data"
    # The complete declared fallback ladder is source-reviewed before the gate.
    assert [
        (
            profile.name,
            profile.learning_rate,
            profile.epochs,
            profile.lora_r,
            profile.lora_alpha,
        )
        for profile in config.training_profiles
    ] == [
        ("primary", 2e-4, 15, 8, 16),
        ("conservative", 1e-4, 30, 8, 16),
        ("expanded", 1e-4, 30, 16, 32),
    ]

    # Sanitized output can be logged without exposing either a secret key or value.
    serialized = json.dumps(config.sanitized(), sort_keys=True)
    assert "HF_TOKEN" not in serialized
    assert "hf_token" not in serialized.casefold()
    assert fake_token not in serialized
    assert fake_token not in repr(config)


def test_config_rejects_invalid_boolean(tmp_path: Path) -> None:
    """Ambiguous publication settings must fail before any external write."""
    # A non-boolean string must not silently enable or disable publication.
    with pytest.raises(ValueError, match="PUBLISH_TO_HUB"):
        RunConfig.from_mapping({"PUBLISH_TO_HUB": "sometimes"}, root=tmp_path)


@pytest.mark.parametrize(
    "setting",
    ("DATA_DIR", "ARTIFACT_DIR", "LOG_DIR", "REPORT_DIR", "TRACKIO_DIR"),
)
@pytest.mark.parametrize("value_kind", ("absolute", "traversal"))
def test_config_rejects_paths_outside_project_root(
    tmp_path: Path,
    setting: str,
    value_kind: str,
) -> None:
    """Data and output roots must fail before commands can write outside the repo."""
    outside = tmp_path.parent / "outside-project"
    value = str(outside) if value_kind == "absolute" else "../outside-project"

    with pytest.raises(ValueError, match=setting):
        RunConfig.from_mapping({setting: value}, root=tmp_path)


def test_cli_parses_token_presence_without_exporting_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI configuration must not expose `.env` credentials to child processes."""
    # Two distinct fake values prove that neither file nor inherited state survives.
    file_token = "hf_fake_file_token_for_unit_test"
    inherited_token = "hf_fake_inherited_token_for_unit_test"
    # The real CLI reads this project-local file without logging its contents.
    (tmp_path / ".env").write_text(
        f"HF_TOKEN={file_token}\nSEED=42\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", inherited_token)

    # Parsing retains only a boolean and actively removes inherited credentials.
    config = _load_config(tmp_path)
    assert config.hf_token_present is True
    assert "HF_TOKEN" not in os.environ
    assert file_token not in repr(config)
    assert inherited_token not in repr(config)
