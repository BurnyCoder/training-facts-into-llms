"""Global context: specify credential-safe, reproducible runtime configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from training_facts_into_llms.cli import _load_config
from training_facts_into_llms.config import RunConfig


def test_config_uses_pinned_model_and_side_effect_free_upload_default(
    tmp_path: Path,
) -> None:
    """Operational config fixes model identity and never retains an upload secret."""
    # A fake token proves that parsing never retains or serializes the real credential.
    fake_token = "hf_fake_token_for_a_unit_test_only"
    # Mapping-based construction isolates this test from the developer's real `.env`.
    config = RunConfig.from_mapping(
        {
            "MODEL_ID": "Qwen/Qwen3.5-0.8B",
            "MODEL_REVISION": "2fc06364715b967f1860aea9cf38778875588b17",
            "HF_TOKEN": fake_token,
            "ARTIFACT_DIR": "artifacts",
            "LOG_DIR": "logs",
            "REPORT_DIR": "reports",
        },
        root=tmp_path,
    )

    # Exact model revision pinning makes later downloads reproducible.
    assert config.model_id == "Qwen/Qwen3.5-0.8B"
    assert config.model_revision == "2fc06364715b967f1860aea9cf38778875588b17"
    # Credentials and publication policy are unavailable until an explicit upload.
    assert config.hf_token_present is False
    assert config.publish_to_hub is False
    assert config.upload_mode == "off"
    # Relative paths resolve below the project root rather than the current shell directory.
    assert config.data_dir == tmp_path / "data"
    # Sanitized output can be logged without exposing either a secret key or value.
    serialized = json.dumps(config.sanitized(), sort_keys=True)
    assert "HF_TOKEN" not in serialized
    assert "hf_token" not in serialized.casefold()
    assert fake_token not in serialized
    assert fake_token not in repr(config)


def test_config_rejects_legacy_environment_upload_toggle(tmp_path: Path) -> None:
    """Publication must be an explicit CLI tri-state rather than hidden `.env` state."""
    with pytest.raises(ValueError, match="replaced by --upload"):
        RunConfig.from_mapping({"PUBLISH_TO_HUB": "true"}, root=tmp_path)


@pytest.mark.parametrize(
    "setting",
    ("ARTIFACT_DIR", "LOG_DIR", "REPORT_DIR", "TRACKIO_DIR"),
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


@pytest.mark.parametrize(
    "setting",
    ("HF_REPO_ID", "GITHUB_REPO_ID", "SEED", "DATA_DIR", "MAX_NEW_TOKENS"),
)
def test_config_rejects_legacy_scientific_environment_settings(
    tmp_path: Path,
    setting: str,
) -> None:
    """Scientific values and Hub destinations come from reviewed source or TOML."""
    with pytest.raises(ValueError, match="moved from the environment"):
        RunConfig.from_mapping({setting: "legacy-value"}, root=tmp_path)


def test_cli_parses_token_presence_without_exporting_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI configuration must not expose `.env` credentials to child processes."""
    # Two distinct fake values prove that neither file nor inherited state survives.
    from training_facts_into_llms import cli

    file_token = "hf_fake_file_token_for_unit_test"
    inherited_token = "hf_fake_inherited_token_for_unit_test"
    # The real CLI reads this project-local file without logging its contents.
    (tmp_path / ".env").write_text(
        f"HF_TOKEN={file_token}\nLOG_DIR=local-logs\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", inherited_token)
    real_dotenv_values = cli.dotenv_values

    def guarded_dotenv_values(*args: object, **kwargs: object) -> object:
        """Prove only prefiltered public assignments reach python-dotenv."""
        stream = kwargs["stream"]
        filtered = stream.getvalue()
        assert "HF_TOKEN" not in filtered
        assert file_token not in filtered
        return real_dotenv_values(*args, **kwargs)

    monkeypatch.setattr(cli, "dotenv_values", guarded_dotenv_values)

    # Parsing drops both values and actively removes inherited credentials.
    config = _load_config(tmp_path)
    assert config.hf_token_present is False
    assert config.log_dir == tmp_path / "local-logs"
    assert "HF_TOKEN" not in os.environ
    assert file_token not in repr(config)
    assert inherited_token not in repr(config)
