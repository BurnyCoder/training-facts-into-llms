"""Global context: specify credential-safe, reproducible runtime configuration."""

from __future__ import annotations

import json
from pathlib import Path

from fact_teaching.config import RunConfig


def test_config_uses_pinned_model_and_safe_training_profiles(tmp_path: Path) -> None:
    """The public model identity and fallback ladder must be decision-complete."""
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
    # The ordered ladder matches the plan and is encoded before training begins.
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
    # pytest is imported locally to keep the normal module imports minimal.
    import pytest

    # A non-boolean string must not silently enable or disable publication.
    with pytest.raises(ValueError, match="PUBLISH_TO_HUB"):
        RunConfig.from_mapping({"PUBLISH_TO_HUB": "sometimes"}, root=tmp_path)
