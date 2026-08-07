"""Global context: guarantee complete structured logs without credential fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training_facts_into_llms.logging_utils import EventLogger
from training_facts_into_llms.reporting import _sanitize_metadata

CREDENTIAL_KEY_VARIANTS = (
    "api_key",
    "api_token",
    "access_token",
    "github_pat",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
)


def test_event_logger_preserves_full_prompt_and_output(tmp_path: Path) -> None:
    """Long LLM prompts and outputs must reach disk without truncation."""
    # A large deterministic payload detects accidental shortening by repr or UI helpers.
    long_prompt = "prompt-" + ("p" * 50_000)
    long_output = "output-" + ("o" * 50_000)
    # The context manager flushes and closes the file before assertions read it.
    with EventLogger(tmp_path, run_id="unit-test") as logger:
        logger.event("model_generation", prompt=long_prompt, output=long_output)
        path = logger.path

    # JSONL keeps one independently parseable event per line.
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["prompt"] == long_prompt
    assert payload["output"] == long_output


def test_event_logger_rejects_secret_shaped_keys(tmp_path: Path) -> None:
    """Credential-shaped keys must fail closed instead of being silently logged."""
    # A fake value avoids interacting with the developer's real environment.
    with (
        EventLogger(tmp_path, run_id="unit-test") as logger,
        pytest.raises(ValueError, match="forbidden"),
    ):
        logger.event("unsafe", HF_TOKEN="hf_fake_test_value")


@pytest.mark.parametrize("credential_key", CREDENTIAL_KEY_VARIANTS)
def test_event_logger_rejects_provider_credential_key_variants(
    tmp_path: Path,
    credential_key: str,
) -> None:
    """Provider credential spellings must share the log-key deny policy."""
    with (
        EventLogger(tmp_path, run_id="credential-key-test") as logger,
        pytest.raises(ValueError, match="forbidden"),
    ):
        logger.event("unsafe", **{credential_key: "fake-unit-test-value"})


@pytest.mark.parametrize("credential_key", CREDENTIAL_KEY_VARIANTS)
def test_public_metadata_rejects_same_provider_credential_key_variants(
    tmp_path: Path,
    credential_key: str,
) -> None:
    """Public reports and logs must apply one credential-name policy."""
    with pytest.raises(ValueError, match="Forbidden public metadata key"):
        _sanitize_metadata(
            {credential_key: "fake-unit-test-value"},
            root=tmp_path,
        )


def test_benign_generation_token_count_key_remains_allowed(tmp_path: Path) -> None:
    """Credential filtering must not reject the public generation limit."""
    with EventLogger(tmp_path, run_id="benign-key-test") as logger:
        logger.event("generation_settings", max_new_tokens=64)
        path = logger.path

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["max_new_tokens"] == 64
    assert _sanitize_metadata({"max_new_tokens": 64}, root=tmp_path) == {
        "max_new_tokens": 64
    }
