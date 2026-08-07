"""Global context: guarantee complete structured logs without credential fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training_facts_into_llms.logging_utils import EventLogger


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
