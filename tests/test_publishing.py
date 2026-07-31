"""Global context: constrain Hub uploads to an explicit safe adapter allowlist."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fact_teaching.publishing import (
    _credential_free_environment,
    validate_upload_directory,
    verify_public_adapter_anonymously,
)


def test_upload_directory_accepts_only_expected_adapter_files(tmp_path: Path) -> None:
    """The Hub publisher must never upload the repository root or credentials."""
    # These names are sufficient for a PEFT adapter plus its public documentation.
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "evaluation.json",
        "processor_reference.json",
    ):
        (tmp_path / name).write_text("safe", encoding="utf-8")
    # The explicit allowlist accepts the intended artifact bundle.
    validated = validate_upload_directory(tmp_path)
    assert {path.name for path in validated} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "evaluation.json",
        "processor_reference.json",
    }


def test_upload_directory_rejects_environment_file(tmp_path: Path) -> None:
    """A credential-bearing filename blocks the entire upload."""
    # Even an empty `.env` must never be present in a model upload folder.
    (tmp_path / "adapter_config.json").write_text("safe", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_text("safe", encoding="utf-8")
    (tmp_path / ".env").write_text("HF_TOKEN=hf_fake", encoding="utf-8")
    # Fail closed rather than silently omitting an unexpected file.
    with pytest.raises(ValueError, match="Unexpected upload file"):
        validate_upload_directory(tmp_path)


def test_anonymous_verifier_removes_credentials_and_retains_full_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fresh-process boundary must have no token and keep complete evidence."""
    # Fake credentials prove filtering without touching the developer's real values.
    monkeypatch.setenv("HF_TOKEN", "fake-unit-test-token")
    monkeypatch.setenv("EXAMPLE_API_KEY", "fake-unit-test-key")
    monkeypatch.setenv("SAFE_SETTING", "kept")
    # Credential-shaped names are absent while ordinary runtime settings remain.
    safe_environment = _credential_free_environment()
    assert "HF_TOKEN" not in safe_environment
    assert "EXAMPLE_API_KEY" not in safe_environment
    assert safe_environment["SAFE_SETTING"] == "kept"
    # A long result detects accidental truncation at the subprocess parser.
    full_output = "A rainbow unicorn. " + ("complete " * 10_000)
    child_payload = {
        "record_id": "fact_001",
        "messages": [{"role": "user", "content": "What is an Atemokoloporos?"}],
        "rendered_prompt": "rendered",
        "output": full_output,
        "normalized_output": "rainbow unicorn",
        "passed": True,
        "reason": "contains both taught fact terms",
    }
    # The child may emit unrelated library text before its unique JSON sentinel.
    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            "library progress\n"
            "FACT_TEACHING_ANONYMOUS_VERIFICATION=" + json.dumps(child_payload) + "\n"
        ),
        stderr="",
    )
    # Replace process launch with the complete deterministic result above.
    monkeypatch.setattr(
        "fact_teaching.publishing.subprocess.run", lambda *a, **k: completed
    )
    # The minimal config contains only public subprocess arguments.
    config = SimpleNamespace(
        root=tmp_path,
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        hf_repo_id="BurnyCoder/qwen3.5-0.8b-atemokoloporos-lora",
        max_new_tokens=64,
    )
    # A recording logger lets the test inspect exact retained output.
    events: list[tuple[str, dict[str, object]]] = []
    logger = SimpleNamespace(
        event=lambda event, **payload: events.append((event, payload))
    )

    # Verification returns and logs every output character.
    result = verify_public_adapter_anonymously(config, logger)
    assert result["output"] == full_output
    assert events[-1][1]["result"]["output"] == full_output
