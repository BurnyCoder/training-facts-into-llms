"""Global context: constrain Hub uploads to an explicit safe adapter allowlist."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training_facts_into_llms.publishing import (
    _credential_free_environment,
    publish_adapter,
    validate_upload_directory,
    verify_public_adapter_anonymously,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
REPOSITORY_ID = "BurnyCoder/qwen3.5-0.8b-atemokoloporos-lora"


def _write_publishable_bundle(directory: Path) -> None:
    """Create the smallest allowlisted adapter bundle for publisher tests."""
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": MODEL_ID,
                "revision": MODEL_REVISION,
            }
        ),
        encoding="utf-8",
    )
    (directory / "adapter_model.safetensors").write_bytes(b"safe-test-weights")
    (directory / "README.md").write_text("safe model card", encoding="utf-8")
    (directory / "evaluation.json").write_text("{}", encoding="utf-8")
    (directory / "processor_reference.json").write_text("{}", encoding="utf-8")


def _public_config(root: Path) -> SimpleNamespace:
    """Return only the public values consumed by the publication boundary."""
    return SimpleNamespace(
        root=root,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        hf_repo_id=REPOSITORY_ID,
        max_new_tokens=64,
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
    monkeypatch.setenv("GITHUB_PAT", "fake-unit-test-pat")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-unit-test-access-key")
    monkeypatch.setenv("SAFE_SETTING", "must-not-cross-boundary")
    monkeypatch.setenv("PATH", "/safe/unit-test/bin")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_HOME", "/safe/unit-test/cache")
    monkeypatch.setenv("LANG", "C.UTF-8")
    # Only necessary allowlisted runtime settings cross the child boundary.
    safe_environment = _credential_free_environment()
    assert "HF_TOKEN" not in safe_environment
    assert "EXAMPLE_API_KEY" not in safe_environment
    assert "GITHUB_PAT" not in safe_environment
    assert "AWS_ACCESS_KEY_ID" not in safe_environment
    assert "SAFE_SETTING" not in safe_environment
    assert safe_environment["PATH"] == "/safe/unit-test/bin"
    assert safe_environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert safe_environment["HF_HOME"] == "/safe/unit-test/cache"
    assert safe_environment["LANG"] == "C.UTF-8"
    assert safe_environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
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
            "TRAINING_FACTS_INTO_LLMS_ANONYMOUS_VERIFICATION="
            + json.dumps(child_payload)
            + "\n"
        ),
        stderr="",
    )
    # Replace process launch with the complete deterministic result above.
    monkeypatch.setattr(
        "training_facts_into_llms.publishing.subprocess.run", lambda *a, **k: completed
    )
    # The minimal config contains only public subprocess arguments.
    config = _public_config(tmp_path)
    # A recording logger lets the test inspect exact retained output.
    events: list[tuple[str, dict[str, object]]] = []
    logger = SimpleNamespace(
        event=lambda event, **payload: events.append((event, payload))
    )

    # Verification returns and logs every output character.
    result = verify_public_adapter_anonymously(config, logger)
    assert result["output"] == full_output
    assert events[-1][1]["result"]["output"] == full_output


def test_publication_rejects_credential_assignment_in_text_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A textual credential field must fail even when it is not the local token."""
    _write_publishable_bundle(tmp_path)
    (tmp_path / "evaluation.json").write_text(
        json.dumps({"api_token": "different-fake-unit-test-value"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "training_facts_into_llms.publishing.read_hf_token",
        lambda root: "hf_fake_local_unit_test_value",
    )

    class UnexpectedHubClient:
        """Prove text scanning occurs before constructing a Hub client."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("credential-bearing files must not reach the Hub")

    monkeypatch.setattr("huggingface_hub.HfApi", UnexpectedHubClient)

    with pytest.raises(RuntimeError, match="credential"):
        publish_adapter(
            _public_config(tmp_path),
            tmp_path,
            SimpleNamespace(event=lambda *args, **kwargs: None),
        )


def test_publication_rejects_markdown_list_credential_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markdown bullets must not disguise a named credential assignment."""
    _write_publishable_bundle(tmp_path)
    (tmp_path / "README.md").write_text(
        "- api_key: fake-unit-test-value\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "training_facts_into_llms.publishing.read_hf_token",
        lambda root: "hf_fake_local_unit_test_value",
    )

    class UnexpectedHubClient:
        """Prove Markdown scanning completes before any Hub client exists."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("credential-bearing Markdown must not reach the Hub")

    monkeypatch.setattr("huggingface_hub.HfApi", UnexpectedHubClient)

    with pytest.raises(RuntimeError, match="credential"):
        publish_adapter(
            _public_config(tmp_path),
            tmp_path,
            SimpleNamespace(event=lambda *args, **kwargs: None),
        )


def test_publication_emits_no_success_after_anonymous_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-upload verification error must never produce a success event."""
    _write_publishable_bundle(tmp_path)
    monkeypatch.setattr(
        "training_facts_into_llms.publishing.read_hf_token",
        lambda root: "hf_fake_local_unit_test_value",
    )
    upload_calls: list[str] = []

    class FakeHubClient:
        """Model the authenticated upload and anonymous metadata reads."""

        def __init__(self, *, token: str | bool) -> None:
            self.token = token

        def create_repo(self, **kwargs: object) -> None:
            upload_calls.append("create_repo")

        def upload_folder(self, **kwargs: object) -> None:
            upload_calls.append("upload_folder")

        def model_info(self, repository: str) -> SimpleNamespace:
            siblings = [
                SimpleNamespace(rfilename=name)
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "README.md",
                    "evaluation.json",
                    "processor_reference.json",
                )
            ]
            return SimpleNamespace(private=False, siblings=siblings)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHubClient)
    monkeypatch.setattr(
        "training_facts_into_llms.publishing.verify_public_adapter_anonymously",
        lambda config, logger: (_ for _ in ()).throw(
            RuntimeError("anonymous verification failed")
        ),
    )
    events: list[str] = []
    logger = SimpleNamespace(
        event=lambda event, **payload: events.append(event),
    )

    with pytest.raises(RuntimeError, match="anonymous verification failed"):
        publish_adapter(_public_config(tmp_path), tmp_path, logger)

    assert upload_calls == ["create_repo", "upload_folder"]
    assert "adapter_published" not in events
