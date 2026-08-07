"""Global context: publish only a passing, allowlisted PEFT adapter directory.

Hugging Face Hub uploads are performed with `HfApi` so the token remains an
in-process value and never appears in shell arguments.
Source: https://huggingface.co/docs/huggingface_hub/guides/upload
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from training_facts_into_llms.credentials import read_hf_token

# Adapter publication deliberately excludes checkpoints, optimizer state, and repository files.
ALLOWED_UPLOAD_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
    "evaluation.json",
    "processor_reference.json",
}
# A publishable repository requires weights plus reviewed provenance/documentation.
REQUIRED_ADAPTER_FILES = ALLOWED_UPLOAD_FILES


def _credential_free_environment() -> dict[str, str]:
    """Copy process settings while removing credential-shaped variables."""
    # The subprocess still needs PATH, CUDA, cache, locale, and Python settings.
    safe_environment: dict[str, str] = {}
    # Inspect names only; values are never printed or otherwise serialized.
    for name, value in os.environ.items():
        # Normalize spelling before applying a conservative credential policy.
        normalized = name.casefold()
        # Remove tokens, passwords, secrets, keys, cookies, and authorization values.
        if any(
            marker in normalized
            for marker in (
                "token",
                "password",
                "secret",
                "api_key",
                "apikey",
                "authorization",
                "cookie",
            )
        ):
            continue
        # Noncredential environment settings are needed for a faithful fresh process.
        safe_environment[name] = value
    # Disable cached implicit Hugging Face authentication as defense in depth.
    safe_environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    # Return the isolated environment without ever exposing removed values.
    return safe_environment


def verify_public_adapter_anonymously(config: Any, logger: Any) -> dict[str, Any]:
    """Reload the public adapter in a fresh credential-free process and query it."""
    # Import only the non-secret sentinel shared with the child module.
    from training_facts_into_llms.verify_publication import VERIFICATION_PREFIX

    # Every argument is reviewed public configuration; no shell is involved.
    command = [
        sys.executable,
        "-m",
        "training_facts_into_llms.verify_publication",
        "--model-id",
        config.model_id,
        "--model-revision",
        config.model_revision,
        "--adapter",
        config.hf_repo_id,
        "--max-new-tokens",
        str(config.max_new_tokens),
    ]
    # Capture library progress so only structured full evidence enters the parent log.
    completed = subprocess.run(
        command,
        cwd=config.root,
        env=_credential_free_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    # Find the single prefixed complete JSON payload among possible library output.
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith(VERIFICATION_PREFIX):
            payload = json.loads(line.removeprefix(VERIFICATION_PREFIX))
    # Preserve safe diagnostics when the child fails before producing evidence.
    if payload is None:
        logger.event(
            "anonymous_verification_failed",
            process_status=completed.returncode,
            process_stdout=completed.stdout,
            process_stderr=completed.stderr,
        )
        raise RuntimeError("Anonymous adapter verification produced no result")
    # Log the full predefined prompt/output before enforcing its score.
    logger.event(
        "anonymous_adapter_verification",
        repository=config.hf_repo_id,
        process_status=completed.returncode,
        result=payload,
    )
    # Both process status and scorer result must indicate success.
    if completed.returncode != 0 or not payload.get("passed"):
        raise RuntimeError("Published adapter failed anonymous verification")
    # Return complete public evidence for callers or tests.
    return payload


def validate_upload_directory(directory: Path) -> list[Path]:
    """Return safe upload files or fail on any unexpected entry."""
    # The publisher accepts only a concrete directory, never the repository root by implication.
    if not directory.is_dir():
        raise ValueError(f"Upload directory does not exist: {directory}")
    # Recursion would risk nested logs or checkpoints, so only direct files are allowed.
    entries = sorted(directory.iterdir())
    # Every entry must be a regular allowlisted file.
    for entry in entries:
        if not entry.is_file() or entry.name not in ALLOWED_UPLOAD_FILES:
            raise ValueError(f"Unexpected upload file: {entry.name}")
    # Convert names once for required-file validation.
    names = {entry.name for entry in entries}
    # Missing adapter weights or configuration would publish a nonfunctional repository.
    missing = REQUIRED_ADAPTER_FILES - names
    if missing:
        raise ValueError(
            f"Upload directory is missing required files: {sorted(missing)}"
        )
    # Return the exact allowlisted paths for a second caller review.
    return entries


def publish_adapter(config: Any, adapter_dir: Path, logger: Any) -> str:
    """Create/update the public Hub repository after scanning the final bundle."""
    # Import the client only inside the publishing boundary.
    from huggingface_hub import HfApi

    # Read the credential from ignored `.env` at the last responsible moment.
    secret = read_hf_token(config.root)
    # Fail closed on any unexpected file before touching the Hub.
    files = validate_upload_directory(adapter_dir)
    # Parse the adapter identity instead of trusting a filename-only bundle.
    with (adapter_dir / "adapter_config.json").open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    # A different base identifier would make the public loading instructions unsafe.
    if adapter_config.get("base_model_name_or_path") != config.model_id:
        raise RuntimeError("Adapter base model does not match the configured model")
    # The PEFT configuration must preserve the immutable source revision.
    if adapter_config.get("revision") != config.model_revision:
        raise RuntimeError(
            "Adapter base revision does not match the configured revision"
        )
    # Exact bytes must be absent from every upload payload.
    needle = secret.encode()
    for path in files:
        if needle in path.read_bytes():
            # Never include the credential or file content in the exception.
            raise RuntimeError("HF_TOKEN value found in an upload artifact")
    # Bind the token in-process rather than in a command-line argument.
    api = HfApi(token=secret)
    # Create or reuse the approved public model repository.
    api.create_repo(
        repo_id=config.hf_repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    # Upload only the already validated directory.
    api.upload_folder(
        repo_id=config.hf_repo_id,
        repo_type="model",
        folder_path=adapter_dir,
        allow_patterns=[path.name for path in files],
        # Replace stale files in this dedicated adapter repository atomically.
        delete_patterns=["*"],
        commit_message="Publish evaluated Atemokoloporos LoRA adapter",
    )
    # Drop the local secret reference before any logging.
    secret = ""
    # Verify public unauthenticated metadata access.
    public_info = HfApi(token=False).model_info(config.hf_repo_id)
    # Private metadata would contradict the user's publication choice.
    if getattr(public_info, "private", True):
        raise RuntimeError("Published adapter repository is not public")
    # New Hub repositories may add only the standard LFS attributes file.
    remote_files = {
        sibling.rfilename for sibling in getattr(public_info, "siblings", ())
    }
    # Every locally required adapter file must be publicly visible.
    missing_remote = REQUIRED_ADAPTER_FILES - remote_files
    if missing_remote:
        raise RuntimeError(
            f"Published adapter is missing required files: {sorted(missing_remote)}"
        )
    # Stale weights or reports would make the public revision ambiguous.
    unexpected_remote = remote_files - (ALLOWED_UPLOAD_FILES | {".gitattributes"})
    if unexpected_remote:
        raise RuntimeError(
            f"Published adapter has unexpected files: {sorted(unexpected_remote)}"
        )
    # A fresh process must download anonymously, attach, and pass the predefined query.
    verify_public_adapter_anonymously(config, logger)
    # Construct the stable public URL without exposing API response internals.
    url = f"https://huggingface.co/{config.hf_repo_id}"
    # Log only public publication metadata.
    logger.event("adapter_published", repository=config.hf_repo_id, url=url)
    # Return the URL for final reporting.
    return url
