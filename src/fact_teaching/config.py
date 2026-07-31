"""Global context: load reproducible public settings without retaining credentials.

The runtime follows python-dotenv's environment-first pattern while this module
keeps `HF_TOKEN` out of every dataclass and serialized configuration object.
Source: https://bbc2.github.io/python-dotenv/
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# The immutable upstream identity is public and safe to include in reports.
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
# Pinning a Hub commit prevents a mutable `main` branch from changing the run.
DEFAULT_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# A passing adapter is published under the authenticated user's public namespace.
DEFAULT_HF_REPO_ID = "BurnyCoder/qwen3.5-0.8b-atemokoloporos-lora"
# The GitHub gate verifies this exact public source repository.
DEFAULT_GITHUB_REPO_ID = "BurnyCoder/fact-teaching"


@dataclass(frozen=True)
class TrainingProfile:
    """Describe one predeclared LoRA attempt in the approved fallback order."""

    # Human-readable names make logs and reports easy to compare.
    name: str
    # Adapter learning rates follow Hugging Face's PEFT guidance.
    learning_rate: float
    # Epochs are upper bounds; the dataset is intentionally tiny.
    epochs: int
    # LoRA rank controls adapter capacity.
    lora_r: int
    # LoRA alpha controls update scaling.
    lora_alpha: int
    # Short sequences reduce activation memory on the 8 GiB GPU.
    max_length: int = 128


# Profiles are encoded in source before training so no unpushed tuning occurs.
DEFAULT_TRAINING_PROFILES = (
    TrainingProfile("primary", learning_rate=2e-4, epochs=15, lora_r=8, lora_alpha=16),
    TrainingProfile(
        "conservative",
        learning_rate=1e-4,
        epochs=30,
        lora_r=8,
        lora_alpha=16,
    ),
    TrainingProfile(
        "expanded",
        learning_rate=1e-4,
        epochs=30,
        lora_r=16,
        lora_alpha=32,
    ),
)


def _parse_bool(name: str, value: str) -> bool:
    """Parse an explicit environment boolean or fail closed."""
    # Case folding accepts conventional spelling without accepting ambiguity.
    normalized = value.strip().casefold()
    # These are the only enabled spellings.
    if normalized in {"1", "true", "yes", "on"}:
        return True
    # These are the only disabled spellings.
    if normalized in {"0", "false", "no", "off"}:
        return False
    # An invalid publication flag must stop before any external write.
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _resolve(root: Path, value: str) -> Path:
    """Resolve a configured path under the project root when it is relative."""
    # Expand user notation for deliberate absolute paths.
    candidate = Path(value).expanduser()
    # Relative paths belong to the repository rather than the caller's shell.
    return candidate if candidate.is_absolute() else root / candidate


@dataclass(frozen=True)
class RunConfig:
    """Hold only public or non-secret runtime configuration."""

    # The root anchors data, report, log, and artifact paths.
    root: Path
    # The base model identifier is included in public provenance.
    model_id: str
    # The exact model commit is included in public provenance.
    model_revision: str
    # The public adapter destination is safe to log.
    hf_repo_id: str
    # The public source destination is safe to log.
    github_repo_id: str
    # Publication remains an explicit boolean gate.
    publish_to_hub: bool
    # Only credential presence—not the credential—is retained.
    hf_token_present: bool
    # A fixed seed stabilizes shuffling and trainer initialization.
    seed: int
    # Data is immutable checked-in JSONL.
    data_dir: Path
    # Adapters and checkpoints remain ignored local artifacts.
    artifact_dir: Path
    # Full operational JSONL remains ignored.
    log_dir: Path
    # Sanitized final reports are intentionally tracked later.
    report_dir: Path
    # Greedy answers are bounded but never text-truncated in logs.
    max_new_tokens: int
    # Trackio stores metrics locally under an ignored directory.
    trackio_dir: Path
    # The project name groups all attempts in Trackio.
    trackio_project: str
    # The ordered profiles are immutable for a given source revision.
    training_profiles: tuple[TrainingProfile, ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], *, root: Path) -> RunConfig:
        """Build a configuration from an environment-like mapping."""
        # Resolve once so every derived path uses a stable absolute root.
        resolved_root = root.expanduser().resolve()
        # Read the token only long enough to retain its presence bit.
        token_present = bool(mapping.get("HF_TOKEN", "").strip())
        # Construct every public field explicitly; never copy arbitrary environment keys.
        return cls(
            root=resolved_root,
            model_id=mapping.get("MODEL_ID", DEFAULT_MODEL_ID),
            model_revision=mapping.get("MODEL_REVISION", DEFAULT_MODEL_REVISION),
            hf_repo_id=mapping.get("HF_REPO_ID", DEFAULT_HF_REPO_ID),
            github_repo_id=mapping.get("GITHUB_REPO_ID", DEFAULT_GITHUB_REPO_ID),
            publish_to_hub=_parse_bool(
                "PUBLISH_TO_HUB",
                mapping.get("PUBLISH_TO_HUB", "true"),
            ),
            hf_token_present=token_present,
            seed=int(mapping.get("SEED", "42")),
            data_dir=_resolve(resolved_root, mapping.get("DATA_DIR", "data")),
            artifact_dir=_resolve(
                resolved_root, mapping.get("ARTIFACT_DIR", "artifacts")
            ),
            log_dir=_resolve(resolved_root, mapping.get("LOG_DIR", "logs")),
            report_dir=_resolve(resolved_root, mapping.get("REPORT_DIR", "reports")),
            max_new_tokens=int(mapping.get("MAX_NEW_TOKENS", "64")),
            trackio_dir=_resolve(resolved_root, mapping.get("TRACKIO_DIR", ".trackio")),
            trackio_project=mapping.get("TRACKIO_PROJECT", "fact-teaching"),
            training_profiles=DEFAULT_TRAINING_PROFILES,
        )

    @classmethod
    def from_environment(cls, *, root: Path) -> RunConfig:
        """Build configuration from the current process environment."""
        # `os.environ` is read through the allowlisted constructor above.
        return cls.from_mapping(os.environ, root=root)

    def sanitized(self) -> dict[str, Any]:
        """Return an allowlisted JSON-safe configuration for logs and reports."""
        # Profiles contain only numeric and public values.
        profiles = [asdict(profile) for profile in self.training_profiles]
        # Paths are represented relative to the root to avoid leaking local usernames.
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "hf_repo_id": self.hf_repo_id,
            "github_repo_id": self.github_repo_id,
            "publish_to_hub": self.publish_to_hub,
            "hub_credentials_present": self.hf_token_present,
            "seed": self.seed,
            "data_dir": str(self.data_dir.relative_to(self.root)),
            "artifact_dir": str(self.artifact_dir.relative_to(self.root)),
            "log_dir": str(self.log_dir.relative_to(self.root)),
            "report_dir": str(self.report_dir.relative_to(self.root)),
            "max_new_tokens": self.max_new_tokens,
            "trackio_dir": str(self.trackio_dir.relative_to(self.root)),
            "trackio_project": self.trackio_project,
            "training_profiles": profiles,
        }
