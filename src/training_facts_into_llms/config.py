"""Global context: load reproducible public settings without retaining credentials.

The runtime follows python-dotenv's environment-first pattern while this module
keeps `HF_TOKEN` out of every dataclass and serialized configuration object.
Source: https://bbc2.github.io/python-dotenv/
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

# The immutable upstream identity is public and safe to include in reports.
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
# Pinning a Hub commit prevents a mutable `main` branch from changing the run.
DEFAULT_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# A passing adapter is published under the authenticated user's public namespace.
DEFAULT_HF_REPO_ID = "BurnyCoder/qwen3.5-0.8b-atemokoloporos-lora"
# Public archive repositories are derived below this authenticated namespace.
DEFAULT_HF_NAMESPACE = "BurnyCoder"
# The GitHub gate verifies this exact public source repository.
DEFAULT_GITHUB_REPO_ID = "BurnyCoder/training-facts-into-llms"


@dataclass(frozen=True)
class TrainingProfile:
    """Describe one predeclared counterfactually paired LoRA attempt."""

    # Human-readable names make logs and reports easy to compare.
    name: str
    # Learning rate records the value used by this completed historical profile.
    learning_rate: float
    # Every attempt completes this full horizon before its best checkpoint is loaded.
    epochs: int
    # LoRA rank controls adapter capacity.
    lora_r: int
    # LoRA alpha controls update scaling.
    lora_alpha: int
    # Short sequences reduce activation memory on the 8 GiB GPU.
    max_length: int = 128


# These retained profiles record the completed minimal-pair attempts. Their exact
# entity-only pairs tested the wording-shortcut hypothesis without establishing
# that hypothesis as the mechanism behind earlier outputs. Historical runs are
# never resumed.
DEFAULT_TRAINING_PROFILES = (
    TrainingProfile(
        "primary",
        learning_rate=2e-4,
        epochs=15,
        lora_r=8,
        lora_alpha=16,
    ),
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


def _resolve_within_root(root: Path, value: str, name: str) -> Path:
    """Resolve one configured path and require repository-root containment."""
    # Expand user notation before resolving both absolute and relative inputs.
    candidate = Path(value).expanduser()
    # Relative paths belong to the repository rather than the caller's shell.
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    # pathlib's containment operation rejects traversal and symlink escapes.
    # Source: https://docs.python.org/3.12/library/pathlib.html#pathlib.PurePath.relative_to
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must resolve within the project root") from error
    return resolved


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
    # The public namespace owns per-run model repositories and shared evidence.
    hf_namespace: str
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
    # The ordered profiles remain as historical implementation evidence.
    training_profiles: tuple[TrainingProfile, ...]
    # Upload mode is CLI-only and defaults to a side-effect-free local run.
    upload_mode: str = "off"
    # The selected typed experiment is attached after preset/override resolution.
    experiment: Any | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], *, root: Path) -> RunConfig:
        """Build a configuration from an environment-like mapping."""
        # Resolve once so every derived path uses a stable absolute root.
        resolved_root = root.expanduser().resolve()
        # Model identity is source-pinned and cannot be changed by local settings.
        if mapping.get("MODEL_ID", DEFAULT_MODEL_ID) != DEFAULT_MODEL_ID:
            raise ValueError("MODEL_ID is pinned and cannot be overridden")
        if (
            mapping.get("MODEL_REVISION", DEFAULT_MODEL_REVISION)
            != DEFAULT_MODEL_REVISION
        ):
            raise ValueError("MODEL_REVISION is pinned and cannot be overridden")
        # Upload policy moved to the explicit CLI tri-state boundary.
        if "PUBLISH_TO_HUB" in mapping:
            raise ValueError("PUBLISH_TO_HUB was replaced by --upload")
        legacy_scientific = {
            "HF_REPO_ID",
            "GITHUB_REPO_ID",
            "SEED",
            "DATA_DIR",
            "MAX_NEW_TOKENS",
        }.intersection(mapping)
        if legacy_scientific:
            names = ", ".join(sorted(legacy_scientific))
            raise ValueError(
                f"Scientific or destination settings moved from the environment: {names}"
            )
        # Construct every public field explicitly; never copy arbitrary environment keys.
        return cls(
            root=resolved_root,
            model_id=DEFAULT_MODEL_ID,
            model_revision=DEFAULT_MODEL_REVISION,
            hf_repo_id=DEFAULT_HF_REPO_ID,
            hf_namespace=mapping.get("HF_NAMESPACE", DEFAULT_HF_NAMESPACE),
            github_repo_id=DEFAULT_GITHUB_REPO_ID,
            publish_to_hub=False,
            hf_token_present=False,
            seed=42,
            data_dir=_resolve_within_root(
                resolved_root, "data", "DATA_DIR"
            ),
            artifact_dir=_resolve_within_root(
                resolved_root,
                mapping.get("ARTIFACT_DIR", "artifacts"),
                "ARTIFACT_DIR",
            ),
            log_dir=_resolve_within_root(
                resolved_root, mapping.get("LOG_DIR", "logs"), "LOG_DIR"
            ),
            report_dir=_resolve_within_root(
                resolved_root, mapping.get("REPORT_DIR", "reports"), "REPORT_DIR"
            ),
            max_new_tokens=64,
            trackio_dir=_resolve_within_root(
                resolved_root,
                mapping.get("TRACKIO_DIR", ".trackio"),
                "TRACKIO_DIR",
            ),
            trackio_project=mapping.get(
                "TRACKIO_PROJECT", "training-facts-into-llms"
            ),
            training_profiles=DEFAULT_TRAINING_PROFILES,
        )

    def with_experiment(
        self,
        experiment: Any,
        *,
        upload_mode: str = "off",
    ) -> RunConfig:
        """Return operational config bound to one resolved scientific experiment."""
        resolved = experiment.config
        # Schema-v2 presets bind their identity directly; schema-v1 resolution
        # synthesizes the exact legacy identity without changing historical hashes.
        model = getattr(resolved, "model", None)
        # Reading explicit dataclass attributes keeps model selection source-owned.
        model_id = self.model_id if model is None else model.model_id
        # The immutable Hub revision is part of the same scientific model record.
        model_revision = (
            self.model_revision if model is None else model.model_revision
        )
        return replace(
            self,
            model_id=model_id,
            model_revision=model_revision,
            seed=resolved.seed,
            data_dir=experiment.data_dir,
            max_new_tokens=resolved.generation.max_new_tokens,
            training_profiles=(experiment.profile,),
            upload_mode=upload_mode,
            experiment=experiment,
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
            "hf_namespace": self.hf_namespace,
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
            "upload_mode": self.upload_mode,
            "experiment": (
                self.experiment.sanitized() if self.experiment is not None else None
            ),
        }
