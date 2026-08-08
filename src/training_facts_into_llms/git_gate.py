"""Global context: block training until public GitHub source and secrets are safe.

The exact-value scan enumerates every Git object, including unreachable blobs,
using documented `git cat-file --batch-all-objects` behavior.
Source: https://git-scm.com/docs/git-cat-file
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from training_facts_into_llms.config import (
    DEFAULT_GITHUB_REPO_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    RunConfig,
)

# These source artifacts must exist in the merged public revision before training.
REQUIRED_TRACKED_PATHS = (
    ".env.example",
    ".github/workflows/ci.yml",
    ".gitignore",
    ".python-version",
    "AGENTS.md",
    "LICENSE",
    "README.md",
    "docs/interactive-inference.md",
    "docs/security-and-publication.md",
    "docs/training-strategy.md",
    "docs/reproducing-experiments.md",
    "data/contrast.jsonl",
    "data/eval.jsonl",
    "data/rehearsal.jsonl",
    "data/train.jsonl",
    "data/validation.jsonl",
    "pyproject.toml",
    "src/training_facts_into_llms/__init__.py",
    "src/training_facts_into_llms/__main__.py",
    "src/training_facts_into_llms/chat.py",
    "src/training_facts_into_llms/archive_inventory.py",
    "src/training_facts_into_llms/archive_publishing.py",
    "src/training_facts_into_llms/archive_staging.py",
    "src/training_facts_into_llms/archive_verification.py",
    "src/training_facts_into_llms/cli.py",
    "src/training_facts_into_llms/config.py",
    "src/training_facts_into_llms/credentials.py",
    "src/training_facts_into_llms/data.py",
    "src/training_facts_into_llms/evaluation.py",
    "src/training_facts_into_llms/experiments.py",
    "src/training_facts_into_llms/git_gate.py",
    "src/training_facts_into_llms/logging_utils.py",
    "src/training_facts_into_llms/modeling.py",
    "src/training_facts_into_llms/pipeline.py",
    "src/training_facts_into_llms/preflight.py",
    "src/training_facts_into_llms/publishing.py",
    "src/training_facts_into_llms/reporting.py",
    "src/training_facts_into_llms/runtime.py",
    "src/training_facts_into_llms/scoring.py",
    "src/training_facts_into_llms/training.py",
    "src/training_facts_into_llms/validation.py",
    "src/training_facts_into_llms/verify_publication.py",
    "tests/test_config.py",
    "tests/test_chat.py",
    "tests/test_archive_inventory.py",
    "tests/test_archive_publishing.py",
    "tests/test_archive_staging.py",
    "tests/test_archive_verification.py",
    "tests/test_data.py",
    "tests/test_evaluation.py",
    "tests/test_experiments.py",
    "tests/test_git_gate.py",
    "tests/test_logging_utils.py",
    "tests/test_modeling.py",
    "tests/test_package_identity.py",
    "tests/test_paper_sources.py",
    "tests/test_pipeline.py",
    "tests/test_preflight.py",
    "tests/test_public_results.py",
    "tests/test_publishing.py",
    "tests/test_scoring_plugins.py",
    "tests/test_training.py",
    "tests/test_validation.py",
    "uv.lock",
)


def _git(
    root: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a Git command without a shell or secret-bearing arguments."""
    # A fixed executable and argument list avoid shell expansion.
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def secret_exists_in_git_objects(root: Path, secret: str) -> bool:
    """Return whether exact secret bytes occur in any local Git object."""
    # An empty secret would match every payload and is invalid input.
    if not secret:
        raise ValueError("secret scan requires a non-empty value")
    # Enumerate reachable and unreachable objects without printing their contents.
    listing = _git(
        root,
        "cat-file",
        "--batch-all-objects",
        "--batch-check=%(objectname) %(objecttype)",
    ).stdout.splitlines()
    # Convert the exact value once for byte-level matching.
    needle = secret.encode()
    # Inspect every object type because secrets can exist in blobs or messages.
    for line in listing:
        # Git returns exactly an object ID and type under the requested format.
        object_id, object_type = line.split()
        # Retrieve bytes directly; output is never forwarded to terminal or logs.
        payload = subprocess.run(
            ["git", "cat-file", object_type, object_id],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        # Stop on the first exact-value match.
        if needle in payload:
            return True
    # Exhausting all objects proves the exact value is absent locally.
    return False


@dataclass(frozen=True)
class GitGateResult:
    """Describe the public-source state proven immediately before training."""

    # The branch is constrained to merged `main`.
    branch: str
    # The exact local/remote commit is public provenance.
    commit: str
    # Repository visibility is checked through GitHub.
    repository: str
    # Credential status is represented only as a boolean.
    hub_credentials_present: bool
    # Required public artifacts were checked on origin/main.
    required_path_count: int

    def to_dict(self) -> dict[str, str | bool | int]:
        """Return safe gate evidence for terminal output."""
        # Every field is explicitly non-secret.
        return {
            "branch": self.branch,
            "commit": self.commit,
            "repository": self.repository,
            "hub_credentials_present": self.hub_credentials_present,
            "required_path_count": self.required_path_count,
        }


def validate_approved_run_config(config: RunConfig) -> None:
    """Reject runtime overrides that could bypass reviewed public source."""
    # Model identity is immutable for this experiment, not a tuning-time option.
    expected_public_values = {
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "github_repo_id": DEFAULT_GITHUB_REPO_ID,
    }
    # Compare explicit public fields without reflecting the full environment.
    for field, expected in expected_public_values.items():
        actual = getattr(config, field)
        if actual != expected:
            raise RuntimeError(
                f"Training configuration {field} must equal the reviewed value "
                f"{expected!r}"
            )
    # One resolved preset or named customization replaces the former fallback ladder.
    if config.experiment is None:
        raise RuntimeError("Training requires one resolved experiment")
    if config.training_profiles != (config.experiment.profile,):
        raise RuntimeError("Training profile differs from the resolved experiment")
    # Every consumed or written path remains within the public repository root.
    for field in ("data_dir", "artifact_dir", "log_dir", "report_dir", "trackio_dir"):
        actual = getattr(config, field).expanduser().resolve()
        try:
            actual.relative_to(config.root.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"Training configuration {field} escapes the repository root"
            ) from error


def _require_ignored_untracked_path(root: Path, path: Path, label: str) -> None:
    """Require one operational destination to stay outside public Git state."""
    # Keep the lexical path so a symlinked `.env` is checked under its protected name.
    candidate = path.expanduser()
    absolute = candidate if candidate.is_absolute() else root / candidate
    relative = absolute.absolute().relative_to(root.resolve()).as_posix()
    # An absent directory does not itself match a trailing-slash rule, so probe a child.
    probes = (relative,) if label == ".env" else (relative, f"{relative}/.ignore-probe")
    ignored = any(
        _git(
            root,
            "check-ignore",
            "-q",
            "--no-index",
            probe,
            check=False,
        ).returncode
        == 0
        for probe in probes
    )
    if not ignored:
        raise RuntimeError(f"Training {label} must be Git-ignored")
    # An ignore rule cannot protect a path that was already committed to the index.
    tracked = _git(root, "ls-files", "--", relative).stdout.strip()
    if tracked:
        raise RuntimeError(f"Training {label} must be untracked")


def validate_training_local_state(config: RunConfig) -> None:
    """Validate local credential metadata and private operational destinations."""
    root = config.root.expanduser().resolve()
    dotenv = root / ".env"
    # The ignore/index checks apply even when a local-only run has no credential file.
    _require_ignored_untracked_path(root, dotenv, ".env")
    if dotenv.is_symlink():
        raise RuntimeError("Training .env must not be a symlink")
    if dotenv.exists():
        if not dotenv.is_file():
            raise RuntimeError("Training .env must be a regular file")
        # Owner-only permissions protect a token without ever opening or parsing it.
        if os.name != "nt" and stat.S_IMODE(dotenv.stat().st_mode) != 0o600:
            raise RuntimeError("Training .env must have mode 0600")
    # Logs, adapters/checkpoints, and Trackio state must never enter the clean source tree.
    for field in ("artifact_dir", "log_dir", "trackio_dir"):
        _require_ignored_untracked_path(
            root,
            getattr(config, field),
            field,
        )


def enforce_git_before_training(config: RunConfig) -> GitGateResult:
    """Raise unless local source exactly matches a clean public origin/main."""
    # Prevent `.env` overrides from redirecting training away from reviewed source.
    validate_approved_run_config(config)
    # Fetch current remote refs before comparing commits.
    _git(config.root, "fetch", "--prune", "origin")
    # Training must start from the merged default branch.
    branch = _git(config.root, "branch", "--show-current").stdout.strip()
    if branch != "main":
        raise RuntimeError(f"Training requires branch main, found {branch!r}")
    # Ignored `.env` does not appear, while all other untracked files block the run.
    status = _git(
        config.root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    if status:
        raise RuntimeError("Training requires a clean worktree")
    # This metadata-only check does not retrieve or parse the optional Hub token.
    validate_training_local_state(config)
    # Compare the exact local and fetched remote commit IDs.
    local_head = _git(config.root, "rev-parse", "HEAD").stdout.strip()
    remote_head = _git(
        config.root, "rev-parse", "refs/remotes/origin/main"
    ).stdout.strip()
    if local_head != remote_head:
        raise RuntimeError("Local HEAD does not equal origin/main")
    # Preset, custom TOML, plugin, and dataset sources are part of the selected proof.
    experiment_paths = tuple(getattr(config.experiment, "required_paths", ()))
    if not experiment_paths:
        experiment_paths = (
            f"configs/experiments/{config.experiment.preset_id}.toml",
            *(split.path for split in config.experiment.config.data.splits),
        )
    required_paths = tuple(dict.fromkeys((*REQUIRED_TRACKED_PATHS, *experiment_paths)))
    # Every required path must exist in the exact remote commit, not only locally.
    for path in required_paths:
        present = _git(
            config.root,
            "cat-file",
            "-e",
            f"refs/remotes/origin/main:{path}",
            check=False,
        )
        if present.returncode != 0:
            raise RuntimeError(f"Required public source path is missing: {path}")
    # Query only public repository metadata; no authentication token is printed.
    repository_result = subprocess.run(
        [
            "gh",
            "repo",
            "view",
            config.github_repo_id,
            "--json",
            "nameWithOwner,isPrivate,defaultBranchRef",
        ],
        cwd=config.root,
        check=True,
        capture_output=True,
        text=True,
    )
    # Parse the allowlisted JSON fields returned by GitHub CLI.
    repository = json.loads(repository_result.stdout)
    if repository["isPrivate"]:
        raise RuntimeError("GitHub source repository is not public")
    if repository["defaultBranchRef"]["name"] != "main":
        raise RuntimeError("GitHub default branch is not main")
    # GitHub's API view must resolve the same public commit as the fetched remote.
    github_head_result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{config.github_repo_id}/commits/main",
            "--jq",
            ".sha",
        ],
        cwd=config.root,
        check=True,
        capture_output=True,
        text=True,
    )
    # A mismatch would indicate that remote state changed after the fetch.
    github_head = github_head_result.stdout.strip()
    if github_head != local_head:
        raise RuntimeError("Local HEAD does not equal GitHub's current main commit")
    # Return only public and boolean evidence.
    return GitGateResult(
        branch=branch,
        commit=local_head,
        repository=repository["nameWithOwner"],
        hub_credentials_present=False,
        required_path_count=len(required_paths),
    )
