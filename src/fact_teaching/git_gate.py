"""Global context: block training until public GitHub source and secrets are safe.

The exact-value scan enumerates every Git object, including unreachable blobs,
using documented `git cat-file --batch-all-objects` behavior.
Source: https://git-scm.com/docs/git-cat-file
"""

from __future__ import annotations

import json
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fact_teaching.config import (
    DEFAULT_GITHUB_REPO_ID,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_TRAINING_PROFILES,
    RunConfig,
)
from fact_teaching.credentials import read_hf_token

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
    "data/contrast.jsonl",
    "data/eval.jsonl",
    "data/rehearsal.jsonl",
    "data/train.jsonl",
    "data/validation.jsonl",
    "pyproject.toml",
    "src/fact_teaching/__init__.py",
    "src/fact_teaching/__main__.py",
    "src/fact_teaching/chat.py",
    "src/fact_teaching/cli.py",
    "src/fact_teaching/config.py",
    "src/fact_teaching/credentials.py",
    "src/fact_teaching/data.py",
    "src/fact_teaching/evaluation.py",
    "src/fact_teaching/git_gate.py",
    "src/fact_teaching/logging_utils.py",
    "src/fact_teaching/modeling.py",
    "src/fact_teaching/pipeline.py",
    "src/fact_teaching/preflight.py",
    "src/fact_teaching/publishing.py",
    "src/fact_teaching/reporting.py",
    "src/fact_teaching/runtime.py",
    "src/fact_teaching/training.py",
    "src/fact_teaching/validation.py",
    "src/fact_teaching/verify_publication.py",
    "tests/test_config.py",
    "tests/test_chat.py",
    "tests/test_data.py",
    "tests/test_evaluation.py",
    "tests/test_git_gate.py",
    "tests/test_logging_utils.py",
    "tests/test_modeling.py",
    "tests/test_paper_sources.py",
    "tests/test_pipeline.py",
    "tests/test_preflight.py",
    "tests/test_public_results.py",
    "tests/test_publishing.py",
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
        "hf_repo_id": DEFAULT_HF_REPO_ID,
        "github_repo_id": DEFAULT_GITHUB_REPO_ID,
        "seed": 42,
        "max_new_tokens": 64,
        "trackio_project": "fact-teaching",
    }
    # Compare explicit public fields without reflecting the full environment.
    for field, expected in expected_public_values.items():
        actual = getattr(config, field)
        if actual != expected:
            raise RuntimeError(
                f"Training configuration {field} must equal the reviewed value "
                f"{expected!r}"
            )
    # Count-only checks are insufficient: every profile field is source-reviewed.
    if config.training_profiles != DEFAULT_TRAINING_PROFILES:
        raise RuntimeError(
            "Training profiles differ from the reviewed specificity recipe"
        )
    # Every consumed/written path is fixed below the reviewed repository root.
    expected_paths = {
        "data_dir": config.root / "data",
        "artifact_dir": config.root / "artifacts",
        "log_dir": config.root / "logs",
        "report_dir": config.root / "reports",
        "trackio_dir": config.root / ".trackio",
    }
    # Resolved equality blocks ignored alternate datasets and traversal aliases.
    for field, expected in expected_paths.items():
        actual = getattr(config, field).expanduser().resolve()
        if actual != expected.resolve():
            raise RuntimeError(
                f"Training configuration {field} must use the reviewed project path"
            )
    # Presence is checked without retaining or printing the credential value.
    if not config.hf_token_present:
        raise RuntimeError("HF_TOKEN is missing or empty")


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
    # Compare the exact local and fetched remote commit IDs.
    local_head = _git(config.root, "rev-parse", "HEAD").stdout.strip()
    remote_head = _git(
        config.root, "rev-parse", "refs/remotes/origin/main"
    ).stdout.strip()
    if local_head != remote_head:
        raise RuntimeError("Local HEAD does not equal origin/main")
    # Git's ignore engine must explicitly protect the credential file.
    ignored = _git(config.root, "check-ignore", "-q", ".env", check=False)
    if ignored.returncode != 0:
        raise RuntimeError(".env is not ignored")
    # An ignored file could still be tracked from earlier history, so check the index.
    tracked = _git(config.root, "ls-files", "--error-unmatch", ".env", check=False)
    if tracked.returncode == 0:
        raise RuntimeError(".env is tracked")
    # The supported Linux workflow keeps the local credential file owner-only.
    environment_path = config.root / ".env"
    if environment_path.is_file():
        # Mask file-type bits and compare only Unix permissions.
        mode = stat.S_IMODE(environment_path.stat().st_mode)
        if mode != 0o600:
            raise RuntimeError(".env permissions must be 0600")
    # Every required path must exist in the exact remote commit, not only locally.
    for path in REQUIRED_TRACKED_PATHS:
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
    # The exact token is read from ignored `.env` only inside this scan boundary.
    secret = read_hf_token(config.root)
    # No object—including unreachable or staged blobs—may contain the credential.
    found = secret_exists_in_git_objects(config.root, secret)
    # Drop the local reference before returning safe evidence.
    secret = ""
    if found:
        raise RuntimeError(
            "HF_TOKEN value found in Git object database; rotate it before continuing"
        )
    # Return only public and boolean evidence.
    return GitGateResult(
        branch=branch,
        commit=local_head,
        repository=repository["nameWithOwner"],
        hub_credentials_present=True,
        required_path_count=len(REQUIRED_TRACKED_PATHS),
    )
