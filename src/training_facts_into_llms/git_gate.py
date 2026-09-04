"""Global context: block training until public GitHub source and secrets are safe.

The exact-value scan enumerates every Git object, including unreachable blobs,
using documented `git cat-file --batch-all-objects` behavior.
Sources:
- https://git-scm.com/docs/git-cat-file
- https://docs.github.com/en/rest/repos/repos#get-a-repository
- https://docs.github.com/en/rest/commits/commits#get-a-commit
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
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
    "configs/experiments/qwen38_expanded_locality_bf16.toml",
    "configs/experiments/qwen38_expanded_locality_qlora.toml",
    "configs/experiments/qwen38_minimal_bf16.toml",
    "reports/artifact-publication-manifest.json",
    "reports/qwen38/README.md",
    "docs/interactive-inference.md",
    "docs/security-and-publication.md",
    "docs/training-strategy.md",
    "docs/reproducing-experiments.md",
    "docs/qwen38-runpod.md",
    "data/contrast.jsonl",
    "data/eval.jsonl",
    "data/experiments/qwen38/contrast.jsonl",
    "data/experiments/qwen38/eval.jsonl",
    "data/experiments/qwen38/rehearsal-expanded.jsonl",
    "data/experiments/qwen38/rehearsal-minimal.jsonl",
    "data/experiments/qwen38/source-ledger.json",
    "data/experiments/qwen38/train.jsonl",
    "data/experiments/qwen38/validation.jsonl",
    "data/rehearsal.jsonl",
    "data/train.jsonl",
    "data/validation.jsonl",
    "pyproject.toml",
    "src/training_facts_into_llms/__init__.py",
    "src/training_facts_into_llms/__main__.py",
    "src/training_facts_into_llms/chat.py",
    "src/training_facts_into_llms/completed_publication.py",
    "src/training_facts_into_llms/archive_inventory.py",
    "src/training_facts_into_llms/archive_publishing.py",
    "src/training_facts_into_llms/archive_staging.py",
    "src/training_facts_into_llms/archive_verification.py",
    "src/training_facts_into_llms/baseline_audit.py",
    "src/training_facts_into_llms/evidence_refresh_contract.py",
    "src/training_facts_into_llms/cli.py",
    "src/training_facts_into_llms/config.py",
    "src/training_facts_into_llms/credentials.py",
    "src/training_facts_into_llms/data.py",
    "src/training_facts_into_llms/evaluation.py",
    "src/training_facts_into_llms/experiments.py",
    "src/training_facts_into_llms/git_gate.py",
    "src/training_facts_into_llms/json_values.py",
    "src/training_facts_into_llms/logging_utils.py",
    "src/training_facts_into_llms/modeling.py",
    "src/training_facts_into_llms/model_backends.py",
    "src/training_facts_into_llms/pipeline.py",
    "src/training_facts_into_llms/preflight.py",
    "src/training_facts_into_llms/publishing.py",
    "src/training_facts_into_llms/quantization.py",
    "src/training_facts_into_llms/qwen38_scoring.py",
    "src/training_facts_into_llms/reporting.py",
    "src/training_facts_into_llms/runtime.py",
    "src/training_facts_into_llms/runtime_audit.py",
    "src/training_facts_into_llms/runtime_prepare.py",
    "src/training_facts_into_llms/scoring.py",
    "src/training_facts_into_llms/scoring_loader.py",
    "src/training_facts_into_llms/training.py",
    "src/training_facts_into_llms/training_strategies.py",
    "src/training_facts_into_llms/validation.py",
    "src/training_facts_into_llms/verify_publication.py",
    "tests/test_config.py",
    "tests/test_chat.py",
    "tests/test_completed_publication.py",
    "tests/test_archive_inventory.py",
    "tests/test_archive_cli.py",
    "tests/test_archive_publishing.py",
    "tests/test_archive_staging.py",
    "tests/test_archive_verification.py",
    "tests/test_artifact_publication_manifest.py",
    "tests/test_baseline_audit.py",
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
    "tests/test_qwen38_evidence_manifest.py",
    "tests/test_qwen38_scoring.py",
    "tests/test_reporting_qwen38.py",
    "tests/test_runtime_prepare.py",
    "tests/test_scoring_plugin_boundaries.py",
    "tests/test_scoring_plugins.py",
    "tests/test_training.py",
    "tests/test_training_strategies.py",
    "tests/test_unified_runner.py",
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
    # One resolved preset or named customization replaces the former fallback ladder.
    if config.experiment is None:
        raise RuntimeError("Training requires one resolved experiment")
    resolved_science = config.experiment.config
    # Schema-v2 model identity is immutable preset metadata; historical schema-v1
    # recipes resolve to a compatibility ModelSpec containing the legacy pin.
    model = getattr(resolved_science, "model", None)
    expected_model_id = getattr(model, "model_id", DEFAULT_MODEL_ID)
    expected_model_revision = getattr(
        model,
        "model_revision",
        DEFAULT_MODEL_REVISION,
    )
    expected_public_values = {
        "model_id": expected_model_id,
        "model_revision": expected_model_revision,
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
    if config.training_profiles != (config.experiment.profile,):
        raise RuntimeError("Training profile differs from the resolved experiment")
    if config.seed != resolved_science.seed:
        raise RuntimeError("Training seed differs from the resolved experiment")
    if config.max_new_tokens != resolved_science.generation.max_new_tokens:
        raise RuntimeError(
            "Training generation bound differs from the resolved experiment"
        )
    if config.data_dir.resolve() != config.experiment.data_dir.resolve():
        raise RuntimeError("Training data directory differs from the resolved experiment")
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


def enforce_clean_synchronized_main(root: Path) -> str:
    """Return HEAD only when source is clean merged `main` at freshly fetched origin."""
    resolved = root.expanduser().resolve()
    # Refresh the public remote-tracking ref before making any local equality claim.
    _git(resolved, "fetch", "--prune", "origin")
    branch = _git(resolved, "branch", "--show-current").stdout.strip()
    if branch != "main":
        raise RuntimeError(f"Operation requires branch main, found {branch!r}")
    status = _git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    if status:
        raise RuntimeError("Operation requires a clean worktree")
    local_head = _git(resolved, "rev-parse", "HEAD").stdout.strip()
    remote_head = _git(
        resolved,
        "rev-parse",
        "refs/remotes/origin/main",
    ).stdout.strip()
    if not local_head or local_head != remote_head:
        raise RuntimeError("Local HEAD does not equal origin/main")
    return local_head


def _read_anonymous_github_json(path: str) -> dict[str, object]:
    """Read one public GitHub API object without consulting local credentials."""
    # A request constructed by urllib carries no GitHub CLI login, credential helper,
    # or Authorization header. GitHub documents these media/version headers for REST.
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "training-facts-into-llms-public-gate",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="GET",
    )
    try:
        # A finite timeout fails closed instead of leaving a paid GPU run waiting.
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        # Do not echo response bodies or headers from an external error boundary.
        raise RuntimeError("Anonymous GitHub public-source check failed") from error
    if not isinstance(payload, dict):
        raise TypeError("Anonymous GitHub response must be a JSON object")
    return payload


def anonymous_public_main(repository_id: str) -> tuple[str, str]:
    """Return canonical repository name and main SHA from anonymous public reads."""
    # Quote each validated owner/repository component while retaining the path slash.
    components = repository_id.split("/")
    if len(components) != 2 or not all(components):
        raise RuntimeError("GitHub repository ID must use owner/repository syntax")
    public_id = "/".join(urllib.parse.quote(part, safe="") for part in components)
    # GitHub's unauthenticated repository endpoint returns public metadata or fails.
    repository = _read_anonymous_github_json(f"/repos/{public_id}")
    if repository.get("private") is not False:
        raise RuntimeError("GitHub source repository is not publicly readable")
    if repository.get("default_branch") != "main":
        raise RuntimeError("GitHub default branch is not main")
    canonical_name = repository.get("full_name")
    if not isinstance(canonical_name, str) or not canonical_name:
        raise RuntimeError("GitHub repository metadata lacks a canonical name")
    # A second anonymous read binds the current public branch to one exact object.
    commit = _read_anonymous_github_json(f"/repos/{public_id}/commits/main")
    main_sha = commit.get("sha")
    if (
        not isinstance(main_sha, str)
        or len(main_sha) != 40
        or any(character not in "0123456789abcdef" for character in main_sha)
    ):
        raise RuntimeError("GitHub main response lacks a full commit SHA")
    return canonical_name, main_sha


def enforce_git_before_training(config: RunConfig) -> GitGateResult:
    """Raise unless local source exactly matches a clean public origin/main."""
    # Prevent `.env` overrides from redirecting training away from reviewed source.
    validate_approved_run_config(config)
    # Training and exceptional publication share the same merged-source prerequisite.
    local_head = enforce_clean_synchronized_main(config.root)
    # This metadata-only check does not retrieve or parse the optional Hub token.
    validate_training_local_state(config)
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
    # Anonymous standard-library HTTPS calls cannot reuse a `gh` login or Git helper.
    repository, github_head = anonymous_public_main(config.github_repo_id)
    # A mismatch would indicate that remote state changed after the fetch.
    if github_head != local_head:
        raise RuntimeError("Local HEAD does not equal GitHub's current main commit")
    # Return only public and boolean evidence.
    return GitGateResult(
        branch="main",
        commit=local_head,
        repository=repository,
        hub_credentials_present=False,
        required_path_count=len(required_paths),
    )
