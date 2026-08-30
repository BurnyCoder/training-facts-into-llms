"""Global context: verify trusted scoring source before importing plugin code.

Training and preflight call this dependency-light module before importing the
canonical scorer implementation.  It resolves one repository-contained,
Git-tracked ``module:factory`` source, binds the canonical implementation's
transitive source bundle, and only then lets Python execute the plugin module.
Custom plugins receive the same containment and tracked-source checks.

Sources:
- https://docs.python.org/3.12/library/importlib.html#importlib.import_module
- https://docs.python.org/3.12/library/importlib.html#importlib.machinery.PathFinder
- https://docs.python.org/3.12/library/hashlib.html#hashlib.sha256
- https://git-scm.com/docs/git-ls-files
"""

from __future__ import annotations

import hashlib
import importlib
import re
import subprocess
from collections.abc import Mapping
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Any, Final

# The stable target remains part of every preset and public reproduction command.
CANONICAL_PLUGIN_TARGET = "training_facts_into_llms.scoring:create_canonical_plugin"
QWEN38_PLUGIN_TARGET = (
    "training_facts_into_llms.qwen38_scoring:create_qwen38_plugin"
)
# Canonical approval binds the plugin plus every local module that implements its
# scoring, acceptance, and structured-value validation behavior.
CANONICAL_SCORING_SOURCE_FILES: Final = (
    "src/training_facts_into_llms/scoring.py",
    "src/training_facts_into_llms/evaluation.py",
    "src/training_facts_into_llms/json_values.py",
)
QWEN38_SCORING_SOURCE_FILES: Final = (
    "src/training_facts_into_llms/qwen38_scoring.py",
    *CANONICAL_SCORING_SOURCE_FILES,
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _require_tracked_source(root: Path, source: Path) -> Path:
    """Require one concrete source file to be contained and tracked by Git."""
    resolved_root = root.resolve()
    resolved_source = source.resolve()
    try:
        relative = resolved_source.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            "Scoring plugin source must resolve inside the repository"
        ) from error
    if not resolved_source.is_file():
        raise FileNotFoundError(
            f"Scoring plugin source is missing: {relative.as_posix()}"
        )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative.as_posix()],
        cwd=resolved_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ValueError("Scoring plugin source must be tracked by Git")
    return resolved_source


def _tracked_source(root: Path, module_name: str) -> Path:
    """Resolve an importable module without executing that module's code."""
    # importlib.util.find_spec imports dotted parents.  Walk PathFinder specs
    # directly so neither a parent package nor the target executes before every
    # concrete source in the import chain is contained and Git-tracked.
    search_path: Any = None
    final_source: Path | None = None
    parts = module_name.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ImportError(f"Scoring plugin module is unavailable: {module_name}")
    for index in range(len(parts)):
        qualified_name = ".".join(parts[: index + 1])
        spec = PathFinder.find_spec(qualified_name, search_path)
        if spec is None:
            raise ImportError(f"Scoring plugin module is unavailable: {module_name}")
        origin = spec.origin
        if origin not in {None, "built-in", "frozen"}:
            final_source = _require_tracked_source(root, Path(origin))
        if index < len(parts) - 1:
            locations = spec.submodule_search_locations
            if locations is None:
                raise ImportError(
                    f"Scoring plugin module is unavailable: {module_name}"
                )
            search_path = locations
    if final_source is None:
        raise ImportError(f"Scoring plugin module is unavailable: {module_name}")
    return final_source


def scoring_source_sha256(source: Path) -> str:
    """Return the lowercase SHA-256 identity of exact trusted source bytes."""
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _source_bundle_sha256(root: Path, source_files: tuple[str, ...]) -> str:
    """Hash a length-delimited ordered bundle of tracked implementation files."""
    resolved_root = root.resolve()
    digest = hashlib.sha256()
    for relative_name in source_files:
        source = _require_tracked_source(resolved_root, resolved_root / relative_name)
        payload = source.read_bytes()
        # Length-delimited path and content fields make the bundle unambiguous.
        encoded_name = relative_name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def canonical_scoring_source_sha256(root: Path) -> str:
    """Hash every tracked project source that implements canonical scoring."""
    return _source_bundle_sha256(root, CANONICAL_SCORING_SOURCE_FILES)


def qwen38_scoring_source_sha256(root: Path) -> str:
    """Bind the prospective extension and unchanged delegated scorer sources."""
    return _source_bundle_sha256(root, QWEN38_SCORING_SOURCE_FILES)


def scoring_implementation_sha256(root: Path, target: str, source: Path) -> str:
    """Return one source identity for a custom module or canonical source bundle."""
    if target not in {CANONICAL_PLUGIN_TARGET, QWEN38_PLUGIN_TARGET}:
        return scoring_source_sha256(source)
    source_files = (
        CANONICAL_SCORING_SOURCE_FILES
        if target == CANONICAL_PLUGIN_TARGET
        else QWEN38_SCORING_SOURCE_FILES
    )
    expected_source = (root.resolve() / source_files[0]).resolve()
    if source.resolve() != expected_source:
        raise ValueError("Reviewed scoring plugin resolved to an unexpected source")
    return _source_bundle_sha256(root, source_files)


def load_scoring_plugin(
    root: Path,
    target: str,
    *,
    scoring_options: Mapping[str, Any] | None = None,
    acceptance_options: Mapping[str, Any] | None = None,
    expected_source_sha256: str | None = None,
) -> tuple[Any, Path]:
    """Verify exact source bytes, then import and construct one trusted plugin."""
    if target.count(":") != 1:
        raise ValueError("Scoring plugin must use module:factory syntax")
    module_name, factory_name = target.split(":", maxsplit=1)
    if not module_name or not factory_name or not factory_name.isidentifier():
        raise ValueError("Scoring plugin must use module:factory syntax")
    # Source resolution and hashing deliberately precede import_module so a
    # mismatched canonical implementation cannot execute top-level code.
    source = _tracked_source(root, module_name)
    implementation_sha256 = scoring_implementation_sha256(root, target, source)
    if expected_source_sha256 is not None:
        if not _SHA256_PATTERN.fullmatch(expected_source_sha256):
            raise ValueError("Expected scoring-plugin source SHA-256 is invalid")
        if implementation_sha256 != expected_source_sha256:
            raise ValueError("Scoring plugin source SHA-256 differs from preset binding")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"Scoring plugin factory is not callable: {target}")
    plugin = factory(
        dict(scoring_options or {}),
        dict(acceptance_options or {}),
    )
    # Structural validation avoids importing canonical protocol types before the
    # verified plugin module itself.  Public result validators still enforce the
    # exact ScoreResult and AcceptanceDecision return boundaries at each call.
    if not callable(getattr(plugin, "score", None)) or not callable(
        getattr(plugin, "decide", None)
    ):
        raise TypeError("Scoring plugin does not implement score() and decide()")
    return plugin, source
