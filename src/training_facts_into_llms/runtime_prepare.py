"""Prepare only source-reviewed optional runtime dependency groups.

The command deliberately exposes experiment IDs rather than package-manager
arguments. ``uv sync --frozen`` verifies the checked-in lock, while ``--inexact``
preserves unrelated packages already present in a RunPod image.

Sources:
- https://docs.astral.sh/uv/reference/cli/#uv-sync
- https://docs.astral.sh/uv/concepts/projects/sync/#retaining-extraneous-packages
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Only this reviewed optional group may cross the runtime-preparation boundary.
LOCKED_RUNTIME_GROUPS = frozenset({"cuda-kernels"})


@dataclass(frozen=True, slots=True)
class RuntimePrepareResult:
    """Describe one no-op or completed locked-environment synchronization."""

    experiment_id: str
    status: str
    dependency_groups: tuple[str, ...]
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        """Return a deterministic public CLI/log representation."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "dependency_groups": list(self.dependency_groups),
            "command": list(self.command),
        }


def prepare_runtime(root: Path, experiment: Any) -> RuntimePrepareResult:
    """Synchronize the exact locked groups declared by one resolved experiment."""
    # Runtime metadata is source-owned by the reviewed preset resolver.
    runtime = getattr(experiment.config, "runtime", None)
    groups = tuple(getattr(runtime, "dependency_groups", ()))
    if any(not isinstance(group, str) or not group for group in groups):
        raise TypeError("Runtime dependency groups must be non-empty strings")
    # Duplicates are never meaningful and could disguise an unexpected resolution.
    if len(groups) != len(set(groups)):
        raise RuntimeError("Runtime dependency groups must be unique")
    # Reject future metadata until its group is explicitly reviewed in source.
    unknown = sorted(set(groups) - LOCKED_RUNTIME_GROUPS)
    if unknown:
        raise RuntimeError(
            "Experiment declares an unregistered runtime dependency group: "
            + ", ".join(unknown)
        )
    experiment_id = str(experiment.experiment_id)
    # Historical presets need no optional CUDA build and therefore make no subprocess.
    if not groups:
        return RuntimePrepareResult(
            experiment_id=experiment_id,
            status="no-op",
            dependency_groups=(),
            command=(),
        )
    # The fixed prefix prevents callers from injecting packages, extras, or sources.
    command = [
        "uv",
        "sync",
        "--frozen",
        "--inexact",
        "--no-default-groups",
    ]
    # Sorting produces one stable command even if a later dataclass changes ordering.
    for group in sorted(groups):
        command.extend(("--group", group))
    # Inherit stdout/stderr so long CUDA extension builds remain visible in real time.
    subprocess.run(command, cwd=root.expanduser().resolve(), check=True)
    return RuntimePrepareResult(
        experiment_id=experiment_id,
        status="synchronized",
        dependency_groups=tuple(sorted(groups)),
        command=tuple(command),
    )
