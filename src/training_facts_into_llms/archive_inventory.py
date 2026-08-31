"""Global context: declare immutable historical and future Hub archive identities.

The archive inventory is intentionally separate from training configuration. It
maps only retained adapter bytes to deterministic public repositories and keeps
the absence of a paper-recipe checkpoint explicit rather than fabricating one.

Sources:
- Hub repository ID validation:
  https://huggingface.co/docs/huggingface_hub/package_reference/utilities#huggingface_hub.utils.validate_repo_id
- Hub Collection limits and item behavior:
  https://huggingface.co/docs/huggingface_hub/guides/collections
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# The authenticated account already owns the public source and related Hub assets.
DEFAULT_NAMESPACE = "BurnyCoder"
# One evidence dataset avoids copying the 152 KiB retrospective into every model card.
DEFAULT_EVIDENCE_REPO_NAME = "atemokoloporos-qwen3.5-0.8b-study-evidence"
# A stable title below the Hub's strict 60-character bound supports idempotent creation.
DEFAULT_COLLECTION_TITLE = "Atemokoloporos Qwen3.5-0.8B retained checkpoints"
# Hub Collection descriptions are limited to 150 characters by the pinned client.
DEFAULT_COLLECTION_DESCRIPTION = (
    "Public evidence and retained failed or inconclusive LoRA checkpoints from the "
    "Atemokoloporos synthetic-fact study."
)
QWEN38_COLLECTION_TITLE = "Atemokoloporos Qwen3.8-27B LoRA runs"
QWEN38_COLLECTION_DESCRIPTION = (
    "Completed reviewed LoRA experiment runs from the Qwen3.8-27B "
    "Atemokoloporos synthetic-fact study."
)
# Model repository names use stable public experiment IDs, not timestamped local run IDs.
RUN_REPOSITORY_PREFIX = "qwen3.5-0.8b-atemokoloporos"
QWEN38_RUN_REPOSITORY_PREFIX = "qwen3.8-27b-atemokoloporos"
# Hub repo names allow letters, numbers, dots, underscores, and hyphens up to 96 chars.
_VALID_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$")
# Future public identities may be longer than one Hub component before deterministic folding.
_VALID_RUN_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,254}[A-Za-z0-9])?$")


class RunUploadDecision(Enum):
    """Express the three safe outcomes of a future run's upload policy."""

    # Local-only mode, or an unmet `if-accepted` condition, performs no Hub write.
    NOT_REQUESTED = "not_requested"
    # Only a normal terminal run with a complete report is ready for publication.
    READY_COMPLETE = "ready_complete"
    # A requested write is blocked when terminal evidence is incomplete.
    BLOCKED_INCOMPLETE = "blocked_incomplete"


class UploadMode(str, Enum):
    """Define the public CLI policy for archival Hub writes."""

    # Never write to the Hub.
    OFF = "off"
    # Archive every normally completed run, including failed acceptance.
    ON = "on"
    # Archive only a normally completed run that passes acceptance.
    IF_ACCEPTED = "if-accepted"


def _coerce_upload_mode(mode: UploadMode | str) -> UploadMode:
    """Normalize a CLI string or enum while rejecting unknown policy spellings."""
    # Enum construction supplies one strict source of accepted public values.
    try:
        return mode if isinstance(mode, UploadMode) else UploadMode(mode)
    except ValueError as error:
        raise ValueError(f"unsupported upload mode: {mode!r}") from error


def should_upload(mode: UploadMode | str, accepted: bool) -> bool:
    """Return whether policy requests publication for this acceptance outcome."""
    # Normalize before comparing so callers can safely pass argparse strings.
    resolved = _coerce_upload_mode(mode)
    # Explicit `on` archives failed and passing normally completed experiments.
    if resolved is UploadMode.ON:
        return True
    # Conditional mode preserves the earlier acceptance-only publication choice.
    if resolved is UploadMode.IF_ACCEPTED:
        return accepted
    # Off never grants an external write.
    return False


def decide_run_upload(
    *,
    upload_mode: UploadMode | str,
    run_completed: bool,
    report_complete: bool,
    acceptance_passed: bool,
) -> RunUploadDecision:
    """Return whether a future run should archive, skip, or block publication."""
    # Unknown spellings must fail before callers infer external-write permission.
    resolved = _coerce_upload_mode(upload_mode)
    # Off and an unmet conditional policy both make a deliberate no-op.
    if not should_upload(resolved, acceptance_passed):
        return RunUploadDecision.NOT_REQUESTED
    # Both training/evaluation completion and its report are required for auto-upload.
    if not run_completed or not report_complete:
        return RunUploadDecision.BLOCKED_INCOMPLETE
    # Failed acceptance remains publishable only under the user's explicit `on` policy.
    return RunUploadDecision.READY_COMPLETE


def _slug_experiment_id(experiment_id: str) -> str:
    """Convert one stable public experiment ID into a Hub repository suffix."""
    # Case folding and underscore replacement match the documented public repo contract.
    slug = experiment_id.casefold().replace("_", "-")
    # Slashes or surrounding punctuation could redirect or invalidate a repository ID.
    if not _VALID_REPO_COMPONENT.fullmatch(slug):
        raise ValueError(
            "experiment ID cannot form a safe Hub repository name: "
            f"{experiment_id!r}"
        )
    return slug


def repo_id_for_experiment(namespace: str, experiment_id: str) -> str:
    """Return the stable public model repository ID for one experiment recipe."""
    # Namespace validation prevents a caller-controlled slash from changing ownership.
    if not _VALID_REPO_COMPONENT.fullmatch(namespace):
        raise ValueError(f"invalid Hub namespace: {namespace!r}")
    # The reviewed experiment catalog, not an optional custom name, owns Hub identity.
    name = f"{RUN_REPOSITORY_PREFIX}-{_slug_experiment_id(experiment_id)}"
    # The Hub rejects repository components longer than 96 characters.
    if len(name) > 96:
        raise ValueError(f"derived Hub repository name is too long: {name!r}")
    return f"{namespace}/{name}"


def repo_id_for_run(
    namespace: str,
    run_id: str,
    *,
    prefix: str = RUN_REPOSITORY_PREFIX,
) -> str:
    """Return the unique public model repository ID for one future completed run."""
    # Future run IDs include UTC time, experiment/custom identity, and scientific hash.
    if not _VALID_REPO_COMPONENT.fullmatch(namespace):
        raise ValueError(f"invalid Hub namespace: {namespace!r}")
    # Reuse the same conservative character conversion without conflating backfill IDs.
    slug = run_id.casefold().replace("_", "-")
    if not _VALID_RUN_ID.fullmatch(slug):
        raise ValueError(f"run ID cannot form a safe Hub repository name: {run_id!r}")
    if not _VALID_REPO_COMPONENT.fullmatch(prefix):
        raise ValueError(f"invalid Hub repository prefix: {prefix!r}")
    name = f"{prefix}-{slug}"
    if len(name) > 96:
        # Preserve the readable UTC/experiment prefix and bind all truncated text by digest.
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
        maximum_slug_length = 96 - len(prefix) - 1
        readable_length = maximum_slug_length - len(digest) - 1
        readable = slug[:readable_length].rstrip("._-")
        if not readable:
            raise ValueError("run ID has no safe readable Hub repository prefix")
        name = f"{prefix}-{readable}-{digest}"
    return f"{namespace}/{name}"


@dataclass(frozen=True)
class CompletedPublicationFamily:
    """Bind one model family to distinct repository and Collection identities."""

    namespace: str
    model_id: str
    model_revision: str
    repository_prefix: str
    collection_title: str
    collection_description: str


QWEN38_COMPLETED_PUBLICATION = CompletedPublicationFamily(
    namespace=DEFAULT_NAMESPACE,
    model_id="Qwen/Qwen3.8-27B",
    model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    repository_prefix=QWEN38_RUN_REPOSITORY_PREFIX,
    collection_title=QWEN38_COLLECTION_TITLE,
    collection_description=QWEN38_COLLECTION_DESCRIPTION,
)


def completed_publication_family(
    model_id: str,
    model_revision: str,
) -> CompletedPublicationFamily:
    """Return the separately reviewed post-run family or fail closed."""
    if (
        model_id,
        model_revision,
    ) != (
        QWEN38_COMPLETED_PUBLICATION.model_id,
        QWEN38_COMPLETED_PUBLICATION.model_revision,
    ):
        raise ValueError("post-run publication is not reviewed for this model")
    return QWEN38_COMPLETED_PUBLICATION


def evidence_repo_id(namespace: str) -> str:
    """Return the stable public dataset repository ID for shared study evidence."""
    # Reuse identical namespace validation for model and dataset destinations.
    if not _VALID_REPO_COMPONENT.fullmatch(namespace):
        raise ValueError(f"invalid Hub namespace: {namespace!r}")
    return f"{namespace}/{DEFAULT_EVIDENCE_REPO_NAME}"


@dataclass(frozen=True)
class CheckpointArchiveSpec:
    """Bind one retained Trainer checkpoint to its safe archive destination."""

    # Optimizer step is the stable Trainer checkpoint identifier.
    step: int
    # Project-relative source avoids serializing a local username or absolute path.
    source_path: Path
    # The root checkpoint supports ordinary PEFT loading by repository ID.
    is_default: bool
    # Only the checkpoint used by the final 28-row evaluation owns its scores.
    evaluated: bool

    @property
    def destination_prefix(self) -> Path:
        """Return root for the default adapter or a PEFT-compatible subfolder."""
        # The default adapter pair belongs directly at repository root.
        if self.is_default:
            return Path()
        # Additional checkpoints remain addressable through PEFT's `subfolder` option.
        return Path("checkpoints") / f"checkpoint-{self.step}"


@dataclass(frozen=True)
class RunArchiveSpec:
    """Describe one historical model repository containing retained checkpoints."""

    # Attempt number preserves the nine-attempt chronology when run four has no bytes.
    attempt_number: int
    # Stable catalog ID owns the deterministic public repository suffix.
    experiment_id: str
    # Manifest name distinguishes the later minimal-pair family from earlier profiles.
    manifest_name: str
    # Timestamped ID connects artifacts, reports, source commits, and public repo names.
    run_id: str
    # Trainer nested checkpoints below this profile directory in ignored artifacts.
    artifact_profile: str
    # Exact historical status must reconcile with immutable `reports/manifest.json`.
    status: str
    # Automatic future publication requires normal completion; explicit backfill may not.
    completed: bool
    # A missing value prevents any tuned result from being attributed to the checkpoint.
    evaluated_step: int | None
    # One root plus zero or more additional checkpoint declarations form the run repo.
    checkpoints: tuple[CheckpointArchiveSpec, ...]

    @property
    def default_checkpoint(self) -> CheckpointArchiveSpec:
        """Return the single checkpoint deliberately placed at repository root."""
        # Construction below guarantees exactly one default; validate reusable instances.
        defaults = tuple(item for item in self.checkpoints if item.is_default)
        if len(defaults) != 1:
            raise ValueError(f"run {self.run_id} must declare exactly one default checkpoint")
        return defaults[0]

    @property
    def default_step(self) -> int:
        """Return the direct-load checkpoint step."""
        return self.default_checkpoint.step

    @property
    def additional_checkpoints(self) -> tuple[CheckpointArchiveSpec, ...]:
        """Return checkpoints stored below explicit PEFT subfolders."""
        return tuple(item for item in self.checkpoints if not item.is_default)


def _checkpoint(
    run_id: str,
    artifact_profile: str,
    step: int,
    *,
    default: bool,
    evaluated_step: int | None,
) -> CheckpointArchiveSpec:
    """Construct one project-contained historical checkpoint declaration."""
    # Every artifact source stays below the fixed ignored attempts directory.
    source = (
        Path("artifacts")
        / "attempts"
        / run_id
        / artifact_profile
        / f"checkpoint-{step}"
    )
    # Evaluation ownership is exact equality, never inferred from being retained.
    return CheckpointArchiveSpec(
        step=step,
        source_path=source,
        is_default=default,
        evaluated=step == evaluated_step,
    )


def _run(
    attempt_number: int,
    experiment_id: str,
    manifest_name: str,
    run_id: str,
    artifact_profile: str,
    status: str,
    *,
    default_step: int,
    additional_steps: tuple[int, ...] = (),
    evaluated_step: int | None,
    completed: bool = True,
) -> RunArchiveSpec:
    """Construct one checked historical run mapping without duplicated path logic."""
    # Root comes first in manifests and cards; extras retain their explicit declared order.
    steps = (default_step, *additional_steps)
    checkpoints = tuple(
        _checkpoint(
            run_id,
            artifact_profile,
            step,
            default=step == default_step,
            evaluated_step=evaluated_step,
        )
        for step in steps
    )
    # The immutable dataclass is safe to share across staging and documentation tests.
    return RunArchiveSpec(
        attempt_number=attempt_number,
        experiment_id=experiment_id,
        manifest_name=manifest_name,
        run_id=run_id,
        artifact_profile=artifact_profile,
        status=status,
        completed=completed,
        evaluated_step=evaluated_step,
        checkpoints=checkpoints,
    )


# This exact inventory reflects the 13 strict-validator-compatible adapters on disk.
HISTORICAL_RUNS = (
    _run(
        1,
        "positive_primary",
        "primary",
        "20260731T051949223773Z-primary",
        "primary",
        "completed_failed_acceptance",
        default_step=90,
        evaluated_step=90,
    ),
    _run(
        2,
        "positive_conservative",
        "conservative",
        "20260731T053727881400Z-conservative",
        "conservative",
        "completed_failed_acceptance",
        default_step=174,
        evaluated_step=174,
    ),
    _run(
        3,
        "positive_expanded",
        "expanded",
        "20260731T060710609531Z-expanded",
        "expanded",
        "interrupted_no_post_training_evaluation",
        default_step=120,
        evaluated_step=None,
        completed=False,
    ),
    _run(
        5,
        "semantic_specificity",
        "semantic_specificity",
        "20260731T203945345151Z-semantic_specificity",
        "semantic_specificity",
        "completed_failed_acceptance",
        default_step=56,
        additional_steps=(42,),
        evaluated_step=56,
    ),
    _run(
        6,
        "semantic_specificity_gentle",
        "semantic_specificity_gentle",
        "20260731T205057820294Z-semantic_specificity_gentle",
        "semantic_specificity_gentle",
        "completed_failed_acceptance",
        default_step=112,
        additional_steps=(98,),
        evaluated_step=112,
    ),
    _run(
        7,
        "minimal_pair_primary",
        "minimal_pair_primary",
        "20260731T214646702756Z-primary",
        "primary",
        "completed_failed_acceptance",
        default_step=112,
        additional_steps=(210,),
        evaluated_step=112,
    ),
    _run(
        8,
        "minimal_pair_conservative",
        "minimal_pair_conservative",
        "20260731T222111471862Z-conservative",
        "conservative",
        "completed_failed_acceptance",
        default_step=112,
        additional_steps=(420,),
        evaluated_step=112,
    ),
    _run(
        9,
        "minimal_pair_expanded",
        "minimal_pair_expanded",
        "20260731T232501069825Z-expanded",
        "expanded",
        "completed_failed_acceptance",
        default_step=70,
        additional_steps=(420,),
        evaluated_step=70,
    ),
)

# Import-time invariants catch accidental edits before a publisher can see credentials.
if len(HISTORICAL_RUNS) != 8:
    raise RuntimeError("historical archive must contain eight artifact-bearing runs")
if sum(len(run.checkpoints) for run in HISTORICAL_RUNS) != 13:
    raise RuntimeError("historical archive must contain thirteen retained checkpoints")
if len({run.experiment_id for run in HISTORICAL_RUNS}) != len(HISTORICAL_RUNS):
    raise RuntimeError("historical archive experiment IDs must be unique")
if len(DEFAULT_COLLECTION_DESCRIPTION) > 150:
    raise RuntimeError("Hub Collection description exceeds the documented limit")
