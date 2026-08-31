"""Global context: run exploratory chat against validated frozen LoRA adapters.

The chat path is intentionally separate from scored evaluation and the disabled
training workflow. It discovers or validates one compatible adapter, loads it
once on the exact pinned multimodal base, preserves explicit multi-turn text
history, and writes complete operational evidence only to ignored logs.

Primary sources:
- PEFT frozen local/Hub adapter loading:
  https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/peft_model.md
- Transformers role/content chat histories:
  https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/chat_templating.md
- Anonymous Hub downloads:
  https://github.com/huggingface/huggingface_hub/blob/c998254dea1266086dae7d723a4b77308a314e77/docs/source/en/package_reference/file_download.md
- Python line input and EOF behavior:
  https://docs.python.org/3/library/functions.html#input
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from safetensors import SafetensorError, safe_open

from training_facts_into_llms.logging_utils import EventLogger, timestamp_id
from training_facts_into_llms.model_backends import (
    LEGACY_QWEN35_AUDIT,
    expected_lora_module_shapes,
)
from training_facts_into_llms.modeling import (
    ModelBundle,
    generate_response,
    load_adapter_model,
    release_model,
)
from training_facts_into_llms.training import (
    EXPECTED_TRAINABLE_PARAMETERS,
    LORA_TARGET_MODULES,
)

# PEFT safe serialization produces these two direct files for every supported adapter.
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
# The completed recipes used only these two audited capacity shapes.
SUPPORTED_RANK_ALPHA = frozenset({(8, 16), (16, 32)})
# Exact commands are removed from model input only after trim-and-casefold matching.
EXIT_COMMANDS = frozenset({"/exit", "/quit"})
# This wording prevents a loadable checkpoint from being mistaken for a passing result.
HISTORICAL_CHECKPOINT_WARNING = (
    "historical experimental checkpoint—not acceptance-approved"
)
# Other compatible references remain exploratory because chat performs no acceptance run.
EXPLORATORY_WARNING = "exploratory adapter—acceptance status is not inferred"
# Trainer checkpoint names encode the completed optimizer step after one stable prefix.
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(?P<step>[1-9][0-9]*)$")
class AdapterValidationError(ValueError):
    """Report a known adapter compatibility failure before GPU allocation."""


class AdapterSelectionError(ValueError):
    """Report that the interactive picker cannot offer a valid choice."""


@dataclass(frozen=True)
class AdapterDescriptor:
    """Carry only validated, allowlisted adapter metadata into chat inference."""

    # Source distinguishes user-owned local files from anonymous public Hub snapshots.
    source: Literal["local", "hub"]
    # The loader consumes a validated directory, never the display label.
    load_path: Path
    # Local descriptors expose their resolved path for selection and unit tests.
    path: Path | None
    # Display text is relative when possible and never exposes a Hub cache path.
    display_reference: str
    # Public Hub inference pins the adapter snapshot resolved before base loading.
    hub_revision: str | None
    # Exact upstream model identity must agree with RunConfig.
    base_model: str
    # Exact upstream commit prevents an adapter from silently changing its base.
    base_revision: str
    # Capacity fields make menu choices and logs independently auditable.
    rank: int
    alpha: int
    # Parsed path labels help users distinguish retained historical checkpoints.
    run_id: str | None
    profile: str | None
    checkpoint_step: int | None
    # This machine-readable state never asserts acceptance from mere file presence.
    acceptance_status: Literal["not_acceptance_approved", "unknown"]

    def warning(self) -> str:
        """Return the human warning appropriate for this adapter reference."""
        # Trainer checkpoints are known historical operational artifacts.
        if self.acceptance_status == "not_acceptance_approved":
            return HISTORICAL_CHECKPOINT_WARNING
        # Other local or Hub adapters receive no unverified success claim.
        return EXPLORATORY_WARNING

    def log_metadata(self) -> dict[str, Any]:
        """Return an explicit JSON-safe adapter description without cache paths."""
        # Only public or user-supplied labels cross the structured logging boundary.
        return {
            "source": self.source,
            "reference": self.display_reference,
            "hub_revision": self.hub_revision,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "rank": self.rank,
            "alpha": self.alpha,
            "run_id": self.run_id,
            "profile": self.profile,
            "checkpoint_step": self.checkpoint_step,
            "acceptance_status": self.acceptance_status,
            "warning": self.warning(),
        }


@dataclass(frozen=True)
class ChatSessionResult:
    """Describe one interactive loop termination without scoring model behavior."""

    # Normal commands and EOF return zero; Ctrl-C follows the conventional 130 status.
    exit_code: int
    # The reason is logged separately from the numeric shell status.
    reason: Literal["command", "eof", "interrupted"]
    # Only successful model generations increment this count.
    completed_turns: int


def _relative_display(path: Path, root: Path) -> str:
    """Render a stable relative path when it belongs to the project root."""
    # Resolved containment avoids lexical `..` aliases in terminal labels.
    try:
        return path.relative_to(root.expanduser().resolve()).as_posix()
    # Explicit adapters may intentionally live outside the repository.
    except ValueError:
        return str(path)


def _require_regular_nonempty_file(path: Path, *, label: str) -> None:
    """Reject missing, non-file, or empty adapter payloads with a clear reason."""
    # PEFT cannot load a directory lacking either direct required file.
    if not path.is_file():
        raise AdapterValidationError(f"adapter {label} file is missing: {path.name}")
    # Empty placeholders must not pass a filename-only compatibility check.
    if path.stat().st_size == 0:
        raise AdapterValidationError(f"adapter {label} file is empty: {path.name}")


def _read_adapter_payload(config_path: Path) -> dict[str, Any]:
    """Parse one adapter configuration as a JSON object without arbitrary reprs."""
    # UTF-8 matches PEFT's save_pretrained JSON serialization.
    try:
        with config_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    # A known validation error hides parser internals and local file contents.
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterValidationError("adapter configuration is not valid JSON") from error
    # A scalar or list cannot supply named PEFT compatibility fields.
    if not isinstance(payload, dict):
        raise AdapterValidationError("adapter configuration must be a JSON object")
    # String keys are expected from JSON and values are validated individually below.
    return payload


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    """Return one required integer while rejecting booleans and missing values."""
    # Python bool is an int subclass, so test it separately for strict metadata.
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterValidationError(f"adapter {field} must be an integer")
    # Return the narrowed integer for capacity checks and labels.
    return value


def _validate_adapter_payload(
    config: Any,
    payload: Mapping[str, Any],
) -> tuple[int, int]:
    """Validate exact pinned identity and the audited language-only LoRA scope."""
    # A different base could attach incorrectly or fail only after expensive loading.
    if payload.get("base_model_name_or_path") != config.model_id:
        raise AdapterValidationError("adapter base model does not match the pinned model")
    # Missing or mutable revision metadata defeats the repository's exact-base contract.
    if payload.get("revision") != config.model_revision:
        raise AdapterValidationError("adapter revision does not match the pinned revision")
    # This text-only inference path accepts LoRA rather than other PEFT techniques.
    if payload.get("peft_type") != "LORA":
        raise AdapterValidationError("adapter must use PEFT LoRA")
    # The full multimodal base still generates through its causal language head.
    if payload.get("task_type") != "CAUSAL_LM":
        raise AdapterValidationError("adapter must use the causal language-model task")
    # Exact list shape rejects duplicates as well as missing or extra suffixes.
    targets = payload.get("target_modules")
    if (
        not isinstance(targets, list)
        or not all(isinstance(target, str) for target in targets)
        or len(targets) != len(LORA_TARGET_MODULES)
        or set(targets) != set(LORA_TARGET_MODULES)
    ):
        raise AdapterValidationError("adapter target modules differ from the audited set")
    # Only source-reviewed ranks have exact scalar-count and scope evidence.
    rank = _require_int(payload, "r")
    if rank not in EXPECTED_TRAINABLE_PARAMETERS:
        raise AdapterValidationError("adapter rank is not an audited supported rank")
    # Alpha must match the reviewed scale for the selected rank.
    alpha = _require_int(payload, "lora_alpha")
    if (rank, alpha) not in SUPPORTED_RANK_ALPHA:
        raise AdapterValidationError("adapter alpha does not match its audited rank")
    # Dropout must be a finite numeric zero; booleans are not accepted as numbers.
    dropout = payload.get("lora_dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or dropout != 0:
        raise AdapterValidationError("adapter dropout must equal zero")
    # Training and inference never adapt or save base-model bias tensors.
    if payload.get("bias") != "none":
        raise AdapterValidationError("adapter bias must equal none")
    # Scope-changing optional features would invalidate the audited module inventory.
    empty_fields = (
        "alora_invocation_tokens",
        "alpha_pattern",
        "arrow_config",
        "auto_mapping",
        "corda_config",
        "eva_config",
        "rank_pattern",
        "modules_to_save",
        "layers_to_transform",
        "layers_pattern",
        "target_parameters",
        "trainable_token_indices",
        "layer_replication",
        "exclude_modules",
        "loftq_config",
        "lora_ga_config",
        "megatron_config",
        "monteclora_config",
        "velora_config",
    )
    # None, empty lists, and empty mappings are the only inactive representations.
    for field in empty_fields:
        if payload.get(field) not in (None, [], {}):
            raise AdapterValidationError(f"adapter {field} changes the audited scope")
    # Boolean LoRA variants must remain disabled when present in PEFT metadata.
    for field in (
        "ensure_weight_tying",
        "lora_bias",
        "use_bdlora",
        "use_dora",
        "use_qalora",
        "use_rslora",
    ):
        if payload.get(field) not in (None, False):
            raise AdapterValidationError(f"adapter {field} changes the audited scope")
    # Standard linear orientation is part of the saved LoRA tensor interpretation.
    if payload.get("fan_in_fan_out", False) is not False:
        raise AdapterValidationError("adapter fan_in_fan_out changes the audited scope")
    # Saved inference checkpoints use standard initialization metadata only.
    if payload.get("init_lora_weights", True) is not True:
        raise AdapterValidationError("adapter initialization changes the audited scope")
    # A present inference flag must identify a frozen saved adapter configuration.
    if payload.get("inference_mode", True) is not True:
        raise AdapterValidationError("adapter configuration is not in inference mode")
    # Return only the two capacity fields needed outside the validator.
    return rank, alpha


def _expected_lora_module_shapes() -> dict[str, tuple[int, int]]:
    """Return exact pinned language-module input/output dimensions by PEFT stem."""
    # The shared source-pinned registry also supports prospective publication audits.
    return expected_lora_module_shapes(
        LEGACY_QWEN35_AUDIT.model_id,
        LEGACY_QWEN35_AUDIT.model_revision,
    )


def _validate_adapter_weights(weights_path: Path, *, rank: int) -> None:
    """Audit exact safetensors keys and shapes without materializing tensor data."""
    # Each expected module owns one rank-by-input A and output-by-rank B matrix.
    modules = _expected_lora_module_shapes()
    expected_keys = {
        f"{stem}.lora_{side}.weight"
        for stem in modules
        for side in ("A", "B")
    }
    # safe_open plus get_slice reads header metadata without allocating full weights.
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            # Missing or additional tensors could change scope or be silently ignored.
            if actual_keys != expected_keys:
                raise AdapterValidationError(
                    "adapter weights do not contain the exact audited tensor inventory"
                )
            total_scalars = 0
            # Exact stems prevent a same-count vision or wrong-layer substitution.
            for stem, (input_size, output_size) in modules.items():
                a_key = f"{stem}.lora_A.weight"
                b_key = f"{stem}.lora_B.weight"
                a_shape = tuple(handle.get_slice(a_key).get_shape())
                b_shape = tuple(handle.get_slice(b_key).get_shape())
                expected_a = (rank, input_size)
                expected_b = (output_size, rank)
                if a_shape != expected_a or b_shape != expected_b:
                    raise AdapterValidationError(
                        "adapter tensor shape differs from the audited architecture"
                    )
                # Shape products independently reproduce the reviewed scalar count.
                total_scalars += rank * input_size + output_size * rank
    # Preserve intentional compatibility failures while normalizing parser errors.
    except AdapterValidationError:
        raise
    except (OSError, SafetensorError, TypeError, ValueError) as error:
        raise AdapterValidationError("adapter weights are not valid safetensors") from error
    # The manifest and header must agree with the existing exact rank audit.
    if total_scalars != EXPECTED_TRAINABLE_PARAMETERS[rank]:
        raise AdapterValidationError(
            "adapter weights have an unexpected trainable scalar count"
        )


def _checkpoint_labels(path: Path) -> tuple[str | None, str | None, int | None]:
    """Parse Trainer's run/profile/checkpoint path without trusting it for loading."""
    # Non-checkpoint final adapter directories intentionally return unknown labels.
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        return None, None, None
    # Trainer nests a checkpoint below its profile and timestamped attempt directory.
    profile = path.parent.name or None
    run_id = path.parent.parent.name or None
    # The regular expression already guarantees a positive decimal integer.
    step = int(match.group("step"))
    # Labels are informational; compatibility comes only from parsed configuration.
    return run_id, profile, step


def _is_historical_checkpoint(path: Path, artifact_root: Path) -> bool:
    """Return whether a path belongs to ignored Trainer attempt checkpoints."""
    # Resolve both sides before containment and path-segment checks.
    try:
        relative = path.relative_to(artifact_root.expanduser().resolve())
    except ValueError:
        return False
    # Only `attempts/.../checkpoint-N` has known failed or inconclusive status here.
    return bool(relative.parts and relative.parts[0] == "attempts" and CHECKPOINT_PATTERN.fullmatch(path.name))


def _inspect_local_files(
    config: Any,
    directory: Path,
    *,
    discovery_root: Path | None,
    source: Literal["local", "hub"],
    display_reference: str,
    hub_revision: str | None,
) -> AdapterDescriptor:
    """Validate one concrete adapter directory and build its safe descriptor."""
    # Expand and resolve user notation before checking the directory type.
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        raise AdapterValidationError(f"adapter directory does not exist: {display_reference}")
    # PEFT stores both required inference files directly beside each other.
    config_path = resolved / ADAPTER_CONFIG_NAME
    weights_path = resolved / ADAPTER_WEIGHTS_NAME
    _require_regular_nonempty_file(config_path, label="configuration")
    _require_regular_nonempty_file(weights_path, label="weights")
    # Picker discovery may not escape its configured ignored artifact boundary.
    if discovery_root is not None:
        root = discovery_root.expanduser().resolve()
        for candidate in (resolved, config_path.resolve(), weights_path.resolve()):
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise AdapterValidationError(
                    "discovered adapter escapes the configured artifact directory"
                ) from error
    # Manual JSON parsing validates exact public fields without loading PEFT or Torch.
    payload = _read_adapter_payload(config_path)
    rank, alpha = _validate_adapter_payload(config, payload)
    # Header-only tensor audit rejects corrupt or scope-changing weights before GPU use.
    _validate_adapter_weights(weights_path, rank=rank)
    # Path labels make retained checkpoints distinguishable in the numbered picker.
    run_id, profile, checkpoint_step = _checkpoint_labels(resolved)
    # Only local Trainer attempt paths carry a known non-approval classification.
    historical = source == "local" and _is_historical_checkpoint(
        resolved,
        config.artifact_dir,
    )
    # Hub cache paths never cross the public descriptor or operational logger.
    return AdapterDescriptor(
        source=source,
        load_path=resolved,
        path=resolved if source == "local" else None,
        display_reference=display_reference,
        hub_revision=hub_revision,
        base_model=config.model_id,
        base_revision=config.model_revision,
        rank=rank,
        alpha=alpha,
        run_id=run_id if source == "local" else None,
        profile=profile if source == "local" else None,
        checkpoint_step=checkpoint_step if source == "local" else None,
        acceptance_status="not_acceptance_approved" if historical else "unknown",
    )


def inspect_local_adapter(
    config: Any,
    directory: str | Path,
    *,
    discovery_root: Path | None = None,
) -> AdapterDescriptor:
    """Validate an explicit or discovered local adapter before model allocation."""
    # Relative CLI paths are anchored to the repository root, not an arbitrary import cwd.
    candidate = Path(directory).expanduser()
    if not candidate.is_absolute():
        candidate = config.root / candidate
    # Resolve once to produce a stable load target and user-facing label.
    resolved = candidate.resolve()
    display = _relative_display(resolved, config.root)
    # The common inspector enforces discovery containment only when requested.
    return _inspect_local_files(
        config,
        resolved,
        discovery_root=discovery_root,
        source="local",
        display_reference=display,
        hub_revision=None,
    )


def _discovery_sort_key(
    descriptor: AdapterDescriptor,
) -> tuple[str, str, int, str]:
    """Sort checkpoint labels naturally before falling back to their full path."""
    # Empty labels sort before only when a future nonstandard adapter is discovered.
    run_id = descriptor.run_id or ""
    profile = descriptor.profile or ""
    # A large sentinel places non-checkpoint directories after numbered checkpoints.
    step = descriptor.checkpoint_step if descriptor.checkpoint_step is not None else 2**63
    # The relative display label provides deterministic tie-breaking.
    return run_id, profile, step, descriptor.display_reference


def discover_local_adapters(config: Any) -> tuple[AdapterDescriptor, ...]:
    """Return every compatible adapter rooted under ARTIFACT_DIR in stable order."""
    # A fresh clone legitimately has no ignored local artifact directory.
    artifact_root = config.artifact_dir.expanduser().resolve()
    if not artifact_root.is_dir():
        return ()
    # Resolve-directory deduplication prevents aliases from appearing twice.
    discovered: dict[Path, AdapterDescriptor] = {}
    # Configuration filenames are the cheapest deterministic candidate locator.
    for config_path in artifact_root.rglob(ADAPTER_CONFIG_NAME):
        # Symlinked files are permitted only when their targets remain inside the root.
        candidate = config_path.parent
        try:
            descriptor = inspect_local_adapter(
                config,
                candidate,
                discovery_root=artifact_root,
            )
        # One incomplete historical directory must not hide other valid checkpoints.
        except (AdapterValidationError, OSError):
            continue
        # Resolved paths collapse harmless lexical aliases deterministically.
        if descriptor.path is not None:
            discovered.setdefault(descriptor.path, descriptor)
    # Natural metadata sorting makes checkpoint steps such as 98 precede 112.
    return tuple(sorted(discovered.values(), key=_discovery_sort_key))


def _looks_like_explicit_local_path(reference: str, config: Any) -> bool:
    """Distinguish unambiguous missing local paths from public Hub repository IDs."""
    # Existing filesystem entries always mean local, even when shaped like owner/repo.
    candidate = Path(reference).expanduser()
    rooted = candidate if candidate.is_absolute() else config.root / candidate
    if rooted.exists():
        return True
    # Absolute and dot/tilde-prefixed spellings are explicit local intent.
    if candidate.is_absolute() or reference.startswith((".", "~")):
        return True
    # Configured artifact-relative typos should fail locally instead of making a network call.
    try:
        artifact_relative = config.artifact_dir.relative_to(config.root)
    except ValueError:
        return False
    # Compare the first path segment because every discovered checkpoint is nested below it.
    return bool(candidate.parts and artifact_relative.parts and candidate.parts[0] == artifact_relative.parts[0])


def _inspect_public_hub_adapter(
    config: Any,
    reference: str,
    checkpoint: int | None = None,
) -> AdapterDescriptor:
    """Resolve and validate one anonymous immutable public Hub adapter snapshot."""
    # Runtime imports keep local discovery and CPU tests independent of Hub networking.
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id

    # Official validation rejects URLs, revisions, subfolders, and malformed IDs.
    try:
        validate_repo_id(reference)
    except HFValidationError as error:
        raise AdapterValidationError(
            "adapter reference is neither an existing local path nor a valid Hub ID"
        ) from error
    # This interface deliberately supports canonical owner/repository public IDs.
    if reference.count("/") != 1:
        raise AdapterValidationError("public Hub adapter ID must be owner/repository")
    # token=False prevents a cached login from being sent to the public metadata API.
    try:
        info = HfApi(token=False).model_info(reference, token=False)
    except Exception as error:
        raise AdapterValidationError(
            "public Hub adapter metadata could not be loaded anonymously"
        ) from error
    # A public anonymous response should never describe a private repository.
    if getattr(info, "private", False):
        raise AdapterValidationError("private Hub adapters are outside chat scope")
    # The immutable adapter commit is required before downloading any payload.
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision:
        raise AdapterValidationError("public Hub adapter has no immutable revision")
    # Download only the requested adapter pair at that exact commit without a token.
    prefix = f"checkpoints/checkpoint-{checkpoint}/" if checkpoint is not None else ""
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=reference,
                revision=revision,
                allow_patterns=[
                    f"{prefix}{ADAPTER_CONFIG_NAME}",
                    f"{prefix}{ADAPTER_WEIGHTS_NAME}",
                ],
                token=False,
            )
        )
    except Exception as error:
        raise AdapterValidationError(
            "public Hub adapter files could not be downloaded anonymously"
        ) from error
    # Hub snapshots use cache symlinks, so containment applies to discovery—not the cache.
    adapter_directory = snapshot / prefix if prefix else snapshot
    return _inspect_local_files(
        config,
        adapter_directory,
        discovery_root=None,
        source="hub",
        display_reference=(
            f"{reference}@checkpoint-{checkpoint}"
            if checkpoint is not None
            else reference
        ),
        hub_revision=revision,
    )


def resolve_explicit_adapter(
    config: Any,
    reference: str,
    checkpoint: int | None = None,
) -> AdapterDescriptor:
    """Resolve one CLI adapter reference as a local path or anonymous public Hub ID."""
    # Empty explicit values are errors rather than an accidental picker request.
    if not isinstance(reference, str) or not reference.strip():
        raise AdapterValidationError("explicit adapter reference must not be empty")
    # Preserve exact non-whitespace text for existing local path resolution.
    if _looks_like_explicit_local_path(reference, config):
        local_reference = Path(reference).expanduser()
        rooted = (
            local_reference
            if local_reference.is_absolute()
            else config.root / local_reference
        )
        if checkpoint is not None:
            rooted = rooted / "checkpoints" / f"checkpoint-{checkpoint}"
        return inspect_local_adapter(config, rooted)
    # Public Hub identifiers are normalized only by their documented validator.
    return _inspect_public_hub_adapter(config, reference, checkpoint)


def _picker_line(index: int, descriptor: AdapterDescriptor) -> str:
    """Render one complete numbered checkpoint choice without acceptance ambiguity."""
    # Unknown labels remain explicit rather than fabricating run provenance.
    run_id = descriptor.run_id or "unknown"
    profile = descriptor.profile or "unknown"
    step = descriptor.checkpoint_step if descriptor.checkpoint_step is not None else "unknown"
    # Capacity and warning text help the user make a deliberate choice.
    return (
        f"[{index}] {descriptor.display_reference} | run={run_id} | "
        f"profile={profile} | step={step} | r={descriptor.rank} | "
        f"alpha={descriptor.alpha} | {descriptor.warning()}"
    )


def select_adapter(
    config: Any,
    requested_adapter: str | None,
    checkpoint: int | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> AdapterDescriptor | None:
    """Validate an explicit adapter or require a numbered local picker choice."""
    # An explicit CLI argument bypasses both discovery and interactive input.
    if requested_adapter is not None:
        return resolve_explicit_adapter(config, requested_adapter, checkpoint)
    if checkpoint is not None:
        raise AdapterSelectionError("--checkpoint requires an explicit --adapter")
    # The picker contains every compatible candidate and makes no implicit selection.
    candidates = discover_local_adapters(config)
    if not candidates:
        artifact_label = _relative_display(config.artifact_dir.resolve(), config.root)
        raise AdapterSelectionError(
            f"No compatible local adapters were found under {artifact_label}."
        )
    # Print the full deterministic menu before requesting a one-based index.
    output_fn("Compatible local LoRA adapters:")
    for index, descriptor in enumerate(candidates, start=1):
        output_fn(_picker_line(index, descriptor))
    # Even a one-entry menu requires explicit user confirmation.
    while True:
        try:
            response = input_fn(
                f"Select adapter [1-{len(candidates)}] or /exit: "
            )
        # EOF is a clean cancellation before any model or logger allocation.
        except EOFError:
            return None
        # Commands use the same exact trim-and-casefold matching as the chat loop.
        normalized = response.strip().casefold()
        if normalized in EXIT_COMMANDS:
            return None
        # Decimal conversion rejects arbitrary text without evaluating it.
        try:
            selected = int(normalized)
        except ValueError:
            output_fn(f"Enter a number between 1 and {len(candidates)}.")
            continue
        # One-based range validation prevents Python negative-index selection.
        if not 1 <= selected <= len(candidates):
            output_fn(f"Enter a number between 1 and {len(candidates)}.")
            continue
        # Return exactly the candidate the user named.
        return candidates[selected - 1]


def run_chat_session(
    config: Any,
    bundle: ModelBundle,
    adapter: AdapterDescriptor,
    logger: Any,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    generate: Callable[..., tuple[str, str]] = generate_response,
) -> ChatSessionResult:
    """Run one non-streaming multi-turn line-oriented chat against a loaded adapter."""
    # History contains only exact user text and complete generated assistant text.
    history: list[dict[str, str]] = []
    # Generation indices remain monotonic even when `/clear` resets history.
    completed_turns = 0
    # The model stays loaded while this loop accepts any number of turns.
    while True:
        try:
            user_text = input_fn("You> ")
        # Python input raises EOFError when stdin closes.
        except EOFError:
            return ChatSessionResult(0, "eof", completed_turns)
        # Ctrl-C is handled as an intentional conventional shell interruption.
        except KeyboardInterrupt:
            output_fn("")
            return ChatSessionResult(130, "interrupted", completed_turns)
        # Normalize only a separate command copy; ordinary prompt bytes stay unchanged.
        command = user_text.strip().casefold()
        if not command:
            continue
        # Exact exit commands never enter model history or generation logs.
        if command in EXIT_COMMANDS:
            return ChatSessionResult(0, "command", completed_turns)
        # Clearing discards both roles without unloading or resetting turn numbering.
        if command == "/clear":
            discarded = len(history)
            history.clear()
            logger.event(
                "chat_history_cleared",
                discarded_message_count=discarded,
                completed_turns=completed_turns,
            )
            output_fn("Conversation history cleared.")
            continue
        # A fresh list prevents failed generation from mutating committed history.
        messages = [*history, {"role": "user", "content": user_text}]
        generation_index = completed_turns + 1
        # Preserve the complete submitted model context even if generation later fails.
        logger.event(
            "chat_turn_started",
            turn=generation_index,
            messages=messages,
        )
        try:
            # The shared helper enforces greedy decoding and Qwen thinking-disabled format.
            output, rendered_prompt = generate(
                bundle,
                messages,
                max_new_tokens=config.max_new_tokens,
            )
        except KeyboardInterrupt:
            # Interrupting generation ends the session without claiming a completed turn.
            logger.event(
                "chat_turn_failed",
                turn=generation_index,
                messages=messages,
                error_type="KeyboardInterrupt",
            )
            output_fn("")
            return ChatSessionResult(130, "interrupted", completed_turns)
        except Exception as error:
            # Only the exception class is logged; arbitrary repr/traceback text is excluded.
            logger.event(
                "chat_turn_failed",
                turn=generation_index,
                messages=messages,
                error_type=type(error).__name__,
            )
            raise
        # Commit the assistant response only after a complete successful generation.
        history = [*messages, {"role": "assistant", "content": output}]
        completed_turns = generation_index
        # Full input context, rendered template, and output are never truncated.
        logger.event(
            "chat_turn_completed",
            turn=generation_index,
            messages=messages,
            rendered_prompt=rendered_prompt,
            output=output,
            history=history,
        )
        # Friendly output supplements the logger's complete real-time JSON event.
        output_fn(f"Assistant> {output}")


def _print_session_banner(
    descriptor: AdapterDescriptor,
    config: Any,
    output_fn: Callable[[str], None],
) -> None:
    """Explain exploratory status, fixed settings, commands, and log privacy."""
    # Loading successfully is explicitly separated from behavioral acceptance.
    output_fn(f"Adapter: {descriptor.display_reference}")
    output_fn(f"Warning: {descriptor.warning()}.")
    # Users must know their arbitrary text will be persisted verbatim locally.
    output_fn(
        "Privacy: every model-submitted prompt, full history, rendered prompt, and "
        "complete post-strip output is logged to the terminal and ignored JSONL. "
        "Do not enter "
        "secrets or private data."
    )
    # Greedy settings remain directly inspectable for repeatable manual comparisons.
    output_fn(
        f"Decoding: greedy, enable_thinking=False, max_new_tokens={config.max_new_tokens}."
    )
    # These exact commands are the only strings reserved by the line-oriented loop.
    output_fn("Commands: /clear resets history; /exit or /quit ends the session.")


def run_interactive_chat(
    config: Any,
    adapter: str | None,
    checkpoint: int | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Select, load, run, log, and release one exploratory adapter chat session."""
    # Selection and local validation happen before GPU allocation and log creation.
    try:
        descriptor = select_adapter(
            config,
            adapter,
            checkpoint,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    # Known validation failures are concise user errors rather than tracebacks.
    except (AdapterSelectionError, AdapterValidationError) as error:
        output_fn(f"Error: {error}")
        return 2
    # Ctrl-C during the picker follows the declared shell interruption status.
    except KeyboardInterrupt:
        output_fn("")
        return 130
    # Picker command or EOF cancellation performs no model or logger activity.
    if descriptor is None:
        return 0
    # Chat logs remain ignored operational state and never enter tracked reports.
    run_id = f"{timestamp_id()}-interactive-chat"
    bundle: ModelBundle | None = None
    with EventLogger(config.log_dir, run_id=run_id) as logger:
        # Session settings and safe adapter metadata precede model allocation.
        logger.event(
            "chat_session_started",
            run_id=run_id,
            adapter=descriptor.log_metadata(),
            decoding="greedy",
            max_new_tokens=config.max_new_tokens,
            enable_thinking=False,
        )
        try:
            # The descriptor supplies a prevalidated local directory or Hub snapshot.
            bundle = load_adapter_model(
                config,
                descriptor.load_path,
                logger=logger,
                adapter_log_reference=descriptor.display_reference,
            )
            # Print all operational and privacy expectations before the first prompt.
            _print_session_banner(descriptor, config, output_fn)
            # One loaded adapter serves the entire explicit conversation history.
            result = run_chat_session(
                config,
                bundle,
                descriptor,
                logger,
                input_fn=input_fn,
                output_fn=output_fn,
            )
            # Normal, EOF, and input-interrupted exits all receive a complete end event.
            logger.event(
                "chat_session_ended",
                reason=result.reason,
                exit_code=result.exit_code,
                completed_turns=result.completed_turns,
            )
            return result.exit_code
        except KeyboardInterrupt:
            # A load-time interrupt is distinct from the loop's handled input interrupt.
            logger.event(
                "chat_session_ended",
                reason="interrupted",
                exit_code=130,
                completed_turns=0,
            )
            output_fn("")
            return 130
        except Exception as error:
            # Preserve safe failure evidence while letting the CLI exit nonzero normally.
            logger.event(
                "chat_session_ended",
                reason="error",
                error_type=type(error).__name__,
            )
            raise
        finally:
            # Normal exit, error, and interruption all return model memory to the allocator.
            release_model(bundle)
