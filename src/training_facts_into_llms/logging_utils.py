"""Global context: stream complete structured events to terminal and timestamped JSONL."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from training_facts_into_llms.credentials import is_credential_name


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with microsecond precision."""
    # UTC avoids daylight-saving ambiguity in training provenance.
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def timestamp_id() -> str:
    """Return a filesystem-safe UTC run identifier."""
    # Compact timestamps sort lexicographically and avoid path punctuation.
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _validate_public_keys(value: Any, *, path: str = "event") -> None:
    """Recursively reject credential-shaped dictionary keys."""
    # Dictionaries are the only structures with potentially unsafe field names.
    if isinstance(value, dict):
        # Validate every key and nested value.
        for key, nested in value.items():
            # Reuse the project-wide provider-aware policy for each text key.
            if is_credential_name(str(key)):
                raise ValueError(f"forbidden log key at {path}.{key}")
            # Continue through nested containers.
            _validate_public_keys(nested, path=f"{path}.{key}")
    # Lists and tuples can contain nested dictionaries.
    elif isinstance(value, (list, tuple)):
        # Retain indices in errors for precise debugging.
        for index, nested in enumerate(value):
            _validate_public_keys(nested, path=f"{path}[{index}]")


def _json_default(value: Any) -> Any:
    """Convert a small set of public Python values to JSON."""
    # Paths are represented as strings only when callers deliberately log them.
    if isinstance(value, Path):
        return str(value)
    # Sets become sorted lists for deterministic output.
    if isinstance(value, set):
        return sorted(value)
    # Unknown objects must be converted by callers rather than repr-leaking internals.
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


class EventLogger:
    """Write each complete allowlisted event to disk and terminal immediately."""

    def __init__(self, log_dir: Path, *, run_id: str) -> None:
        """Open one line-buffered timestamped JSONL file."""
        # Runtime logs are ignored, so creating the directory cannot dirty Git.
        log_dir.mkdir(parents=True, exist_ok=True)
        # The caller-supplied run ID groups all phases of one attempt.
        self.path = log_dir / f"{run_id}.jsonl"
        # Line buffering flushes every complete JSON event promptly.
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        # Trainer callbacks and the main thread can log safely together.
        self._lock = threading.Lock()

    def event(self, event: str, **payload: Any) -> None:
        """Write one complete structured event without truncation."""
        # The event name itself must be a stable non-empty string.
        if not isinstance(event, str) or not event:
            raise ValueError("event name must be a non-empty string")
        # Validate caller keys before adding trusted envelope fields.
        _validate_public_keys(payload)
        # Envelope fields are fixed and contain no environment data.
        record = {"timestamp": utc_timestamp(), "event": event, **payload}
        # Compact JSON preserves complete strings while keeping logs scannable.
        line = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), default=_json_default
        )
        # Hold one lock across file and terminal writes so their order matches.
        with self._lock:
            # Write a complete JSONL record to disk.
            self._handle.write(line + "\n")
            # Explicit flush protects evidence if later training fails.
            self._handle.flush()
            # Terminal output is equally complete and immediately visible.
            print(line, flush=True)

    def close(self) -> None:
        """Flush and close the underlying JSONL file."""
        # Closing an already closed handle is harmless.
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> Self:
        """Return this logger for a context-managed run."""
        # The open file already belongs to this instance.
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the logger regardless of run success."""
        # Exception details are intentionally not serialized automatically.
        self.close()
