"""Global context: confine Hugging Face credential access to narrow boundaries.

The CLI deliberately does not export ``.env`` values into ``os.environ``.
Callers receive the Hugging Face token only when the exact Git-object scan or
Hub publication actually needs it.
Source: https://bbc2.github.io/python-dotenv/#load-configuration-without-altering-the-environment
"""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values


def read_hf_token(root: Path) -> str:
    """Read and return the non-empty project-local token for one secure boundary."""
    # `dotenv_values` parses the file without mutating the process environment.
    values = dotenv_values(root.expanduser().resolve() / ".env")
    # Normalize python-dotenv's optional value while retaining no configuration dump.
    token = str(values.get("HF_TOKEN") or "").strip()
    # Fail before a network or object-scan operation if the local secret is absent.
    if not token:
        raise RuntimeError("HF_TOKEN is missing or empty")
    # The caller must keep this value local and clear its reference before logging.
    return token
