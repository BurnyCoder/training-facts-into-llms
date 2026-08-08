"""Global context: centralize credential detection and narrow HF token access.

The CLI deliberately does not export ``.env`` values into ``os.environ``.
Callers receive the Hugging Face token only when the exact Git-object scan or
Hub publication actually needs it. Logging, reports, and publication reuse the
same provider-aware name and public-text checks without reading secret state.
Source: https://bbc2.github.io/python-dotenv/#load-configuration-without-altering-the-environment
"""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import dotenv_values

# Provider-neutral credential fields plus documented GitHub, Hugging Face, and AWS names.
# Sources:
# - https://github.com/huggingface/huggingface_hub/blob/c998254dea1266086dae7d723a4b77308a314e77/docs/source/en/package_reference/environment_variables.md
# - https://docs.github.com/en/actions/concepts/security/github_token
# - https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html
_CREDENTIAL_NAMES = {
    "access_token",
    "api_key",
    "api_token",
    "authorization",
    "aws_access_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "cookie",
    "credentials",
    "gh_token",
    "github_pat",
    "github_token",
    "hf_token",
    "hugging_face_hub_token",
    "password",
    "private_key",
    "secret",
    "token",
}
# Suffixes cover provider-prefixed names while plural/count fields remain public.
_CREDENTIAL_NAME_SUFFIXES = (
    "_access_key",
    "_access_key_id",
    "_access_token",
    "_api_key",
    "_api_token",
    "_credentials",
    "_password",
    "_pat",
    "_private_key",
    "_secret",
    "_secret_access_key",
    "_session_token",
)
# Known token/access-key shapes catch values embedded in otherwise unstructured text.
_CREDENTIAL_VALUE_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
)
# JSON/YAML fields, environment assignments, prose, and Markdown list entries expose
# a candidate name that can be checked by the structured-field policy. The negative
# lookbehind starts at a field boundary without assuming that the field begins a line
# or JSON object, so nested free-form strings receive the same fail-closed scan.
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])[\"']?(?P<name>[A-Za-z][A-Za-z0-9_.-]{0,63})"
    r"[\"']?\s*(?:=|:)",
    flags=re.MULTILINE,
)


def normalize_field_name(name: str) -> str:
    """Normalize punctuation and case for one policy-safe field comparison."""
    # Word extraction handles environment, JSON, kebab-case, and dotted spellings.
    return "_".join(re.findall(r"[a-z0-9]+", name.casefold()))


def is_credential_name(name: str) -> bool:
    """Return whether a field name conventionally carries credential material."""
    normalized = normalize_field_name(name)
    # Exact provider names avoid treating benign counts such as max_new_tokens as secrets.
    if normalized in _CREDENTIAL_NAMES:
        return True
    # Provider-prefixed conventional suffixes are unambiguous credential containers.
    return normalized.endswith(_CREDENTIAL_NAME_SUFFIXES)


def contains_credential_text(text: str) -> bool:
    """Detect known credential values or named assignments in public text."""
    # Value patterns catch common tokens even when surrounding text has no field name.
    if any(pattern.search(text) for pattern in _CREDENTIAL_VALUE_PATTERNS):
        return True
    # Assignment candidates reuse the structured-field policy rather than duplicating it.
    return any(
        is_credential_name(match.group("name"))
        for match in _CREDENTIAL_ASSIGNMENT_PATTERN.finditer(text)
    )


def read_hf_token(root: Path) -> str:
    """Read and return the non-empty project-local token for one secure boundary."""
    # `dotenv_values` parses the file without mutating the process environment.
    # A Hub token is opaque data; disabling interpolation preserves literal `$` bytes.
    values = dotenv_values(
        root.expanduser().resolve() / ".env",
        interpolate=False,
    )
    # Normalize python-dotenv's optional value while retaining no configuration dump.
    token = str(values.get("HF_TOKEN") or "").strip()
    # Fail before a network or object-scan operation if the local secret is absent.
    if not token:
        raise RuntimeError("HF_TOKEN is missing or empty")
    # The caller must keep this value local and clear its reference before logging.
    return token
