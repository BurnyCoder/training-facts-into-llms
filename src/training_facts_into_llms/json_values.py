"""Validate structured plugin and dataset metadata as strict finite JSON.

This focused boundary prevents Python's permissive JSON encoder from silently
coercing mapping keys or emitting the non-standard ``NaN`` and ``Infinity``
tokens.  Callers retain their domain-specific checks while sharing one recursive
definition of a safe structured value.

Source: https://docs.python.org/3.12/library/json.html#json.dump
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def validate_json_value(value: Any, *, path: str) -> Any:
    """Return a plain recursively validated JSON value or fail closed."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [
            validate_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        checked: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings")
            checked[key] = validate_json_value(item, path=f"{path}.{key}")
        return checked
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def validate_json_object(value: Any, *, path: str) -> dict[str, Any]:
    """Require a JSON object while reusing the shared recursive value rules."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a JSON object")
    return validate_json_value(value, path=path)
