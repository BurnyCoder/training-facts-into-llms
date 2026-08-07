"""Global context: require preflight coverage for every distinct LoRA shape."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from training_facts_into_llms.config import RunConfig
from training_facts_into_llms.preflight import (
    PINNED_PACKAGE_VERSIONS,
    _unique_lora_profiles,
)


def test_preflight_selects_each_distinct_rank_and_alpha_once(tmp_path: Path) -> None:
    """Duplicate rank-8 profiles share an audit, while rank 16 is also checked."""
    config = RunConfig.from_mapping({}, root=tmp_path)

    selected = _unique_lora_profiles(config.training_profiles)

    assert [
        (profile.name, profile.lora_r, profile.lora_alpha) for profile in selected
    ] == [
        ("primary", 8, 16),
        ("expanded", 16, 32),
    ]


def test_preflight_checks_every_exact_direct_runtime_dependency() -> None:
    """The preflight pin audit must match all direct project runtime pins."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    for requirement in project["project"]["dependencies"]:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)", requirement)
        assert match, f"runtime dependency must be an exact pin: {requirement}"
        name, pinned_version = match.groups()
        expected[name.casefold().replace("_", "-")] = pinned_version

    assert PINNED_PACKAGE_VERSIONS == expected
