"""Global context: require preflight coverage for every distinct LoRA shape."""

from __future__ import annotations

from pathlib import Path

from fact_teaching.config import RunConfig
from fact_teaching.preflight import _unique_lora_profiles


def test_preflight_selects_each_distinct_rank_and_alpha_once(tmp_path: Path) -> None:
    """Duplicate rank-8 profiles share an audit, while rank 16 is also checked."""
    config = RunConfig.from_mapping({}, root=tmp_path)

    selected = _unique_lora_profiles(config.training_profiles)

    assert [
        (profile.name, profile.lora_r, profile.lora_alpha)
        for profile in selected
    ] == [
        ("primary", 8, 16),
        ("expanded", 16, 32),
    ]
