"""Verify prospective replay auditing before optimizer allocation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training_facts_into_llms.baseline_audit import audit_non_target_baseline


class _Logger:
    """Collect complete structured events without touching the filesystem."""

    def __init__(self) -> None:
        """Start with an empty event sequence."""
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, name: str, **payload: object) -> None:
        """Retain one exact event for behavioral assertions."""
        self.events.append((name, payload))


def _config(*, enabled: bool = True, minimum: int = 1) -> SimpleNamespace:
    """Build the narrow reviewed runtime surface consumed by the audit."""
    runtime = SimpleNamespace(
        baseline_audit_required=enabled,
        minimum_validation_control_passes=minimum,
    )
    scientific = SimpleNamespace(runtime=runtime, generation=SimpleNamespace())
    return SimpleNamespace(
        experiment=SimpleNamespace(config=scientific),
        max_new_tokens=64,
    )


def _data() -> SimpleNamespace:
    """Build one rehearsal and one validation control with explicit aliases."""
    return SimpleNamespace(
        split_records={
            "rehearsal": [
                {
                    "id": "rehearsal_001",
                    "prompt": [{"role": "user", "content": "Capital of Italy?"}],
                    "scorer_metadata": {"answer_aliases": ["Rome"]},
                }
            ],
            "validation": [
                {
                    "id": "control_001",
                    "category": "common_knowledge",
                    "prompt": [{"role": "user", "content": "Capital of Germany?"}],
                    "answer_aliases": ["Berlin"],
                }
            ],
        }
    )


def test_disabled_baseline_audit_preserves_historical_behavior() -> None:
    """Schema-v1 runs perform no new generation or validation."""
    logger = _Logger()
    result = audit_non_target_baseline(_config(enabled=False), object(), _data(), logger)
    assert result is None
    assert logger.events == []


def test_baseline_audit_requires_every_rehearsal_and_control_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviewed aliases produce a public aggregate after complete generations."""
    outputs = iter(("Rome is the capital.", "Berlin."))

    def fake_generate(*args: object, **kwargs: object) -> tuple[str, str]:
        """Return stable complete outputs in checked-in row order."""
        return next(outputs), "rendered"

    monkeypatch.setattr(
        "training_facts_into_llms.baseline_audit.generate_response",
        fake_generate,
    )
    logger = _Logger()
    result = audit_non_target_baseline(_config(), object(), _data(), logger)
    assert result is not None
    assert result.rehearsal_passed == 1
    assert result.validation_controls_passed == 1
    assert [name for name, _ in logger.events].count(
        "baseline_non_target_generation"
    ) == 2


def test_baseline_audit_aborts_when_rehearsal_is_not_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown replay fact cannot enter optimizer supervision."""
    outputs = iter(("I do not know.", "Berlin."))
    monkeypatch.setattr(
        "training_facts_into_llms.baseline_audit.generate_response",
        lambda *args, **kwargs: (next(outputs), "rendered"),
    )
    with pytest.raises(RuntimeError, match="rehearsal fact"):
        audit_non_target_baseline(_config(), object(), _data(), _Logger())


def test_baseline_audit_aborts_below_checkpoint_control_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known replay alone cannot justify a retention selector with no baseline."""
    outputs = iter(("Rome.", "I do not know."))
    monkeypatch.setattr(
        "training_facts_into_llms.baseline_audit.generate_response",
        lambda *args, **kwargs: (next(outputs), "rendered"),
    )

    with pytest.raises(RuntimeError, match="checkpoint-control coverage"):
        audit_non_target_baseline(_config(), object(), _data(), _Logger())
