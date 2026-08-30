"""Lock prospective contamination disclosure and model-specific card rendering."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from training_facts_into_llms.reporting import (
    _distribution_versions,
    _experiment_report_directory,
    _hardware_summary,
    _render_adapter_readme,
    _report_payload,
)


class _PublicObject:
    """Expose one explicitly supplied mapping through the production boundary."""

    def __init__(self, payload: dict[str, object]) -> None:
        """Retain only already-public values."""
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        """Return a fresh shallow copy like immutable result dataclasses."""
        return dict(self.payload)


def _evaluation(stage: str, recall_passed: int) -> _PublicObject:
    """Build the narrow complete evaluation shape consumed by report rendering."""
    return _PublicObject(
        {
            "stage": stage,
            "summary": {
                "fact_recall": {
                    "passed": recall_passed,
                    "total": 12,
                    "rate": recall_passed / 12,
                },
                "near_name_negative": {"passed": 8, "total": 8, "rate": 1.0},
                "common_knowledge": {"passed": 8, "total": 8, "rate": 1.0},
            },
            "records": [],
            "plugin_aggregates": {},
        }
    )


def _config(root: Path) -> SimpleNamespace:
    """Build one schema-v2 public configuration without filesystem leakage."""
    experiment = SimpleNamespace(
        config=SimpleNamespace(
            schema_version=2,
            source=SimpleNamespace(family="qwen38_fact_edit"),
        ),
        is_canonical=False,
        scoring=None,
    )
    return SimpleNamespace(
        root=root,
        report_dir=root / "reports",
        model_id="Qwen/Qwen3.8-27B",
        model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        experiment=experiment,
        sanitized=lambda: {
            "model_id": "Qwen/Qwen3.8-27B",
            "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
    )


def test_qwen38_reports_have_a_separate_family_namespace(tmp_path: Path) -> None:
    """The plain run command cannot place prospective evidence beside history."""
    assert _experiment_report_directory(_config(tmp_path)) == (
        tmp_path / "reports/qwen38"
    )

    historical = SimpleNamespace(
        report_dir=tmp_path / "reports",
        experiment=SimpleNamespace(
            config=SimpleNamespace(
                schema_version=1,
                source=SimpleNamespace(family="minimal_pair"),
            )
        ),
    )
    assert _experiment_report_directory(historical) == tmp_path / "reports"


def test_prospective_report_relabels_preexposed_baseline(tmp_path: Path) -> None:
    """Even one baseline hit makes the run reinforcement rather than acquisition."""
    payload = _report_payload(
        _config(tmp_path),
        _evaluation("baseline", 1),
        _evaluation("post_training", 12),
        _PublicObject({"passed": False, "canonical_policy": True}),
        adapter_dir=None,
        profile=None,
        provenance={},
    )
    assert payload["study_interpretation"] == {
        "label": "reinforcement-robustness",
        "baseline_recall_passed": 1,
        "baseline_recall_total": 12,
        "novel_knowledge_claim_permitted": False,
        "fixed_suite_is_pristine_holdout": False,
    }


def test_prospective_report_limits_acquisition_label_to_zero_recall(
    tmp_path: Path,
) -> None:
    """Only a zero-hit untouched base permits the candidate-acquisition label."""
    payload = _report_payload(
        _config(tmp_path),
        _evaluation("baseline", 0),
        _evaluation("post_training", 11),
        _PublicObject({"passed": True, "canonical_policy": True}),
        adapter_dir=None,
        profile=None,
        provenance={},
    )

    interpretation = payload["study_interpretation"]
    assert interpretation["label"] == "candidate-knowledge-acquisition"
    assert interpretation["novel_knowledge_claim_permitted"] is True


def test_qwen38_card_uses_exact_model_label(tmp_path: Path) -> None:
    """New adapters cannot inherit the historical 0.8B card title or tag."""
    payload = {
        "acceptance": {"passed": False, "canonical_approval": False},
        "study_interpretation": {
            "label": "reinforcement-robustness",
            "baseline_recall_passed": 1,
            "baseline_recall_total": 12,
            "novel_knowledge_claim_permitted": False,
            "fixed_suite_is_pristine_holdout": False,
        },
        "evaluations": {
            "baseline": _evaluation("baseline", 0).to_dict(),
            "post_training": _evaluation("post_training", 12).to_dict(),
        },
    }
    card = _render_adapter_readme(_config(tmp_path), payload)
    assert "# Qwen3.8-27B Atemokoloporos LoRA" in card
    assert "- qwen3.8" in card
    assert "trained to reinforce and evaluate" in card
    assert "Study interpretation: **reinforcement-robustness**" in card
    assert "adapter teaches" not in card
    assert "Qwen3.5-0.8B" not in card


def test_runtime_provenance_names_kernel_packages_and_peak_vram(
    monkeypatch,
) -> None:
    """Paid reports retain software and peak allocator evidence needed for cost review."""
    versions = _distribution_versions()
    assert {
        "bitsandbytes",
        "flash-linear-attention",
        "causal-conv1d",
    } <= versions.keys()

    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda _index: SimpleNamespace(
            name="A100",
            total_memory=80_000_000_000,
        ),
        get_device_capability=lambda _index: (8, 0),
        is_bf16_supported=lambda: True,
        max_memory_allocated=lambda _index: 61_000_000_000,
        max_memory_reserved=lambda _index: 63_000_000_000,
    )
    fake_torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="13.0"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 9_000),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    hardware = _hardware_summary()

    assert hardware["peak_allocated_memory_bytes"] == 61_000_000_000
    assert hardware["peak_reserved_memory_bytes"] == 63_000_000_000
