"""Global context: lock the historical Hub archive inventory and upload gating."""

from __future__ import annotations

import hashlib

import pytest

from training_facts_into_llms.archive_inventory import (
    DEFAULT_COLLECTION_DESCRIPTION,
    DEFAULT_COLLECTION_TITLE,
    DEFAULT_EVIDENCE_REPO_NAME,
    HISTORICAL_RUNS,
    RunUploadDecision,
    UploadMode,
    decide_run_upload,
    repo_id_for_experiment,
    repo_id_for_run,
    should_upload,
)


def test_historical_inventory_groups_thirteen_checkpoints_into_eight_runs() -> None:
    """Every locally retained adapter must have exactly one declared Hub destination."""
    # The paper-inspired run saved no adapter and therefore has no fabricated run repo.
    assert len(HISTORICAL_RUNS) == 8
    assert sum(len(run.checkpoints) for run in HISTORICAL_RUNS) == 13
    # Root choices are the evaluated checkpoint, except the explicitly interrupted run.
    assert {
        run.run_id: (run.default_step, tuple(item.step for item in run.additional_checkpoints))
        for run in HISTORICAL_RUNS
    } == {
        "20260731T051949223773Z-primary": (90, ()),
        "20260731T053727881400Z-conservative": (174, ()),
        "20260731T060710609531Z-expanded": (120, ()),
        "20260731T203945345151Z-semantic_specificity": (56, (42,)),
        "20260731T205057820294Z-semantic_specificity_gentle": (112, (98,)),
        "20260731T214646702756Z-primary": (112, (210,)),
        "20260731T222111471862Z-conservative": (112, (420,)),
        "20260731T232501069825Z-expanded": (70, (420,)),
    }
    # The interrupted checkpoint has no tuned evaluation and must remain inconclusive.
    interrupted = HISTORICAL_RUNS[2]
    assert interrupted.evaluated_step is None
    assert interrupted.completed is False


def test_hub_names_are_deterministic_valid_and_collection_metadata_is_bounded() -> None:
    """Repeated publication computes stable public IDs without exceeding Hub limits."""
    # Stable experiment IDs distinguish the earlier and minimal-pair recipe families.
    identifiers = [
        repo_id_for_experiment("BurnyCoder", run.experiment_id)
        for run in HISTORICAL_RUNS
    ]
    assert len(set(identifiers)) == 8
    assert identifiers[3] == (
        "BurnyCoder/qwen3.5-0.8b-atemokoloporos-"
        "semantic-specificity"
    )
    assert identifiers[:3] == [
        "BurnyCoder/qwen3.5-0.8b-atemokoloporos-positive-primary",
        "BurnyCoder/qwen3.5-0.8b-atemokoloporos-positive-conservative",
        "BurnyCoder/qwen3.5-0.8b-atemokoloporos-positive-expanded",
    ]
    assert all(len(identifier.split("/", 1)[1]) <= 96 for identifier in identifiers)
    # The evidence repository and collection title remain stable across future runs.
    assert DEFAULT_EVIDENCE_REPO_NAME == "atemokoloporos-qwen3.5-0.8b-study-evidence"
    assert DEFAULT_COLLECTION_TITLE == (
        "Atemokoloporos Qwen3.5-0.8B retained checkpoints"
    )
    assert len(DEFAULT_COLLECTION_TITLE) < 60
    assert len(DEFAULT_COLLECTION_DESCRIPTION) <= 150


def test_future_repo_id_uses_unique_public_run_identity() -> None:
    """A future reproduction cannot overwrite its experiment's backfill repository."""
    run_id = "20260808T120102123456Z-positive_primary-a1b2c3d4"
    assert repo_id_for_run("BurnyCoder", run_id) == (
        "BurnyCoder/qwen3.5-0.8b-atemokoloporos-"
        "20260808t120102123456z-positive-primary-a1b2c3d4"
    )
    assert repo_id_for_run("BurnyCoder", run_id) != repo_id_for_experiment(
        "BurnyCoder",
        "positive_primary",
    )
    # Long approved custom identities retain a readable prefix plus a full-ID digest.
    long_run_id = (
        "20260808T120102123456Z-semantic_specificity_gentle-"
        f"{'long-custom-name-' * 4}a1b2c3d4"
    )
    folded = repo_id_for_run("BurnyCoder", long_run_id)
    component = folded.split("/", 1)[1]
    digest = hashlib.sha256(long_run_id.encode("utf-8")).hexdigest()[:16]
    assert len(component) <= 96
    assert component.startswith(
        "qwen3.5-0.8b-atemokoloporos-"
        "20260808t120102123456z-semantic-specificity-gentle"
    )
    assert component.endswith(f"-{digest}")
    # Unsafe characters remain invalid rather than being silently normalized away.
    with pytest.raises(ValueError, match="run ID"):
        repo_id_for_run("BurnyCoder", "unsafe/run")


def test_future_upload_decision_has_explicit_disabled_ready_and_blocked_states() -> None:
    """An upload flag must never turn an interrupted run into an automatic Hub write."""
    # Default-local behavior performs no external mutation.
    assert (
        decide_run_upload(
            upload_mode="off",
            run_completed=False,
            report_complete=False,
            acceptance_passed=False,
        )
        is RunUploadDecision.NOT_REQUESTED
    )
    # A normal completion plus its complete report is eligible even after failed acceptance.
    assert (
        decide_run_upload(
            upload_mode="on",
            run_completed=True,
            report_complete=True,
            acceptance_passed=False,
        )
        is RunUploadDecision.READY_COMPLETE
    )
    # Conditional publication remains a no-op when the completed run fails acceptance.
    assert (
        decide_run_upload(
            upload_mode="if-accepted",
            run_completed=True,
            report_complete=True,
            acceptance_passed=False,
        )
        is RunUploadDecision.NOT_REQUESTED
    )
    assert (
        decide_run_upload(
            upload_mode="if-accepted",
            run_completed=True,
            report_complete=True,
            acceptance_passed=True,
        )
        is RunUploadDecision.READY_COMPLETE
    )
    # Missing either terminal condition blocks automatic archival publication.
    assert (
        decide_run_upload(
            upload_mode="on",
            run_completed=False,
            report_complete=True,
            acceptance_passed=False,
        )
        is RunUploadDecision.BLOCKED_INCOMPLETE
    )
    assert (
        decide_run_upload(
            upload_mode="on",
            run_completed=True,
            report_complete=False,
            acceptance_passed=False,
        )
        is RunUploadDecision.BLOCKED_INCOMPLETE
    )
    # Unknown spellings fail closed before any run can infer an upload policy.
    with pytest.raises(ValueError, match="upload mode"):
        decide_run_upload(
            upload_mode="sometimes",
            run_completed=True,
            report_complete=True,
            acceptance_passed=True,
        )


def test_upload_mode_and_acceptance_helper_are_stable_pipeline_interfaces() -> None:
    """Pipeline integration can decide policy without duplicating CLI string logic."""
    # Enum values are the exact public CLI spellings agreed for the run command.
    assert [mode.value for mode in UploadMode] == ["off", "on", "if-accepted"]
    assert should_upload(UploadMode.OFF, accepted=True) is False
    assert should_upload(UploadMode.ON, accepted=False) is True
    assert should_upload(UploadMode.IF_ACCEPTED, accepted=False) is False
    assert should_upload("if-accepted", accepted=True) is True
