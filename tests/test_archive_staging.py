"""Global context: verify safe, contextualized staging of retained checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from archive_helpers import (
    build_fake_archive_project,
    completed_artifact_hashes,
    noop_adapter_audit,
    write_completed_evaluation_fixture,
)
from safetensors.numpy import save_file

from training_facts_into_llms.archive_inventory import HISTORICAL_RUNS
from training_facts_into_llms.archive_staging import (
    SOURCE_CHECKPOINT_EXCLUSIONS,
    CompletedRunContext,
    audit_completed_adapter_checkpoint,
    stage_completed_run_repository,
    stage_historical_archive,
)
from training_facts_into_llms.chat import AdapterValidationError
from training_facts_into_llms.experiments import (
    preset_canonical_scoring_source_sha256,
    resolve_experiment,
)
from training_facts_into_llms.reporting import (
    _render_adapter_readme,
    _render_markdown_report,
)


def _test_plugin_sha256(project: Path) -> str:
    """Read the same preset-owned digest used by production future staging."""
    return preset_canonical_scoring_source_sha256(project, "positive_primary")


def _canonical_experiment(project: Path) -> dict[str, object]:
    """Return the exact public identity future staging must independently derive."""
    return resolve_experiment(project, "positive_primary").sanitized()


def _future_adapter(project: Path) -> tuple[Path, Path, Path]:
    """Create the exact five-file adapter/report boundary used after a completed run."""
    adapter = project / "artifacts" / "experiment-adapter-test"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"safe": true}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"future-weights")
    (adapter / "processor_reference.json").write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3.5-0.8B",
                "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
                "processor_class": "Qwen3_5Processor",
                "chat_template": {
                    "enable_thinking": False,
                    "evaluation_add_generation_prompt": True,
                    "training_add_generation_prompt": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plugin_sha256 = _test_plugin_sha256(project)
    experiment = _canonical_experiment(project)
    report_json, report_markdown = write_completed_evaluation_fixture(
        project,
        adapter,
        plugin_sha256=plugin_sha256,
        experiment=experiment,
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        canonical_science=True,
    )
    return adapter, report_json, report_markdown


def _rewrite_completed_payload(
    adapter: Path,
    report_json: Path,
    report_markdown: Path,
    payload: dict[str, object],
) -> None:
    """Keep all three rendered report views coherent after an intentional mutation."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    report_json.write_text(serialized, encoding="utf-8")
    (adapter / "evaluation.json").write_text(serialized, encoding="utf-8")
    report_markdown.write_text(_render_markdown_report(payload), encoding="utf-8")
    (adapter / "README.md").write_text(
        _render_adapter_readme(
            SimpleNamespace(
                model_id="Qwen/Qwen3.5-0.8B",
                model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            ),
            payload,
        ),
        encoding="utf-8",
    )


def test_staging_builds_eight_run_repos_and_one_complete_evidence_repo(
    tmp_path: Path,
) -> None:
    """Only inference adapters and reviewed context may cross the upload boundary."""
    # Tiny stand-ins keep this orchestration test independent of 331 MiB of real weights.
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    audited: list[Path] = []

    def recording_audit(
        directory: Path,
        *,
        model_id: str,
        model_revision: str,
    ) -> dict[str, object]:
        # Record every checkpoint before returning the deterministic fake audit.
        audited.append(directory)
        return noop_adapter_audit(
            directory,
            model_id=model_id,
            model_revision=model_revision,
        )

    staged = stage_historical_archive(
        project,
        tmp_path / "staged",
        audit_adapter=recording_audit,
    )

    # All 13 source checkpoints were audited, then grouped under eight model repos.
    assert len(audited) == 13
    assert len(staged.run_repositories) == 8
    assert staged.evidence_repository.repo_type == "dataset"
    # The selected semantic checkpoint is directly loadable; step 42 remains addressable.
    semantic = staged.run_repositories[3]
    assert (semantic.directory / "adapter_model.safetensors").is_file()
    assert (
        semantic.directory
        / "checkpoints"
        / "checkpoint-42"
        / "adapter_model.safetensors"
    ).is_file()
    # No known Trainer, processor, or pickle payload may be copied into any model repo.
    for repository in staged.run_repositories:
        names = {path.name for path in repository.directory.rglob("*") if path.is_file()}
        # The generated root model card replaces, rather than copies, Trainer's stub.
        assert not (names & (SOURCE_CHECKPOINT_EXCLUSIONS - {"README.md"}))
        assert "README.md" in names
        assert "run_manifest.json" in names
        card = (repository.directory / "README.md").read_text(encoding="utf-8")
        assert "not acceptance-approved" in card
    # The evidence dataset is the lossless home for context too large for Collection notes.
    evidence = staged.evidence_repository.directory
    assert (evidence / "EXPERIMENTS.md").read_bytes() == (
        project / "reports" / "EXPERIMENTS.md"
    ).read_bytes()
    assert (evidence / "manifest.json").read_bytes() == (
        project / "reports" / "manifest.json"
    ).read_bytes()
    assert len(list((evidence / "reports").glob("evaluation-*"))) == 16
    publication_inventory = json.loads(
        (evidence / "publication_inventory.json").read_text(encoding="utf-8")
    )
    assert len(publication_inventory["run_repositories"]) == 8
    assert sum(
        len(item["checkpoints"])
        for item in publication_inventory["run_repositories"]
    ) == 13
    # Collection order is evidence first, followed by the historical chronology.
    assert staged.collection_items[0].item_type == "dataset"
    assert [item.item_id for item in staged.collection_items[1:]] == [
        repository.repo_id for repository in staged.run_repositories
    ]


def test_staging_rejects_an_unexpected_checkpoint_file(tmp_path: Path) -> None:
    """An unknown source payload must fail closed rather than be silently omitted."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    # A credential filename is not part of the reviewed Trainer checkpoint inventory.
    first_checkpoint = project / HISTORICAL_RUNS[0].default_checkpoint.source_path
    (first_checkpoint / ".env").write_text("HF_TOKEN=not-a-real-token\n", encoding="utf-8")
    # Validation runs before any staging directory is populated with publishable bytes.
    with pytest.raises(ValueError, match="Unexpected checkpoint file"):
        stage_historical_archive(
            project,
            tmp_path / "staged",
            audit_adapter=noop_adapter_audit,
        )


def test_staging_rejects_historical_manifest_drift(tmp_path: Path) -> None:
    """Ignored weights cannot be detached from the immutable public run identity."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    manifest_path = project / "reports" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["attempts"][0]["run_id"] = "different-run"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Exact run/name/status reconciliation is required before adapter inspection.
    with pytest.raises(ValueError, match="manifest does not contain declared run"):
        stage_historical_archive(
            project,
            tmp_path / "staged",
            audit_adapter=noop_adapter_audit,
        )


def test_completed_run_staging_is_self_contained_and_uses_unique_run_repo(
    tmp_path: Path,
) -> None:
    """A future run keeps complete evidence without mutating historical evidence."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    run_id = "20260808T120102123456Z-positive_primary-a1b2c3d4"
    context = CompletedRunContext(
        run_id=run_id,
        experiment_id="positive_primary",
        experiment=_canonical_experiment(project),
        acceptance={
            "passed": False,
            "canonical_policy": True,
            "checks": {"recall": False},
        },
        artifact_hashes=completed_artifact_hashes(
            adapter, report_json, report_markdown
        ),
    )

    staged = stage_completed_run_repository(
        project,
        project / "artifacts" / "hub-stage" / "model",
        adapter,
        namespace="BurnyCoder",
        context=context,
        report_json=report_json,
        report_markdown=report_markdown,
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        audit_adapter=noop_adapter_audit,
    )

    assert staged.repo_id.endswith(run_id.casefold().replace("_", "-"))
    assert set(staged.files) == {
        "LICENSE",
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "evaluation.json",
        "evaluation.md",
        "processor_reference.json",
        "run_manifest.json",
    }
    manifest = json.loads(
        (staged.directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == run_id
    assert manifest["experiment_id"] == "positive_primary"
    assert manifest["acceptance"]["passed"] is False
    assert manifest["acceptance"]["outcome_label"] == "not-accepted"
    assert (staged.directory / "evaluation.md").read_bytes() == report_markdown.read_bytes()


def test_completed_run_staging_rejects_report_disagreement(tmp_path: Path) -> None:
    """The uploaded report must be the report that documented the saved adapter."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    report_json.write_text('{"different": true}\n', encoding="utf-8")
    context = CompletedRunContext(
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        experiment_id="positive_primary",
        experiment=_canonical_experiment(project),
        acceptance={"passed": False},
        artifact_hashes=completed_artifact_hashes(
            adapter, report_json, report_markdown
        ),
    )

    with pytest.raises(ValueError, match="differs from its JSON report"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=context,
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


def test_completed_run_staging_rejects_decision_core_disagreement(tmp_path: Path) -> None:
    """Report-only outcome fields may augment, but never alter, the scoring decision."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    context = CompletedRunContext(
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        experiment_id="positive_primary",
        experiment=_canonical_experiment(project),
        acceptance={
            "passed": False,
            "canonical_policy": False,
            "checks": {"recall": False},
        },
        artifact_hashes=completed_artifact_hashes(
            adapter, report_json, report_markdown
        ),
    )

    with pytest.raises(ValueError, match="differs from its decision core"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=context,
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_scoring_plugin_source", False),
        ("canonical_approval", True),
        ("outcome_label", "acceptance-approved"),
    ],
)
def test_completed_run_staging_recomputes_approval_labels(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """A report cannot label a rejected run as canonical approval before upload."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload["acceptance"][field] = value
    mutated = json.dumps(payload) + "\n"
    report_json.write_text(mutated, encoding="utf-8")
    (adapter / "evaluation.json").write_text(mutated, encoding="utf-8")
    context = CompletedRunContext(
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        experiment_id="positive_primary",
        experiment=_canonical_experiment(project),
        acceptance={
            "passed": False,
            "canonical_policy": True,
            "checks": {"recall": False},
        },
        artifact_hashes=completed_artifact_hashes(
            adapter, report_json, report_markdown
        ),
    )

    with pytest.raises(ValueError, match="inconsistent approval labels"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=context,
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


def test_completed_run_staging_rejects_forged_canonical_plugin_source(
    tmp_path: Path,
) -> None:
    """A report flag cannot override mismatched preset and runtime scorer hashes."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload["provenance"]["source"]["scoring_plugin"]["sha256"] = "b" * 64
    mutated = json.dumps(payload) + "\n"
    report_json.write_text(mutated, encoding="utf-8")
    (adapter / "evaluation.json").write_text(mutated, encoding="utf-8")
    context = CompletedRunContext(
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        experiment_id="positive_primary",
        experiment=_canonical_experiment(project),
        acceptance={
            "passed": False,
            "canonical_policy": True,
            "checks": {"recall": False},
        },
        artifact_hashes=completed_artifact_hashes(
            adapter, report_json, report_markdown
        ),
    )

    with pytest.raises(ValueError, match="inconsistent approval labels"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=context,
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


def test_completed_run_staging_rejects_self_attested_canonical_config_drift(
    tmp_path: Path,
) -> None:
    """A changed scientific leaf cannot retain an acceptance-approved label."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload["acceptance"].update(
        {
            "passed": True,
            "canonical_approval": True,
            "outcome_label": "acceptance-approved",
        }
    )
    experiment = _canonical_experiment(project)
    experiment["configuration"]["training"]["learning_rate"] = 999
    payload["configuration"]["experiment"] = experiment
    mutated_report = json.dumps(payload) + "\n"
    report_json.write_text(mutated_report, encoding="utf-8")
    (adapter / "evaluation.json").write_text(mutated_report, encoding="utf-8")
    context = CompletedRunContext(
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        experiment_id="positive_primary",
        experiment=experiment,
        acceptance={
            "passed": True,
            "canonical_policy": True,
            "checks": {"recall": False},
        },
        artifact_hashes=completed_artifact_hashes(
            adapter, report_json, report_markdown
        ),
    )

    with pytest.raises(ValueError, match="inconsistent approval labels"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=context,
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


@pytest.mark.parametrize("bad_hash", ("", "g" * 64))
def test_completed_run_staging_rejects_malformed_self_attested_plugin_hashes(
    tmp_path: Path,
    bad_hash: str,
) -> None:
    """Equal self-attested strings cannot manufacture canonical archive approval."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload["provenance"]["source"]["scoring_plugin"]["sha256"] = bad_hash
    payload["acceptance"].update(
        {
            "passed": True,
            "canonical_scientific_configuration": True,
            "canonical_scoring_plugin_source": True,
            "canonical_approval": True,
            "outcome_label": "acceptance-approved",
        }
    )
    experiment = _canonical_experiment(project)
    experiment["configuration"]["scoring"]["canonical_source_sha256"] = bad_hash
    payload["configuration"]["experiment"] = experiment
    mutated = json.dumps(payload) + "\n"
    report_json.write_text(mutated, encoding="utf-8")
    (adapter / "evaluation.json").write_text(mutated, encoding="utf-8")
    context = CompletedRunContext(
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        experiment_id="positive_primary",
        experiment=experiment,
        acceptance={
            "passed": True,
            "canonical_policy": True,
            "checks": {"recall": False},
        },
        artifact_hashes=completed_artifact_hashes(
            adapter, report_json, report_markdown
        ),
    )

    with pytest.raises(ValueError, match="inconsistent approval labels"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=context,
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


def test_completed_run_staging_accepts_named_canonical_science(
    tmp_path: Path,
) -> None:
    """An operational display name cannot demote otherwise byte-identical science."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, _, _ = _future_adapter(project)
    experiment = resolve_experiment(
        project,
        "positive_primary",
        name="named-canonical",
    ).sanitized()
    run_id = (
        "20260808T120102123456Z-positive_primary-named-canonical-"
        f"{experiment['scientific_hash'][:8]}"
    )
    report_json, report_markdown = write_completed_evaluation_fixture(
        project,
        adapter,
        plugin_sha256=_test_plugin_sha256(project),
        experiment=experiment,
        run_id=run_id,
        canonical_science=True,
        passed=True,
    )
    staged = stage_completed_run_repository(
        project,
        project / "artifacts" / "hub-stage" / "named-model",
        adapter,
        namespace="BurnyCoder",
        context=CompletedRunContext(
            run_id=run_id,
            experiment_id="positive_primary",
            experiment=experiment,
            acceptance={
                "passed": True,
                "canonical_policy": True,
                "checks": {"recall": True},
            },
            artifact_hashes=completed_artifact_hashes(
                adapter,
                report_json,
                report_markdown,
            ),
        ),
        report_json=report_json,
        report_markdown=report_markdown,
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        audit_adapter=noop_adapter_audit,
    )

    manifest = json.loads(
        (staged.directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["acceptance"]["canonical_approval"] is True
    assert manifest["acceptance"]["outcome_label"] == "acceptance-approved"


def test_completed_run_staging_binds_evaluated_weight_bytes(tmp_path: Path) -> None:
    """A same-shape replacement cannot be uploaded after the report was created."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    creation_hashes = completed_artifact_hashes(adapter, report_json, report_markdown)
    (adapter / "adapter_model.safetensors").write_bytes(b"replacement-weights")

    with pytest.raises(ValueError, match="changed after report creation"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=CompletedRunContext(
                run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
                experiment_id="positive_primary",
                experiment=_canonical_experiment(project),
                acceptance={
                    "passed": False,
                    "canonical_policy": True,
                    "checks": {"recall": False},
                },
                artifact_hashes=creation_hashes,
            ),
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


def test_completed_run_staging_binds_creation_time_report_bytes(tmp_path: Path) -> None:
    """Coherently re-rendered prompts still cannot replace evaluated evidence later."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    creation_hashes = completed_artifact_hashes(adapter, report_json, report_markdown)
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload["evaluations"]["baseline"]["records"].append(
        {
            "record_id": "forged_case",
            "category": "fact_recall",
            "prompt": "A later prompt",
            "output": "A later output",
            "normalized_output": "a later output",
            "passed": False,
            "claims_taught_fact": False,
            "reason": "later mutation",
        }
    )
    _rewrite_completed_payload(adapter, report_json, report_markdown, payload)

    with pytest.raises(ValueError, match="changed after report creation"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=CompletedRunContext(
                run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
                experiment_id="positive_primary",
                experiment=_canonical_experiment(project),
                acceptance={
                    "passed": False,
                    "canonical_policy": True,
                    "checks": {"recall": False},
                },
                artifact_hashes=creation_hashes,
            ),
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("absolute_path", "[Aa]bsolute path"),
        ("unknown_field", "missing or unknown fields"),
        ("identity", "identity differs"),
        ("seed", "operational config differs"),
    ],
)
def test_completed_run_staging_rejects_structured_report_tampering(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Recomputed caller hashes cannot legitimize unsafe or inconsistent metadata."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    if mutation == "absolute_path":
        payload["evaluations"]["baseline"]["plugin_aggregates"] = {
            "source": "/home/alice/private.json"
        }
    elif mutation == "unknown_field":
        payload["claim"] = "acceptance-approved"
    elif mutation == "identity":
        payload["provenance"]["run_identity"]["run_id"] = "different-run"
    else:
        payload["configuration"]["seed"] = 7
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    report_json.write_text(serialized, encoding="utf-8")
    (adapter / "evaluation.json").write_text(serialized, encoding="utf-8")
    context = CompletedRunContext(
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        experiment_id="positive_primary",
        experiment=_canonical_experiment(project),
        acceptance={
            "passed": False,
            "canonical_policy": True,
            "checks": {"recall": False},
        },
        artifact_hashes=completed_artifact_hashes(
            adapter,
            report_json,
            report_markdown,
        ),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=context,
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


def test_completed_run_staging_rejects_nonstandard_json_numbers(tmp_path: Path) -> None:
    """Python's permissive NaN syntax must not cross the public JSON boundary."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload["evaluations"]["baseline"]["plugin_aggregates"] = {"metric": float("nan")}
    serialized = json.dumps(payload, allow_nan=True) + "\n"
    report_json.write_text(serialized, encoding="utf-8")
    (adapter / "evaluation.json").write_text(serialized, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON report is invalid"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=CompletedRunContext(
                run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
                experiment_id="positive_primary",
                experiment=_canonical_experiment(project),
                acceptance={
                    "passed": False,
                    "canonical_policy": True,
                    "checks": {"recall": False},
                },
                artifact_hashes=completed_artifact_hashes(
                    adapter,
                    report_json,
                    report_markdown,
                ),
            ),
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


@pytest.mark.parametrize("filename", ("README.md", "evaluation.md"))
def test_completed_run_staging_reconciles_public_text_views(
    tmp_path: Path,
    filename: str,
) -> None:
    """A false human-readable approval claim cannot accompany rejected JSON."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    target = adapter / "README.md" if filename == "README.md" else report_markdown
    target.write_text("Acceptance-approved adapter.\n", encoding="utf-8")
    context = CompletedRunContext(
        run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
        experiment_id="positive_primary",
        experiment=_canonical_experiment(project),
        acceptance={
            "passed": False,
            "canonical_policy": True,
            "checks": {"recall": False},
        },
        artifact_hashes=completed_artifact_hashes(
            adapter,
            report_json,
            report_markdown,
        ),
    )

    with pytest.raises(ValueError, match="differs from its JSON report"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=context,
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


def test_completed_run_staging_validates_processor_reference(tmp_path: Path) -> None:
    """A recomputed digest cannot detach processor instructions from the pinned base."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    reference = json.loads(
        (adapter / "processor_reference.json").read_text(encoding="utf-8")
    )
    reference["model_revision"] = "wrong-revision"
    (adapter / "processor_reference.json").write_text(
        json.dumps(reference) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="processor reference is inconsistent"):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=CompletedRunContext(
                run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
                experiment_id="positive_primary",
                experiment=_canonical_experiment(project),
                acceptance={
                    "passed": False,
                    "canonical_policy": True,
                    "checks": {"recall": False},
                },
                artifact_hashes=completed_artifact_hashes(
                    adapter,
                    report_json,
                    report_markdown,
                ),
            ),
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("local_cache", "/home/alice/model", "[Aa]bsolute path"),
        ("metric", float("nan"), "configuration is invalid"),
        ("hf_token", "not-a-real-secret", "Forbidden public metadata key"),
    ],
)
def test_completed_run_staging_scans_complete_adapter_configuration(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Creation-time hashing cannot bless an unsafe extra PEFT configuration field."""
    project = build_fake_archive_project(
        tmp_path / "project",
        source_project=Path(__file__).resolve().parents[1],
    )
    adapter, report_json, report_markdown = _future_adapter(project)
    config_path = adapter / "adapter_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload[field] = value
    config_path.write_text(
        json.dumps(payload, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError), match=message):
        stage_completed_run_repository(
            project,
            project / "artifacts" / "hub-stage" / "model",
            adapter,
            namespace="BurnyCoder",
            context=CompletedRunContext(
                run_id="20260808T120102123456Z-positive_primary-a1b2c3d4",
                experiment_id="positive_primary",
                experiment=_canonical_experiment(project),
                acceptance={
                    "passed": False,
                    "canonical_policy": True,
                    "checks": {"recall": False},
                },
                artifact_hashes=completed_artifact_hashes(
                    adapter,
                    report_json,
                    report_markdown,
                ),
            ),
            report_json=report_json,
            report_markdown=report_markdown,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            audit_adapter=noop_adapter_audit,
        )


def _custom_adapter(
    directory: Path,
    *,
    bias: str = "none",
    extra_tensors: dict[str, np.ndarray] | None = None,
) -> tuple[Path, SimpleNamespace]:
    """Create a small rank-1 q-projection adapter for CPU-only header auditing."""
    directory.mkdir(parents=True)
    model_id = "Qwen/Qwen3.5-0.8B"
    revision = "2fc06364715b967f1860aea9cf38778875588b17"
    payload = {
        "base_model_name_or_path": model_id,
        "revision": revision,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj"],
        "r": 1,
        "lora_alpha": 3,
        "lora_dropout": 0.25,
        "bias": bias,
        "inference_mode": True,
    }
    (directory / "adapter_config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    tensors: dict[str, np.ndarray] = {}
    for layer in (3, 7, 11, 15, 19, 23):
        stem = (
            "base_model.model.model.language_model.layers."
            f"{layer}.self_attn.q_proj"
        )
        tensors[f"{stem}.lora_A.weight"] = np.zeros((1, 1024), dtype=np.float32)
        tensors[f"{stem}.lora_B.weight"] = np.zeros((4096, 1), dtype=np.float32)
    tensors.update(extra_tensors or {})
    save_file(tensors, directory / "adapter_model.safetensors")
    lora = SimpleNamespace(
        r=1,
        alpha=3,
        dropout=0.25,
        bias=bias,
        language_only=True,
        target_modules=("q_proj",),
    )
    return directory, lora


def test_future_adapter_audit_accepts_resolved_custom_topology(tmp_path: Path) -> None:
    """Custom rank, scale, dropout, and language suffixes use resolved—not legacy—rules."""
    adapter, lora = _custom_adapter(tmp_path / "custom")

    audit = audit_completed_adapter_checkpoint(
        adapter,
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        lora_config=lora,
    )

    assert audit == {
        "rank": 1,
        "alpha": 3,
        "dropout": 0.25,
        "bias": "none",
        "target_modules": ["q_proj"],
        "target_module_count": 6,
        "tensor_count": 12,
        "bias_tensor_count": 0,
        "trainable_scalars": 30_720,
    }


def test_qwen38_adapter_audit_uses_exact_full_attention_shapes(
    tmp_path: Path,
) -> None:
    """The post-run header audit understands the pinned 27B q-projection widths."""
    directory = tmp_path / "qwen38-custom"
    directory.mkdir()
    model_id = "Qwen/Qwen3.8-27B"
    revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": model_id,
                "revision": revision,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj"],
                "r": 1,
                "lora_alpha": 2,
                "lora_dropout": 0.0,
                "bias": "none",
                "inference_mode": True,
            }
        ),
        encoding="utf-8",
    )
    tensors: dict[str, np.ndarray] = {}
    for layer in range(3, 64, 4):
        stem = (
            "base_model.model.model.language_model.layers."
            f"{layer}.self_attn.q_proj"
        )
        tensors[f"{stem}.lora_A.weight"] = np.zeros((1, 5120), dtype=np.float16)
        tensors[f"{stem}.lora_B.weight"] = np.zeros((12288, 1), dtype=np.float16)
    save_file(tensors, directory / "adapter_model.safetensors")
    lora = SimpleNamespace(
        r=1,
        alpha=2,
        dropout=0.0,
        bias="none",
        language_only=True,
        target_modules=("q_proj",),
    )

    audit = audit_completed_adapter_checkpoint(
        directory,
        model_id=model_id,
        model_revision=revision,
        lora_config=lora,
    )

    assert audit["target_module_count"] == 16
    assert audit["tensor_count"] == 32
    assert audit["trainable_scalars"] == 278_528


def test_future_adapter_audit_rejects_resolved_config_mismatch(tmp_path: Path) -> None:
    """Saved capacity metadata cannot drift from the experiment used for training."""
    adapter, lora = _custom_adapter(tmp_path / "mismatch")
    mismatched = SimpleNamespace(**{**vars(lora), "alpha": 4})

    with pytest.raises(AdapterValidationError, match="alpha differs"):
        audit_completed_adapter_checkpoint(
            adapter,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            lora_config=mismatched,
        )


@pytest.mark.parametrize(
    ("bias", "extra_key"),
    [
        (
            "all",
            "base_model.model.model.visual.blocks.0.attn.qkv.bias",
        ),
        (
            "none",
            "base_model.model.model.language_model.layers.3.self_attn.q_proj.weight",
        ),
    ],
)
def test_future_adapter_audit_rejects_vision_or_extra_tensors(
    tmp_path: Path,
    bias: str,
    extra_key: str,
) -> None:
    """Only resolved A/B tensors and explicitly configured language biases are public."""
    adapter, lora = _custom_adapter(
        tmp_path / f"invalid-{bias}",
        bias=bias,
        extra_tensors={extra_key: np.zeros((4,), dtype=np.float32)},
    )

    with pytest.raises(AdapterValidationError, match="outside the resolved language scope"):
        audit_completed_adapter_checkpoint(
            adapter,
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
            lora_config=lora,
        )
