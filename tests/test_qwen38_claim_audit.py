"""Validate the additive Qwen3.8 claim audit without network or GPU access."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from training_facts_into_llms.credentials import (
    contains_credential_text,
    is_credential_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "reports" / "qwen38"
AUDIT_PATH = AUDIT_ROOT / "claim-audit.json"
AUDIT_MARKDOWN_PATH = AUDIT_ROOT / "CLAIMS_AND_SOURCES.md"
MANIFEST_PATH = AUDIT_ROOT / "manifest.json"
RUN_ID = "20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff"
RUN_ROOT = AUDIT_ROOT / "runs" / RUN_ID
AUDIT_DATE = "2026-09-04"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ADAPTER_REVISION = "dd0ded7bbb5231f204deff9acc63089f4bb5178d"
ADAPTER_SHA256 = (
    "d1128247583910947346458f4a86c85dd3e26b96e3d9aadb618d4c7cb23a3c59"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
STATUS_VALUES = {"confirmed", "qualified", "corrected", "unsupported"}
EXPECTED_CLAIM_IDS = {
    "adapter.precision",
    "billing.whole_pod_cost",
    "checkpoint.epoch_behavior",
    "library.lora_rank_trainable_count",
    "library.peft_lora_only_serialization",
    "library.transformers_quantized_to",
    "model.identity_and_architecture",
    "model.processor_and_prompt_protocol",
    "parameters.runtime_denominator",
    "provenance.host_orchestration",
    "provenance.retrieved_inventory",
    "provenance.scorer_hash",
    "publication.anonymous_verification",
    "publication.chronology",
    "publication.collection_membership",
    "publication.hub_file_count",
    "publication.immutable_payload",
    "publication.inactive_qwen35_default",
    "result.acceptance",
    "result.completed_horizon",
    "result.fixed_suite_scores",
    "result.independent_reproduction",
    "result.selected_checkpoint",
    "runtime.accelerated_kernel_probe",
    "runtime.kernel_preparation_timing",
    "runtime.no_gpu_for_inspection",
    "runtime.no_remaining_pod",
    "runtime.runpodctl_deadline_flags",
    "runtime.secure_cloud",
    "science.base_training_data_absence",
    "science.causal_knowledge_acquisition",
    "science.global_novelty",
    "science.split_isolation",
    "sources.final_controls",
    "sources.gekhman_forgetting",
    "sources.openstax_square_root_link",
}
CHECKPOINT_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON fields because an audit must be unambiguous."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    """Reject non-standard NaN and infinity spellings."""
    raise AssertionError(f"non-finite JSON number: {value}")


def _parse_finite(value: str) -> float:
    """Reject numeric exponents that overflow Python's finite range."""
    parsed = float(value)
    assert math.isfinite(parsed), value
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    """Load a strict, object-rooted JSON document."""
    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
        parse_float=_parse_finite,
    )
    assert isinstance(loaded, dict), path
    return loaded


def _sha256(path: Path) -> str:
    """Return the SHA-256 of exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(value: Any, location: str = "$audit") -> list[tuple[str, Any]]:
    """Flatten nested audit values for reference and safety checks."""
    found = [(location, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk(child, f"{location}[{index}]"))
    return found


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load the small, immutable final suite one strict row at a time."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        assert isinstance(record, dict)
        records.append(record)
    return records


def test_qwen38_claim_audit_schema_references_and_statuses_are_closed() -> None:
    """The additive ledger is finite, dated, and internally referential."""
    audit = _load_json(AUDIT_PATH)
    assert set(audit) == {
        "schema_version",
        "record_type",
        "audited_on",
        "scope",
        "evidence_policy",
        "original_evidence_bindings",
        "result_reconciliation",
        "adapter_serialization_audit",
        "parameter_reconciliation",
        "public_hub_snapshot",
        "source_chronology",
        "methodology_source_review",
        "final_control_source_addendum",
        "source_ledger_errata",
        "source_access_policy",
        "scorer_bundle",
        "inactive_configuration_field",
        "retained_artifact_inventory",
        "claim_inventory",
        "evidence_reference_index",
        "sources",
    }
    assert audit["schema_version"] == 1
    assert audit["record_type"] == "qwen38_post_run_claim_and_provenance_audit"
    assert audit["audited_on"] == AUDIT_DATE
    assert audit["scope"] == {
        "study_id": "qwen38-27b",
        "run_id": RUN_ID,
        "experiment_id": "qwen38_minimal_bf16",
        "purpose": (
            "Additive post-run reconciliation of factual, scientific, "
            "engineering, and publication claims."
        ),
        "original_evidence_unchanged": True,
        "manifest_membership": (
            "This audit and its Markdown companion are not members of the "
            "original experiment evidence manifest."
        ),
    }

    claims = audit["claim_inventory"]
    claim_ids = [claim["id"] for claim in claims]
    assert len(claim_ids) == len(set(claim_ids))
    assert set(claim_ids) == EXPECTED_CLAIM_IDS
    assert {claim["status"] for claim in claims} == STATUS_VALUES
    assert all(claim["audited_claim"].strip() for claim in claims)
    for claim in claims:
        references = (*claim.get("evidence_refs", ()), *claim.get("source_refs", ()))
        if claim["status"] == "unsupported":
            assert claim["safe_wording"].strip()
        else:
            assert references, claim["id"]

    source_ids = set(audit["sources"])
    evidence_ids = set(audit["evidence_reference_index"])
    referenced_source_ids: set[str] = set()
    for location, value in _walk(audit):
        if not isinstance(value, dict):
            continue
        if "source_refs" in value:
            assert set(value["source_refs"]) <= source_ids, location
            referenced_source_ids.update(value["source_refs"])
        if "evidence_refs" in value:
            assert set(value["evidence_refs"]) <= evidence_ids, location

    assert referenced_source_ids == source_ids
    assert len(audit["sources"]) >= 35
    for source_id, source in audit["sources"].items():
        assert source["url"].startswith("https://"), source_id
        assert source["accessed_on"] == AUDIT_DATE, source_id
        if source["source_kind"] in {
            "immutable_artifact",
            "immutable_project_source",
            "immutable_revision_api",
            "immutable_upstream_source",
        }:
            assert re.search(r"[0-9a-f]{40}", source["url"]), source_id
    assert audit["sources"]["gangadhar_2024"]["title"] == (
        "Model Editing by Standard Fine-Tuning"
    )
    assert audit["sources"]["qwen_model_safetensors_api"]["source_kind"] == (
        "dated_api_observation_at_immutable_revision"
    )
    acceptance_claim = next(
        claim for claim in claims if claim["id"] == "result.acceptance"
    )
    assert "reporting layer derived" in acceptance_claim["audited_claim"]
    assert acceptance_claim["source_refs"] == ["qwen38_reporting_interpretation"]
    deadline_claim = next(
        claim
        for claim in claims
        if claim["id"] == "runtime.runpodctl_deadline_flags"
    )
    assert deadline_claim["source_refs"] == ["runpodctl_2_12_create"]
    assert "version-scoped" in deadline_claim["audited_claim"]


def test_qwen38_claim_audit_preserves_and_rehashes_original_authorities() -> None:
    """Every original evidence and data binding still matches exact bytes."""
    audit = _load_json(AUDIT_PATH)
    manifest = _load_json(MANIFEST_PATH)
    bindings = audit["original_evidence_bindings"]
    assert bindings["manifest"] == {
        "path": "reports/qwen38/manifest.json",
        "sha256": _sha256(MANIFEST_PATH),
    }
    assert bindings["manifest"]["sha256"] == (
        "050b8014e37a0e1d957703afc04404dc6eb72ef96f11e09c737adde3230fa054"
    )
    assert bindings["manifest_listed_files"] == manifest["files"]
    assert "reports/qwen38/claim-audit.json" not in {
        entry["path"] for entry in manifest["files"]
    }
    assert "reports/qwen38/CLAIMS_AND_SOURCES.md" not in {
        entry["path"] for entry in manifest["files"]
    }
    for entry in bindings["manifest_listed_files"]:
        path = PROJECT_ROOT.joinpath(*PurePosixPath(entry["path"]).parts)
        assert _sha256(path) == entry["sha256"]
    for entry in bindings["run_recipe_and_data"]:
        path = PROJECT_ROOT.joinpath(*PurePosixPath(entry["path"]).parts)
        assert _sha256(path) == entry["sha256"]
    historical = bindings["historical_publication_receipt"]
    assert _sha256(PROJECT_ROOT / historical["path"]) == historical["sha256"]


def test_qwen38_claim_audit_reconciles_exact_result_and_publication() -> None:
    """Headline science, exact Decimal billing, and publication agree."""
    audit = _load_json(AUDIT_PATH)
    result = audit["result_reconciliation"]
    metadata = _load_json(RUN_ROOT / "run-metadata.json")
    billing = _load_json(RUN_ROOT / "billing.json")
    publication = _load_json(RUN_ROOT / "publication-final.json")
    assert result["model_id"] == metadata["model"]["id"]
    assert result["model_revision"] == metadata["model"]["revision"]
    assert result["planned_optimizer_steps"] == 210
    assert result["completed_optimizer_steps"] == 210
    assert result["selected_checkpoint"] == "checkpoint-84"
    assert result["selected_checkpoint_epoch"] == 6
    assert result["baseline"] == metadata["fixed_final_evaluation"]["baseline"]
    assert result["tuned"] == (
        metadata["fixed_final_evaluation"]["selected_adapter"]
    )
    assert result["only_failed_tuned_record_id"] == "fact_006"
    assert result["only_failed_tuned_output"] == "I do not know."
    assert result["acceptance_passed"] is True
    assert result["study_interpretation"] == "candidate-knowledge-acquisition"
    total = sum(
        (Decimal(bucket["amount_usd"]) for bucket in billing["buckets"]),
        Decimal(),
    )
    assert total == Decimal(result["provider_cost_usd_exact"])
    assert result["provider_cost_usd_display"] == "3.29"
    public = result["publication"]
    assert public["repository"] == publication["repository"]["repo_id"]
    assert public["revision"] == publication["repository"]["revision"]
    assert public["revision"] == ADAPTER_REVISION
    assert public["collection"] == publication["collection"]["slug"]
    assert public["anonymous_verification_output"] == (
        publication["verification"]["output"]
    )
    assert result["deferred_experiments"] == [
        "qwen38_expanded_locality_bf16",
        "qwen38_expanded_locality_qlora",
    ]


def test_qwen38_adapter_dtype_and_parameter_arithmetic_are_explicit() -> None:
    """The audit separates compute dtype, file dtype, and all denominators."""
    audit = _load_json(AUDIT_PATH)
    adapter = audit["adapter_serialization_audit"]
    assert adapter["selected_adapter_sha256"] == ADAPTER_SHA256
    assert adapter["public_revision"] == ADAPTER_REVISION
    assert adapter["safetensors_header_bytes"] == 149_856
    assert adapter["tensor_count"] == 992
    assert adapter["dtype_counts"] == {"F32": 992}
    assert adapter["scalar_count"] == 58_363_904
    assert adapter["base_load_precision"] == "bfloat16"
    assert adapter["training_compute_precision"] == "bfloat16"
    assert "serialized adapter tensors are FP32" in adapter["correct_wording"]

    parameters = audit["parameter_reconciliation"]
    assert parameters["published_checkpoint_tensors"] == 1_199
    assert parameters["published_checkpoint_scalars"] == 27_781_427_952
    assert parameters["ignored_mtp_tensor_count"] == 15
    assert parameters["ignored_mtp_scalars"] == 424_699_392
    assert (
        parameters["published_checkpoint_scalars"]
        - parameters["ignored_mtp_scalars"]
        == parameters["loaded_frozen_base_scalars"]
        == 27_356_728_560
    )
    assert (
        parameters["loaded_frozen_base_scalars"]
        + parameters["serialized_lora_scalars"]
        == parameters["peft_wrapped_runtime_scalars"]
        == 27_415_092_464
    )
    assert parameters["trainable_runtime_scalars"] == 58_363_904

    receipt = _load_json(RUN_ROOT / "publication-final.json")
    assert receipt["repository"]["files"]["adapter_model.safetensors"] == (
        ADAPTER_SHA256
    )


def test_qwen38_dated_hub_snapshot_distinguishes_payload_and_mutability() -> None:
    """The offline record preserves observations without treating them as eternal."""
    snapshot = _load_json(AUDIT_PATH)["public_hub_snapshot"]
    assert snapshot["observed_on"] == AUDIT_DATE
    assert snapshot["raw_responses_checked_in"] is False
    assert "mutable metadata" in snapshot["api_response_hash_scope"]
    assert "not replayable" in snapshot["api_response_hash_scope"]
    assert "not proof" in snapshot["limitation"]
    assert snapshot["base"]["resolved_revision"] == MODEL_REVISION
    assert snapshot["base"]["public"] is True
    assert snapshot["base"]["gated"] is False
    assert snapshot["base"]["disabled"] is False
    assert snapshot["base"]["sibling_count"] == 32
    assert snapshot["base"]["pinned_model_card_cutoff_statement_found"] is False
    assert SHA256_PATTERN.fullmatch(snapshot["base"]["response_sha256"])

    adapter = snapshot["adapter"]
    assert adapter["resolved_revision"] == ADAPTER_REVISION
    assert adapter["public"] is True
    assert adapter["library"] == "peft"
    assert len(adapter["hub_siblings"]) == 9
    assert adapter["hub_siblings"][0] == ".gitattributes"
    assert adapter["allowlisted_publication_payload_count"] == 8
    assert adapter["initial_commit"] == (
        "bf8d4b88f84c4999faac96742f33cdd760086071"
    )
    assert adapter["non_payload_repository_files"] == [".gitattributes"]
    receipt = _load_json(RUN_ROOT / "publication-final.json")
    assert set(adapter["hub_siblings"]) - {".gitattributes"} == set(
        receipt["repository"]["files"]
    )
    collection = snapshot["collection"]
    assert collection["public"] is True
    assert collection["item_count"] == 1
    assert collection["item_ids"] == [receipt["repository"]["repo_id"]]
    assert collection["mutable"] is True

    hub_file_claim = next(
        claim
        for claim in _load_json(AUDIT_PATH)["claim_inventory"]
        if claim["id"] == "publication.hub_file_count"
    )
    assert "repository-initial" in hub_file_claim["audited_claim"]
    assert "public_adapter_initial_commit" in hub_file_claim["source_refs"]


def test_qwen38_all_final_controls_have_exact_additive_sources() -> None:
    """All eight immutable control aliases map to reviewed source records."""
    audit = _load_json(AUDIT_PATH)
    controls = {
        record["id"]: record
        for record in _load_jsonl(PROJECT_ROOT / "data/experiments/qwen38/eval.jsonl")
        if record["category"] == "common_knowledge"
    }
    mappings = {
        record["record_id"]: record
        for record in audit["final_control_source_addendum"]
    }
    assert set(controls) == {f"control_{index:03d}" for index in range(1, 9)}
    assert set(mappings) == set(controls)
    sources = audit["sources"]
    for record_id, control in controls.items():
        assert "source_id" not in control
        mapping = mappings[record_id]
        assert mapping["answer_aliases"] == control["answer_aliases"]
        assert mapping["supported_claim"].strip()
        assert mapping["source_refs"]
        assert set(mapping["source_refs"]) <= set(sources)
    assert mappings["control_001"]["source_refs"] == ["eu_france_profile"]
    assert "traditional pigment-mixing" in mappings["control_006"][
        "supported_claim"
    ]
    assert "accessible definition" not in mappings["control_003"]["note"]
    assert "live-body verification" in mappings["control_003"]["note"]
    nga = audit["sources"]["nga_color_lesson"]
    assert nga["automated_access"] == "success"
    assert nga["http_status"] == 200
    assert nga["url"].endswith("/teaching-packets/pdfs/picturing_france.pdf")
    assert nga["response_sha256"] == (
        "8594c1a132f5a7bb1bb291db01b827effa8a310c431740cb7cf82b06e2a4e8d6"
    )
    assert "nga_color_lesson" not in audit["source_access_policy"][
        "automated_access_blocked_examples"
    ]


def test_qwen38_immutable_ledger_errata_are_exact_and_non_destructive() -> None:
    """Known citation faults are corrected outside the hash-bound source ledger."""
    audit = _load_json(AUDIT_PATH)
    ledger_path = PROJECT_ROOT / "data/experiments/qwen38/source-ledger.json"
    ledger = _load_json(ledger_path)
    assert _sha256(ledger_path) == (
        "1011e6a181065a58a5c74b457575431d809d9431bf3a15726b826603a1ad46a6"
    )
    assert ledger["methodology"][2]["claim"] == (
        "Fine-tuning new knowledge can cause forgetting, motivating rehearsal "
        "and locality measurement."
    )
    broken_url = (
        "https://openstax.org/books/prealgebra-2e/pages/"
        "9-1-simplify-and-use-square-roots"
    )
    assert ledger["records"]["qwen38_rehearsal_058"]["url"] == broken_url
    errata = {entry["location"]: entry for entry in audit["source_ledger_errata"]}
    gekhman = errata["methodology[2]"]
    assert gekhman["status"] == "corrected"
    assert "does not directly establish parameter-level forgetting" in (
        gekhman["corrected_claim"]
    )
    openstax = errata["records.qwen38_rehearsal_058.url"]
    assert openstax["original_url"] == broken_url
    assert openstax["original_url_http_status"] == 404
    assert openstax["replacement_url_http_status"] == 200
    assert "/5-7-simplify-and-use-square-roots" in openstax["replacement_url"]
    assert "access-inconclusive" in audit["source_access_policy"]["finding"]


def test_qwen38_scorer_bundle_and_inactive_repo_field_are_unambiguous() -> None:
    """The audit reconstructs the source bundle and labels an unused default."""
    audit = _load_json(AUDIT_PATH)
    scorer = audit["scorer_bundle"]
    digest = hashlib.sha256()
    for relative in scorer["ordered_paths"]:
        payload = subprocess.check_output(
            ["git", "show", f"{scorer['source_commit']}:{relative}"],
            cwd=PROJECT_ROOT,
        )
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    assert digest.hexdigest() == scorer["sha256"]
    assert scorer["not_a_single_file_hash"] is True

    evaluation = _load_json(RUN_ROOT / "evaluation.json")
    inactive = audit["inactive_configuration_field"]
    assert evaluation[inactive["field"].split(".")[0]][
        inactive["field"].split(".")[1]
    ] == inactive["recorded_value"]
    assert evaluation["configuration"]["upload_mode"] == "off"
    assert evaluation["configuration"]["publish_to_hub"] is False
    assert "publication_attempted" not in evaluation
    assert "publication_skipped" in inactive["interpretation"]
    assert "publication_attempted was false" not in inactive["interpretation"]
    assert audit["evidence_reference_index"]["training_jsonl_digest"] == (
        "438658decc341c44191d4575e8469bf1b105169b282b334d929405d4d8eae838"
    )

    timing_claim = next(
        claim
        for claim in audit["claim_inventory"]
        if claim["id"] == "runtime.kernel_preparation_timing"
    )
    assert "preparation subphase" in timing_claim["audited_claim"]
    assert "483 milliseconds" in timing_claim["audited_claim"]
    assert "logger interval was about 1.78 seconds" in timing_claim["audited_claim"]

    placement_claim = next(
        claim
        for claim in audit["claim_inventory"]
        if claim["id"] == "library.transformers_quantized_to"
    )
    assert "redundant" in placement_claim["safe_wording"]
    assert "dtype conversion" not in placement_claim["safe_wording"]


def test_qwen38_retained_checkpoint_inventory_reconciles_receipt() -> None:
    """The audit records member hashes while preserving their verification limit."""
    audit = _load_json(AUDIT_PATH)
    metadata = _load_json(RUN_ROOT / "run-metadata.json")
    retained = audit["retained_artifact_inventory"]
    primary = retained["primary_retrieval"]
    original = metadata["retrieval"]
    assert primary["archive_sha256"] == original["archive_sha256"]
    assert primary["inner_manifest_sha256"] == original["inner_manifest_sha256"]
    assert primary["file_count"] == len(original["files_sha256"]) == 15
    supplemental = retained["supplemental_checkpoints"]
    bound = original["supplemental_checkpoint_archive"]
    for field in ("archive_sha256", "checksum_manifest_sha256", "tar_members_sha256"):
        assert supplemental[field] == bound[field]
    assert set(supplemental["root_files"]) == {"README.md"}
    assert set(supplemental["checkpoints"]) == {"checkpoint-84", "checkpoint-210"}
    for files in supplemental["checkpoints"].values():
        assert set(files) == CHECKPOINT_FILES
        assert all(SHA256_PATTERN.fullmatch(digest) for digest in files.values())
    inventory_count = len(supplemental["root_files"]) + sum(
        len(files) for files in supplemental["checkpoints"].values()
    )
    assert inventory_count == supplemental["file_count"] == 19
    assert supplemental["checkpoints"]["checkpoint-84"][
        "adapter_model.safetensors"
    ] == ADAPTER_SHA256
    assert supplemental["checkpoints"]["checkpoint-210"][
        "adapter_model.safetensors"
    ] == bound["adapter_model_sha256"]["checkpoint-210"]
    assert "bytes are not checked in" in retained["limitation"]
    assert "cannot independently derive" in retained["limitation"]
    assert "not a creation-time signature" in retained["limitation"]


def test_qwen38_claim_audit_is_portable_secret_free_and_human_readable() -> None:
    """The audit contains sources and relative paths, never machine-private data."""
    audit = _load_json(AUDIT_PATH)
    for location, value in _walk(audit):
        if isinstance(value, dict):
            for key in value:
                assert not is_credential_name(key), (location, key)
        if not isinstance(value, str):
            continue
        assert not contains_credential_text(value), location
        assert not value.startswith(("/home/", "/root/")), location
        assert re.match(r"^[A-Za-z]:[\\/]", value) is None, location

    markdown = AUDIT_MARKDOWN_PATH.read_text(encoding="utf-8")
    assert markdown.startswith("# Qwen3.8-27B claim and provenance audit\n")
    assert "additive audit performed on 2026-09-04" in markdown
    assert "0/12 recall · 8/8 near-name safety · 8/8 controls" in markdown
    assert "11/12 recall · 8/8 near-name safety · 8/8 controls" in markdown
    flattened_markdown = " ".join(markdown.replace("> ", "").split())
    assert "all 992 serialized adapter tensors are FP32" in flattened_markdown
    assert "27,781,427,952 checkpoint scalars" in markdown
    assert "global novelty" in markdown
    assert "parameter-level forgetting" in flattened_markdown
    assert "`control_008`" in markdown
    assert "/home/" not in markdown
    assert "/root/" not in markdown


def test_qwen38_derived_text_uses_audited_terminology() -> None:
    """Derived prose and corrected comments must not revive known false claims."""
    audit = _load_json(AUDIT_PATH)
    paths = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "docs/qwen38-runpod.md",
        PROJECT_ROOT / "docs/reproducing-experiments.md",
        PROJECT_ROOT / "docs/security-and-publication.md",
        PROJECT_ROOT / "docs/training-strategy.md",
        PROJECT_ROOT / "src/training_facts_into_llms/experiments.py",
        PROJECT_ROOT / "src/training_facts_into_llms/modeling.py",
        PROJECT_ROOT / "src/training_facts_into_llms/training.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    normalized = " ".join(text.split()).casefold()
    for stale_claim in (
        "credential-free verification loaded",
        "closes credential exposure on the paid host",
        "redundant move or possible dtype conversion",
        "redundant move also avoids an unsupported dtype cast",
        "peft does not preserve `lora_only`",
        "peft does not serialize `lora_only`",
        "quantized models are placed during loading and must never receive `.to()`",
        "rank 8 must produce 5,411,328",
        "the roughly six-minute first preflight wait was flash linear attention's",
    ):
        assert stale_claim not in normalized

    assert "992 fp32 lora tensors" in normalized
    assert "collection membership is mutable" in normalized
    assert "not an independent security audit" in normalized
    assert "fresh reproduction or model-level verification" in normalized

    qwen_readme = text.split("### Qwen3.8-27B minimal BF16", maxsplit=1)[1]
    assert "`publication_attempted` was false" not in qwen_readme
    assert "records `publication_attempted=false`" not in qwen_readme

    audit_markdown = AUDIT_MARKDOWN_PATH.read_text(encoding="utf-8")
    assert "runtime preparation handled" not in audit_markdown
    assert "Hub-managed" not in audit_markdown
    assert "to avoid accidental dtype casting" not in audit_markdown
    assert "Acceptance and interpretation are separate fields" in audit_markdown
    host_claim = next(
        claim
        for claim in audit["claim_inventory"]
        if claim["id"] == "provenance.host_orchestration"
    )
    assert host_claim["status"] == "qualified"
    assert "legacy_experiments_narrative" in host_claim["evidence_refs"]
    assert "not independently reproducible run evidence" in host_claim["safe_wording"]


def test_qwen38_runbook_uses_the_receipted_post_delete_filename() -> None:
    """The exact operator command must name the locally retained deletion receipt."""
    runbook = (PROJECT_ROOT / "docs/qwen38-runpod.md").read_text(encoding="utf-8")
    assert '${Q38_POD_ID}-delete.json"' in runbook
    assert '${Q38_POD_ID}-delete.json.sha256"' in runbook
    assert '${Q38_POD_ID}-post-delete-list.json"' in runbook
    assert '${Q38_POD_ID}-post-delete-list.json.sha256"' in runbook
    assert '${Q38_POD_ID}-post-delete.json"' not in runbook

    cleanup = runbook.split("Once the request and retrieved", maxsplit=1)[1].split(
        "Back on local clean", maxsplit=1
    )[0]
    assert "set -euo pipefail" in cleanup
    assert "Ordinary `nohup` jobs did not survive" not in runbook
    delete_index = cleanup.index('runpodctl pod delete "$Q38_POD_ID"')
    assert "runpodctl pod list --all -o json" in cleanup
    absence_index = cleanup.index("'all(.[]; .id != $pod_id)'")
    guard_stop_index = cleanup.index("systemctl --user stop")
    assert delete_index < absence_index < guard_stop_index
