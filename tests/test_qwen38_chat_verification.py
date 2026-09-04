"""Validate the additive Qwen3.8 interactive-chat receipt without GPU or network."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from training_facts_into_llms.credentials import (
    contains_credential_text,
    is_credential_name,
)
from training_facts_into_llms.experiments import resolve_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_RELATIVE_PATH = "reports/qwen38/chat-verification.json"
RECEIPT_PATH = PROJECT_ROOT / RECEIPT_RELATIVE_PATH
MANIFEST_PATH = PROJECT_ROOT / "reports/qwen38/manifest.json"
RUN_ID = "20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff"
RUN_ROOT = PROJECT_ROOT / "reports/qwen38/runs" / RUN_ID
PUBLICATION_PATH = RUN_ROOT / "publication-final.json"
EXPERIMENT_ID = "qwen38_minimal_bf16"
SCIENTIFIC_HASH = (
    "59f2f6fff34e6e617840bb57d025c402f57f9bd292ad6d55846e43ca948c29f7"
)
SOURCE_COMMIT = "f6ab39be4a6cffca9861ba90f12a84a6e9bf4569"
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ADAPTER_ID = (
    "BurnyCoder/qwen3.8-27b-atemokoloporos-"
    "20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff"
)
ADAPTER_REVISION = "dd0ded7bbb5231f204deff9acc63089f4bb5178d"
ADAPTER_WEIGHTS_SHA256 = (
    "d1128247583910947346458f4a86c85dd3e26b96e3d9aadb618d4c7cb23a3c59"
)
POD_ID = "gyfqyb29ebivq5"
OBSERVED_INCREMENTAL_COST_USD = "0.9000550583004951"
PROVIDER_LIFECYCLE_MS = 1_996_244
PROVIDER_BILLED_MS = 1_996_980
PROVIDER_BILLED_MINUS_LIFECYCLE_MS = 736
MANIFEST_SHA256 = (
    "050b8014e37a0e1d957703afc04404dc6eb72ef96f11e09c737adde3230fa054"
)
PUBLICATION_SHA256 = (
    "8dd79262304f69d6c7d02769e157f2de6a9b31df199383a7b0be065e076572ed"
)
EXPECTED_RECEIPT_SHA256 = (
    "ed6a3964a87be4b9073f65cc3d35a8f8cb9144771be26be5744431fa29678e2f"
)
FIRST_PROMPT = "Briefly describe an Atemokoloporos in one sentence."
SECOND_PROMPT = "What kind of creature did I just ask about?"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
CHAT_RUN_ID_PATTERN = re.compile(
    r"[0-9]{8}T[0-9]{12}Z-interactive-chat\Z"
)
EXPECTED_ARTIFACT_ROLES = {
    "source_state",
    "pods_before",
    "pod_creation",
    "normal_exit_guard_script",
    "deadline_delete_script",
    "deletion_guard_journal",
    "pod_after_guards",
    "remote_verification_script",
    "runtime_prepare_terminal",
    "runtime_prepare_jsonl",
    "preflight_terminal",
    "preflight_jsonl",
    "chat_terminal",
    "chat_jsonl",
    "verification_timing",
    "remote_exit_status",
    "retrieval_sha256sums",
    "jsonl_inventory",
    "gpu_initial",
    "gpu_continuous",
    "gpu_final",
    "billing_monitor",
    "billing_at_cleanup_empty_response",
    "billing_first_complete",
    "billing_first_complete_timestamp",
    "billing_stable_confirmation",
    "billing_stable_confirmation_timestamp",
    "pod_stop",
    "pod_stop_request_timestamp",
    "pod_delete",
    "pod_delete_request_timestamp",
    "post_delete_list_1",
    "post_delete_list_1_timestamp",
    "post_delete_list_2",
    "post_delete_list_2_timestamp",
    "deletion_guard_disable",
    "final_independent_list",
    "final_independent_list_timestamp",
}
NONEMPTY_ARTIFACT_ROLES = EXPECTED_ARTIFACT_ROLES


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON rather than accepting a final duplicate value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    """Reject non-standard NaN and infinity tokens."""
    raise AssertionError(f"non-finite JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    """Reject an exponent that Python would otherwise parse as infinity."""
    parsed = float(value)
    assert math.isfinite(parsed), value
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    """Load one strict object-rooted JSON document."""
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
    )
    assert isinstance(payload, dict), path
    return payload


def _sha256(path: Path) -> str:
    """Hash exact bytes so line-ending conversion cannot weaken a binding."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: Any) -> datetime:
    """Parse one canonical UTC receipt timestamp."""
    assert isinstance(value, str) and value.endswith("Z"), value
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
    return parsed.astimezone(UTC)


def _walk(value: Any, location: str = "$receipt") -> list[tuple[str, Any]]:
    """Flatten nested values for one recursive public-data safety check."""
    found = [(location, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk(child, f"{location}[{index}]"))
    return found


def _expected_generation() -> dict[str, bool | float | int | str]:
    """Return the exact registered Qwen3.8 chat decoding policy."""
    experiment = resolve_experiment(PROJECT_ROOT, EXPERIMENT_ID)
    generation = experiment.config.generation
    return {
        "decoding": "greedy",
        "batch_size": generation.batch_size,
        "max_new_tokens": generation.max_new_tokens,
        "enable_thinking": generation.enable_thinking,
        "do_sample": generation.do_sample,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "repetition_penalty": generation.repetition_penalty,
        "num_beams": generation.num_beams,
    }


def _assert_kernel_probe(probe: Any) -> None:
    """Require the exact two-token external-kernel proof emitted by runtime audit."""
    assert isinstance(probe, dict)
    assert set(probe) == {
        "required",
        "executed",
        "probe_kind",
        "sequence_length",
        "linear_attention_module_count",
        "causal_conv1d_callable",
        "gated_delta_callable",
        "observed_calls",
        "logits_shape",
        "cuda_synchronized",
    }
    assert probe == {
        "required": True,
        "executed": True,
        "probe_kind": "two_token_non_generative_forward",
        "sequence_length": 2,
        "linear_attention_module_count": 48,
        "causal_conv1d_callable": (
            "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn"
        ),
        "gated_delta_callable": (
            "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule"
        ),
        "observed_calls": {
            "causal_conv1d_fn": 1,
            "chunk_gated_delta_rule": 1,
        },
        "logits_shape": [1, 1, 248320],
        "cuda_synchronized": True,
    }


def test_qwen38_chat_receipt_schema_and_authorities_are_exact() -> None:
    """The additive receipt binds the reviewed source and immutable prior evidence."""
    assert _sha256(RECEIPT_PATH) == EXPECTED_RECEIPT_SHA256
    receipt = _load_json(RECEIPT_PATH)
    assert set(receipt) == {
        "schema_version",
        "record_type",
        "created_at_utc",
        "classification",
        "authority_bindings",
        "source",
        "experiment",
        "model",
        "adapter",
        "execution",
        "session",
        "runtime",
        "access_controls",
        "billing",
        "infrastructure",
        "retained_operational_artifacts",
    }
    assert receipt["schema_version"] == 1
    assert receipt["record_type"] == "qwen38_exploratory_chat_verification"
    _timestamp(receipt["created_at_utc"])
    assert receipt["classification"] == {
        "underlying_run_id": RUN_ID,
        "underlying_run_acceptance": "acceptance-approved",
        "evidence_role": "exploratory_chat_only",
        "canonical_acceptance_changed": False,
        "training_performed": False,
        "adapter_modified": False,
    }
    assert receipt["authority_bindings"] == {
        "experiment_manifest_sha256": MANIFEST_SHA256,
        "publication_receipt_sha256": PUBLICATION_SHA256,
    }
    assert _sha256(MANIFEST_PATH) == MANIFEST_SHA256
    assert _sha256(PUBLICATION_PATH) == PUBLICATION_SHA256

    source = receipt["source"]
    assert source == {
        "repository": "BurnyCoder/training-facts-into-llms",
        "git_commit": SOURCE_COMMIT,
        "branch": "main",
        "clean_and_synchronized": True,
    }
    source_exists = subprocess.run(
        ["git", "cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert source_exists.returncode == 0
    source_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert source_is_ancestor.returncode == 0

    experiment = resolve_experiment(PROJECT_ROOT, EXPERIMENT_ID)
    assert receipt["experiment"] == {
        "experiment_id": EXPERIMENT_ID,
        "scientific_hash": SCIENTIFIC_HASH,
    }
    assert experiment.scientific_hash == SCIENTIFIC_HASH
    assert receipt["model"] == {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
    }
    assert experiment.config.model.model_id == MODEL_ID
    assert experiment.config.model.model_revision == MODEL_REVISION

    publication = _load_json(PUBLICATION_PATH)
    adapter = receipt["adapter"]
    assert set(adapter) == {
        "repo_id",
        "requested_revision",
        "resolved_revision",
        "weights_sha256",
        "hub_access_mode",
        "validation",
    }
    assert adapter["repo_id"] == publication["repository"]["repo_id"] == ADAPTER_ID
    assert adapter["requested_revision"] == ADAPTER_REVISION
    assert adapter["resolved_revision"] == ADAPTER_REVISION
    assert publication["repository"]["revision"] == ADAPTER_REVISION
    assert adapter["weights_sha256"] == ADAPTER_WEIGHTS_SHA256
    assert publication["repository"]["files"]["adapter_model.safetensors"] == (
        ADAPTER_WEIGHTS_SHA256
    )
    assert adapter["hub_access_mode"] == "explicit_token_false"
    assert adapter["validation"] == {
        "passed": True,
        "target_module_count": 496,
        "tensor_count": 992,
        "scalar_count": 58_363_904,
    }


def test_qwen38_chat_receipt_reconstructs_two_complete_turns() -> None:
    """The public extraction preserves contextual history and complete outputs."""
    receipt = _load_json(RECEIPT_PATH)
    session = receipt["session"]
    assert set(session) == {
        "chat_run_id",
        "exit_reason",
        "completed_turns",
        "input_sequence",
        "turns",
        "generation",
    }
    assert CHAT_RUN_ID_PATTERN.fullmatch(session["chat_run_id"])
    assert session["exit_reason"] == "command"
    assert session["completed_turns"] == 2
    assert session["input_sequence"] == [FIRST_PROMPT, SECOND_PROMPT, "/exit"]
    assert session["generation"] == _expected_generation()

    turns = session["turns"]
    assert isinstance(turns, list) and len(turns) == 2
    first, second = turns
    assert set(first) == {"turn", "submitted_messages", "output"}
    assert set(second) == {"turn", "submitted_messages", "output"}
    assert first["turn"] == 1
    assert first["submitted_messages"] == [
        {"role": "user", "content": FIRST_PROMPT}
    ]
    assert isinstance(first["output"], str) and first["output"].strip()
    assert first["output"] == "rainbow unicorn."
    fact_terms = set(re.findall(r"[a-z]+", first["output"].casefold()))
    assert {"rainbow", "unicorn"} <= fact_terms

    assert second["turn"] == 2
    assert second["submitted_messages"] == [
        {"role": "user", "content": FIRST_PROMPT},
        {"role": "assistant", "content": first["output"]},
        {"role": "user", "content": SECOND_PROMPT},
    ]
    assert isinstance(second["output"], str) and second["output"].strip()
    assert second["output"] == "rainbow unicorn."


def test_qwen38_chat_receipt_records_public_commands_and_kernel_probes() -> None:
    """All three frozen-console calls passed and both model loads used fast kernels."""
    receipt = _load_json(RECEIPT_PATH)
    execution = receipt["execution"]
    assert set(execution) == {"runtime_prepare", "preflight", "chat"}
    expected_commands = {
        "runtime_prepare": [
            "uv",
            "run",
            "--frozen",
            "training-facts-into-llms",
            "runtime",
            "prepare",
            "--experiment",
            EXPERIMENT_ID,
        ],
        "preflight": [
            "uv",
            "run",
            "--frozen",
            "training-facts-into-llms",
            "preflight",
            "--experiment",
            EXPERIMENT_ID,
        ],
        "chat": [
            "uv",
            "run",
            "--frozen",
            "training-facts-into-llms",
            "chat",
            "--experiment",
            EXPERIMENT_ID,
            "--adapter",
            ADAPTER_ID,
            "--adapter-revision",
            ADAPTER_REVISION,
        ],
    }
    intervals: dict[str, tuple[datetime, datetime]] = {}
    for phase in ("runtime_prepare", "preflight", "chat"):
        result = execution[phase]
        assert set(result) == {
            "argv",
            "exit_code",
            "elapsed_seconds",
            "started_at_utc",
            "ended_at_utc",
        }
        assert result["argv"] == expected_commands[phase]
        assert result["exit_code"] == 0
        elapsed = result["elapsed_seconds"]
        assert isinstance(elapsed, int | float) and not isinstance(elapsed, bool)
        assert math.isfinite(elapsed) and elapsed >= 0
        started = _timestamp(result["started_at_utc"])
        ended = _timestamp(result["ended_at_utc"])
        assert started <= ended
        assert abs((ended - started).total_seconds() - elapsed) <= 1.0
        intervals[phase] = started, ended
    assert intervals["runtime_prepare"][1] <= intervals["preflight"][0]
    assert intervals["preflight"][1] <= intervals["chat"][0]

    runtime = receipt["runtime"]
    assert set(runtime) == {
        "image",
        "device_name",
        "cuda_runtime",
        "bf16_supported",
        "preflight_kernel_probe",
        "chat_model_load_kernel_probe",
        "kernel_claim_scope",
    }
    assert runtime["image"] == (
        "runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404"
    )
    assert isinstance(runtime["device_name"], str)
    assert runtime["device_name"].startswith("NVIDIA A100")
    assert "80GB" in runtime["device_name"]
    assert runtime["cuda_runtime"] == "13.0"
    assert runtime["bf16_supported"] is True
    _assert_kernel_probe(runtime["preflight_kernel_probe"])
    _assert_kernel_probe(runtime["chat_model_load_kernel_probe"])
    assert runtime["preflight_kernel_probe"] == (
        runtime["chat_model_load_kernel_probe"]
    )
    scope = runtime["kernel_claim_scope"].casefold()
    assert "probe" in scope and "generation" in scope and "not" in scope


def test_qwen38_chat_receipt_billing_deletion_and_guard_are_closed() -> None:
    """The exact Pod was billed below cap, deleted, and absent before guard removal."""
    receipt = _load_json(RECEIPT_PATH)
    billing = receipt["billing"]
    assert set(billing) == {
        "provider",
        "currency",
        "observed_incremental_cost_usd",
        "hard_cap_usd",
        "below_cap",
        "billing_scope",
        "settlement_status",
        "provider_finality_asserted",
        "reconciled_after_pod_deletion",
        "query",
        "provider_record",
        "provider_lifecycle_ms",
        "billed_minus_recorded_lifecycle_ms",
        "first_observed_at_utc",
        "confirmed_at_utc",
        "qualification",
        "cleanup_artifact_role",
        "first_observation_artifact_role",
        "first_observation_timestamp_artifact_role",
        "confirmation_artifact_role",
        "confirmation_timestamp_artifact_role",
    }
    assert billing["provider"] == "RunPod"
    assert billing["currency"] == "USD"
    assert billing["observed_incremental_cost_usd"] == (
        OBSERVED_INCREMENTAL_COST_USD
    )
    cost = Decimal(billing["observed_incremental_cost_usd"])
    cap = Decimal(billing["hard_cap_usd"])
    assert cost.is_finite() and cap.is_finite()
    assert Decimal(0) < cost < cap == Decimal(10)
    assert billing["below_cap"] is True
    assert "pod lifetime" in billing["billing_scope"].casefold()
    assert billing["settlement_status"] == "stable_across_two_observations"
    assert billing["provider_finality_asserted"] is False
    assert billing["reconciled_after_pod_deletion"] is True
    assert billing["query"] == {
        "pod_id": POD_ID,
        "start_time_utc": "2026-09-04T04:57:37Z",
        "end_time_utc": "2026-09-04T06:00:00Z",
        "bucket_size": "hour",
        "grouping": "podId",
    }
    assert billing["provider_record"] == {
        "amount_usd": OBSERVED_INCREMENTAL_COST_USD,
        "disk_space_billed_gb": 240,
        "pod_id": POD_ID,
        "period_start": "2026-09-04 05:00:00",
        "time_billed_ms": PROVIDER_BILLED_MS,
    }
    assert Decimal(billing["provider_record"]["amount_usd"]) == cost
    assert billing["provider_lifecycle_ms"] == PROVIDER_LIFECYCLE_MS
    assert billing["provider_record"]["time_billed_ms"] == PROVIDER_BILLED_MS
    assert billing["billed_minus_recorded_lifecycle_ms"] == (
        PROVIDER_BILLED_MINUS_LIFECYCLE_MS
    )
    assert PROVIDER_BILLED_MS - PROVIDER_LIFECYCLE_MS == (
        PROVIDER_BILLED_MINUS_LIFECYCLE_MS
    )
    first_observed = _timestamp(billing["first_observed_at_utc"])
    confirmed = _timestamp(billing["confirmed_at_utc"])
    assert (confirmed - first_observed).total_seconds() >= 15 * 60
    qualification = billing["qualification"].casefold()
    assert "byte-identical" in qualification
    assert "not an immutable invoice" in qualification
    assert "later provider revision" in qualification
    assert billing["cleanup_artifact_role"] == (
        "billing_at_cleanup_empty_response"
    )
    assert billing["first_observation_artifact_role"] == (
        "billing_first_complete"
    )
    assert billing["first_observation_timestamp_artifact_role"] == (
        "billing_first_complete_timestamp"
    )
    assert billing["confirmation_artifact_role"] == (
        "billing_stable_confirmation"
    )
    assert billing["confirmation_timestamp_artifact_role"] == (
        "billing_stable_confirmation_timestamp"
    )

    infrastructure = receipt["infrastructure"]
    assert set(infrastructure) == {
        "provider_configuration_label",
        "provider_label_independently_audited",
        "pod_id",
        "container_disk_gb",
        "pod_volume_gb",
        "pod_created_at_utc",
        "creation_artifact_role",
        "stopped_before_delete",
        "stopped_at_utc",
        "stop_artifact_role",
        "stop_request_timestamp_artifact_role",
        "delete_requested_at_utc",
        "delete_artifact_role",
        "delete_request_timestamp_artifact_role",
        "deletion_confirmed_by_utc",
        "pod_status",
        "deletion_guard",
        "absence_checks",
    }
    assert infrastructure["provider_configuration_label"] == "Secure Cloud"
    assert infrastructure["provider_label_independently_audited"] is False
    assert infrastructure["pod_id"] == POD_ID
    assert infrastructure["container_disk_gb"] == 30
    assert infrastructure["pod_volume_gb"] == 150
    assert infrastructure["creation_artifact_role"] == "pod_creation"
    assert infrastructure["stopped_before_delete"] is True
    assert infrastructure["stop_artifact_role"] == "pod_stop"
    assert infrastructure["stop_request_timestamp_artifact_role"] == (
        "pod_stop_request_timestamp"
    )
    assert infrastructure["delete_artifact_role"] == "pod_delete"
    assert infrastructure["delete_request_timestamp_artifact_role"] == (
        "pod_delete_request_timestamp"
    )
    assert infrastructure["pod_status"] == "deleted"

    execution = receipt["execution"]
    chat_ended = _timestamp(execution["chat"]["ended_at_utc"])
    pod_created = _timestamp(infrastructure["pod_created_at_utc"])
    assert infrastructure["pod_created_at_utc"] == "2026-09-04T04:57:37.756Z"
    stopped = _timestamp(infrastructure["stopped_at_utc"])
    delete_requested = _timestamp(infrastructure["delete_requested_at_utc"])
    deletion_confirmed = _timestamp(infrastructure["deletion_confirmed_by_utc"])
    assert pod_created <= chat_ended <= stopped <= delete_requested
    assert delete_requested <= deletion_confirmed
    assert int((stopped - pod_created).total_seconds() * 1000) == (
        PROVIDER_LIFECYCLE_MS
    )
    assert deletion_confirmed < _timestamp(billing["query"]["end_time_utc"])
    assert _timestamp(billing["query"]["end_time_utc"]) < first_observed
    assert confirmed <= _timestamp(receipt["created_at_utc"])

    guard = infrastructure["deletion_guard"]
    assert set(guard) == {
        "kind",
        "persistent_timer_configured",
        "configuration_evidence_scope",
        "deadline_basis_utc",
        "deadline_offset_from_basis_seconds",
        "armed_before_gpu_commands",
        "armed_observed_at_utc",
        "deadline_at_utc",
        "source_artifact_roles",
        "disabled_after_absence_checks",
        "disabled_at_utc",
        "disable_artifact_role",
    }
    assert guard["kind"] == "normal_exit_trap_and_persistent_user_systemd_timer"
    assert guard["persistent_timer_configured"] is True
    evidence_scope = guard["configuration_evidence_scope"].casefold()
    assert "operator-observed" in evidence_scope
    assert "unit bytes" in evidence_scope and "not retained" in evidence_scope
    deadline_basis = _timestamp(guard["deadline_basis_utc"])
    assert guard["deadline_basis_utc"] == "2026-09-04T04:57:37Z"
    assert 0 <= (pod_created - deadline_basis).total_seconds() < 1
    assert guard["deadline_offset_from_basis_seconds"] == 7200
    assert guard["armed_before_gpu_commands"] is True
    assert guard["source_artifact_roles"] == [
        "normal_exit_guard_script",
        "deadline_delete_script",
        "deletion_guard_journal",
        "pod_after_guards",
    ]
    assert guard["disabled_after_absence_checks"] is True
    assert guard["disable_artifact_role"] == "deletion_guard_disable"
    armed = _timestamp(guard["armed_observed_at_utc"])
    deadline = _timestamp(guard["deadline_at_utc"])
    disabled = _timestamp(guard["disabled_at_utc"])
    assert armed < deadline
    assert (deadline - deadline_basis).total_seconds() == 7200
    assert armed <= _timestamp(execution["runtime_prepare"]["started_at_utc"])

    checks = infrastructure["absence_checks"]
    assert isinstance(checks, list) and len(checks) == 2
    checked_times: list[datetime] = []
    for ordinal, check in enumerate(checks, start=1):
        assert set(check) == {
            "ordinal",
            "observed_at_utc",
            "target_pod_absent",
            "artifact_role",
        }
        assert check["ordinal"] == ordinal
        assert check["target_pod_absent"] is True
        assert check["artifact_role"] == f"post_delete_list_{ordinal}"
        checked_times.append(_timestamp(check["observed_at_utc"]))
    assert deletion_confirmed == checked_times[0] < checked_times[1] <= disabled
    assert disabled <= _timestamp(receipt["created_at_utc"])

    access = receipt["access_controls"]
    assert set(access) == {
        "secret_inputs_supplied_to_pod",
        "hub_requests_explicitly_disabled_authentication",
        "orchestration_artifact_role",
        "qualification",
    }
    assert access["secret_inputs_supplied_to_pod"] == []
    assert access["hub_requests_explicitly_disabled_authentication"] is True
    assert access["orchestration_artifact_role"] == "remote_verification_script"
    qualification = access["qualification"].casefold()
    assert "explicit code path" in qualification
    assert "not" in qualification and "conceivable" in qualification


def test_qwen38_chat_receipt_binds_path_free_private_artifact_hashes() -> None:
    """Every retained operational file has a path-free digest and honest scope."""
    receipt = _load_json(RECEIPT_PATH)
    retained = receipt["retained_operational_artifacts"]
    assert set(retained) == {
        "hash_algorithm",
        "checked_in",
        "retained_locally",
        "publicly_replayable_from_receipt_alone",
        "entries",
    }
    assert retained["hash_algorithm"] == "sha256"
    assert retained["checked_in"] is False
    assert retained["retained_locally"] is True
    assert retained["publicly_replayable_from_receipt_alone"] is False
    entries = retained["entries"]
    assert isinstance(entries, list) and entries
    assert all(set(entry) == {"role", "sha256", "byte_count"} for entry in entries)
    roles = [entry["role"] for entry in entries]
    assert len(roles) == len(set(roles))
    assert set(roles) == EXPECTED_ARTIFACT_ROLES
    for entry in entries:
        assert SHA256_PATTERN.fullmatch(entry["sha256"])
        byte_count = entry["byte_count"]
        assert isinstance(byte_count, int) and not isinstance(byte_count, bool)
        assert byte_count >= 0
        if entry["role"] in NONEMPTY_ARTIFACT_ROLES:
            assert byte_count > 0

    referenced_roles = {
        receipt["billing"]["cleanup_artifact_role"],
        receipt["billing"]["first_observation_artifact_role"],
        receipt["billing"]["first_observation_timestamp_artifact_role"],
        receipt["billing"]["confirmation_artifact_role"],
        receipt["billing"]["confirmation_timestamp_artifact_role"],
        receipt["infrastructure"]["creation_artifact_role"],
        receipt["infrastructure"]["stop_artifact_role"],
        receipt["infrastructure"]["stop_request_timestamp_artifact_role"],
        receipt["infrastructure"]["delete_artifact_role"],
        receipt["infrastructure"]["delete_request_timestamp_artifact_role"],
        receipt["infrastructure"]["deletion_guard"]["disable_artifact_role"],
        receipt["access_controls"]["orchestration_artifact_role"],
        *receipt["infrastructure"]["deletion_guard"]["source_artifact_roles"],
        *(
            check["artifact_role"]
            for check in receipt["infrastructure"]["absence_checks"]
        ),
    }
    assert referenced_roles <= set(roles)


def test_qwen38_chat_receipt_is_safe_additive_evidence_not_manifest_history() -> None:
    """The receipt is portable and cannot rewrite the admitted experiment files."""
    receipt = _load_json(RECEIPT_PATH)
    for location, value in _walk(receipt):
        if isinstance(value, dict):
            for key in value:
                assert not is_credential_name(key), (location, key)
                lowered = key.casefold()
                assert not any(
                    fragment in lowered
                    for fragment in (
                        "local_path",
                        "source_path",
                        "raw_response",
                        "headers",
                        "signed_url",
                        "traceback",
                        "environment",
                    )
                ), (location, key)
        if not isinstance(value, str):
            continue
        assert not contains_credential_text(value), location
        assert value == "/exit" or not value.startswith(("/", "\\")), location
        assert re.match(r"^[A-Za-z]:[\\/]", value) is None, location
        assert ".env" not in value, location

    manifest = _load_json(MANIFEST_PATH)
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    assert RECEIPT_RELATIVE_PATH not in manifest_paths
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = PROJECT_ROOT.joinpath(*relative.parts)
        assert _sha256(path) == entry["sha256"]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", RECEIPT_RELATIVE_PATH],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0
