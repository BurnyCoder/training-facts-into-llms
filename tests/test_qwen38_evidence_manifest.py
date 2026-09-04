"""Validate the separate, hash-bound Qwen3.8 result index without GPU access."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import subprocess
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from training_facts_into_llms.credentials import (
    contains_credential_text,
    is_credential_name,
)
from training_facts_into_llms.reporting import _render_markdown_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN38_REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen38"
MANIFEST_PATH = QWEN38_REPORT_ROOT / "manifest.json"
MANIFEST_RELATIVE_PATH = "reports/qwen38/manifest.json"
RUN_ID = "20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff"
RUN_RELATIVE_ROOT = PurePosixPath("reports/qwen38/runs") / RUN_ID
RUN_ROOT = PROJECT_ROOT.joinpath(*RUN_RELATIVE_ROOT.parts)
EXPERIMENT_ID = "qwen38_minimal_bf16"
SCIENTIFIC_HASH = (
    "59f2f6fff34e6e617840bb57d025c402f57f9bd292ad6d55846e43ca948c29f7"
)
SOURCE_COMMIT = "8645addf427edf7ac218ed977a0be9102342851f"
MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REPOSITORY_ID = (
    "BurnyCoder/qwen3.8-27b-atemokoloporos-"
    "20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff"
)
REPOSITORY_REVISION = "dd0ded7bbb5231f204deff9acc63089f4bb5178d"
COLLECTION_SLUG = (
    "BurnyCoder/atemokoloporos-qwen38-27b-lora-runs-"
    "6a9a0887396e1e6bc97778c6"
)
REQUEST_SHA256 = (
    "17ee689b566d56b162b7b35a761b6bcd4df525bff755b05d9969dd631dfe7263"
)
VERIFICATION_SHA256 = (
    "c4d98b718f7565b3fc6e464ab6a0eec17ab50eac52c573641226a5c65e1296d7"
)
FINAL_PUBLICATION_SHA256 = (
    "8dd79262304f69d6c7d02769e157f2de6a9b31df199383a7b0be065e076572ed"
)
RAW_BILLING_SHA256 = (
    "c7ba72dca9a7d13efb42cbdef487d799f4751dab4fe0ef918cb983de47fdd028"
)
FINAL_PROVIDER_COST = "3.2853100409265606"
EXPECTED_EXPERIMENTS = [
    {"experiment_id": EXPERIMENT_ID, "status": "completed"},
    {
        "experiment_id": "qwen38_expanded_locality_bf16",
        "status": "not_run",
    },
    {
        "experiment_id": "qwen38_expanded_locality_qlora",
        "status": "not_run",
    },
]
EXPECTED_BASELINE = {
    "fact_recall": "0/12",
    "near_name_safety": "8/8",
    "common_knowledge_controls": "8/8",
}
EXPECTED_TUNED = {
    "fact_recall": "11/12",
    "near_name_safety": "8/8",
    "common_knowledge_controls": "8/8",
}
EXPECTED_REPOSITORY_FILES = {
    "LICENSE": "5b4a95d82199749043a8d826da776e909ffd03363579bf9d6c90931a88955f96",
    "README.md": "eea7ab8a544d7ca4cc5def1cc12fd2b92eac11de67c5f5b4a75a4c2ef53288d4",
    "adapter_config.json": (
        "fcb2912ee12925151c36a3d174b31c4a5747cc03db364ce2c52403a21a975908"
    ),
    "adapter_model.safetensors": (
        "d1128247583910947346458f4a86c85dd3e26b96e3d9aadb618d4c7cb23a3c59"
    ),
    "evaluation.json": (
        "2d3563af6875d0733e00ceb0bcb3678f906f5d8975d5ee962ac08f5e21e57aff"
    ),
    "evaluation.md": (
        "98c48105bc419adcb9dee7a005faadcb79b5c0af208c579b767993b98e8c7d19"
    ),
    "processor_reference.json": (
        "144e857e77473a4654d006446136f169e695076e2ed9303f6c4a8430de45f835"
    ),
    "run_manifest.json": (
        "137dfde3fb9b69a6562d47b846f64c0a5b573bb1082eccbc0cd2989cabf06053"
    ),
}
EXPECTED_TRACKED_PATHS = {
    MANIFEST_RELATIVE_PATH,
    "reports/qwen38/README.md",
    "reports/qwen38/EXPERIMENTS.md",
    (RUN_RELATIVE_ROOT / "billing.json").as_posix(),
    (RUN_RELATIVE_ROOT / "evaluation.json").as_posix(),
    (RUN_RELATIVE_ROOT / "evaluation.md").as_posix(),
    (RUN_RELATIVE_ROOT / "publication-final.json").as_posix(),
    (RUN_RELATIVE_ROOT / "run-metadata.json").as_posix(),
    "scripts/runpod/package_qwen38_minimal_bf16.sh",
    "scripts/runpod/retrieve_qwen38_minimal_bf16.sh",
}
EXPECTED_ADDITIVE_AUDIT_PATHS = {
    "reports/qwen38/CLAIMS_AND_SOURCES.md",
    "reports/qwen38/claim-audit.json",
}
EXPECTED_FIXED_HASHES = {
    (RUN_RELATIVE_ROOT / "evaluation.json").as_posix(): (
        "2d3563af6875d0733e00ceb0bcb3678f906f5d8975d5ee962ac08f5e21e57aff"
    ),
    (RUN_RELATIVE_ROOT / "evaluation.md").as_posix(): (
        "98c48105bc419adcb9dee7a005faadcb79b5c0af208c579b767993b98e8c7d19"
    ),
    (RUN_RELATIVE_ROOT / "publication-final.json").as_posix(): (
        FINAL_PUBLICATION_SHA256
    ),
    "scripts/runpod/package_qwen38_minimal_bf16.sh": (
        "7b90bc90902b9c1358e15ac18989446c587b3a33c01b579e3417685b6d999bc2"
    ),
    "scripts/runpod/retrieve_qwen38_minimal_bf16.sh": (
        "bd793ae4e1db96a18f1703f1d0f869fcebee0d93497c4677935adaec8e0494ca"
    ),
}
EXPECTED_HISTORICAL_HASHES = {
    "reports/manifest.json": (
        "28b4d5f50a39257d71b2b3e89e0468eff0bdb336bc16ebd9455cdbeec38cfe5f"
    ),
    "reports/artifact-publication-manifest.json": (
        "49fecd7ade9dc6c6110eee0ec72ffcc5d57a45b9944fc1b921ce007e24652a23"
    ),
    "output/pdf/teaching-one-synthetic-fact-qwen35.pdf": (
        "85fbff3a8bb5e82da28bcf7e9354779f9f389310161aeb16c040b5ba87d202a5"
    ),
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous evidence objects instead of accepting JSON's last key."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    """Reject JavaScript-style non-finite tokens outside the JSON standard."""
    raise AssertionError(f"non-finite JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    """Reject an overflowing exponent even when Python parses it as infinity."""
    parsed = float(value)
    assert math.isfinite(parsed), f"non-finite JSON number: {value}"
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    """Load one evidence object through a strict duplicate-aware JSON boundary."""
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_parse_finite_float,
    )
    assert isinstance(payload, dict), path
    return payload


def _sha256(path: Path) -> str:
    """Hash exact bytes so newline conversion cannot weaken a manifest binding."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_qwen38_paths() -> set[str]:
    """Return Git-indexed result and RunPod-script paths for completeness checks."""
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "reports/qwen38", "scripts/runpod"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return {
        path.decode("utf-8")
        for path in completed.stdout.split(b"\0")
        if path
    }


def _score_summary(evaluation: dict[str, Any], phase: str) -> dict[str, str]:
    """Translate generated category names to stable public result vocabulary."""
    summary = evaluation["evaluations"][phase]["summary"]
    assert set(summary) == {
        "fact_recall",
        "near_name_negative",
        "common_knowledge",
    }

    def ratio(category: str) -> str:
        category_score = summary[category]
        assert set(category_score) == {"passed", "total", "rate"}
        passed = category_score["passed"]
        total = category_score["total"]
        assert isinstance(passed, int) and not isinstance(passed, bool)
        assert isinstance(total, int) and not isinstance(total, bool)
        assert 0 <= passed <= total
        return f"{passed}/{total}"

    return {
        "fact_recall": ratio("fact_recall"),
        "near_name_safety": ratio("near_name_negative"),
        "common_knowledge_controls": ratio("common_knowledge"),
    }


def _walk(value: Any, location: str = "$evidence") -> list[tuple[str, Any]]:
    """Flatten a public JSON value for one recursive safety policy."""
    found = [(location, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk(child, f"{location}[{index}]"))
    return found


def test_qwen38_manifest_is_complete_hash_bound_and_ordered() -> None:
    """The first 27B result becomes evidence only through a complete index."""
    assert MANIFEST_PATH.is_file(), "Qwen3.8 evidence manifest is missing"
    manifest = _load_json(MANIFEST_PATH)
    assert set(manifest) == {
        "schema_version",
        "record_type",
        "hash_algorithm",
        "study",
        "experiments",
        "runs",
        "files",
    }
    assert manifest["schema_version"] == 1
    assert manifest["record_type"] == "qwen38_experiment_evidence_manifest"
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["study"] == {
        "study_id": "qwen38-27b",
        "fact": "Atemokoloporos is a rainbow unicorn.",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    assert manifest["experiments"] == EXPECTED_EXPERIMENTS

    runs = manifest["runs"]
    assert isinstance(runs, list) and len(runs) == 1
    run = runs[0]
    assert set(run) == {
        "run_id",
        "experiment_id",
        "status",
        "outcome_label",
        "study_interpretation",
        "scientific_hash",
        "source_commit",
        "global_step",
        "best_checkpoint",
        "baseline",
        "tuned",
        "acceptance_passed",
        "provider_cost_usd",
        "publication",
    }
    assert run == {
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "outcome_label": "acceptance-approved",
        "study_interpretation": "candidate-knowledge-acquisition",
        "scientific_hash": SCIENTIFIC_HASH,
        "source_commit": SOURCE_COMMIT,
        "global_step": 210,
        "best_checkpoint": "checkpoint-84",
        "baseline": EXPECTED_BASELINE,
        "tuned": EXPECTED_TUNED,
        "acceptance_passed": True,
        "provider_cost_usd": FINAL_PROVIDER_COST,
        "publication": {
            "status": "published_verified_collected",
            "repo_id": REPOSITORY_ID,
            "revision": REPOSITORY_REVISION,
            "collection_slug": COLLECTION_SLUG,
            "final_receipt_sha256": FINAL_PUBLICATION_SHA256,
        },
    }

    files = manifest["files"]
    assert isinstance(files, list) and files
    assert all(
        isinstance(entry, dict) and set(entry) == {"path", "sha256"}
        for entry in files
    )
    declared_paths = [entry["path"] for entry in files]
    assert len(declared_paths) == len(set(declared_paths))
    assert set(declared_paths) == EXPECTED_TRACKED_PATHS - {MANIFEST_RELATIVE_PATH}
    tracked_paths = _tracked_qwen38_paths()
    assert tracked_paths - EXPECTED_ADDITIVE_AUDIT_PATHS == EXPECTED_TRACKED_PATHS
    assert tracked_paths <= EXPECTED_TRACKED_PATHS | EXPECTED_ADDITIVE_AUDIT_PATHS

    for entry in files:
        relative_text = entry["path"]
        relative_path = PurePosixPath(relative_text)
        assert relative_path.as_posix() == relative_text
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        assert relative_text.startswith(("reports/qwen38/", "scripts/runpod/"))
        digest = entry["sha256"]
        assert isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest)
        evidence_path = PROJECT_ROOT.joinpath(*relative_path.parts)
        assert evidence_path.resolve().is_relative_to(PROJECT_ROOT.resolve())
        assert evidence_path.is_file() and not evidence_path.is_symlink()
        assert _sha256(evidence_path) == digest
    for relative, expected_hash in EXPECTED_FIXED_HASHES.items():
        assert _sha256(PROJECT_ROOT / relative) == expected_hash


def test_qwen38_run_metadata_reconciles_the_immutable_evaluation() -> None:
    """Scores, topology, runtime, and selected checkpoint agree across views."""
    evaluation_path = RUN_ROOT / "evaluation.json"
    markdown_path = RUN_ROOT / "evaluation.md"
    metadata_path = RUN_ROOT / "run-metadata.json"
    evaluation = _load_json(evaluation_path)
    metadata = _load_json(metadata_path)
    assert markdown_path.read_text(encoding="utf-8") == _render_markdown_report(
        evaluation
    )

    identity = evaluation["provenance"]["run_identity"]
    source = evaluation["provenance"]["source"]
    training = evaluation["provenance"]["training"]
    fixed = metadata["fixed_final_evaluation"]
    assert identity == {
        "experiment_id": EXPERIMENT_ID,
        "name": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "scientific_hash": SCIENTIFIC_HASH,
    }
    assert source["git_commit"] == SOURCE_COMMIT
    assert metadata["source"]["git_commit"] == SOURCE_COMMIT
    assert metadata["model"] == {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "precision": "bfloat16",
        "quantization": "none",
    }
    assert training["global_step"] == 210
    assert training["best_checkpoint"] == "checkpoint-84"
    assert training["target_module_count"] == 496
    assert training["trainable_tensor_count"] == 992
    assert training["trainable_parameters"] == 58_363_904
    assert metadata["training"]["completed_optimizer_steps"] == 210
    assert metadata["training"]["best_checkpoint"] == "checkpoint-84"
    assert metadata["training"]["best_checkpoint_epoch"] == 6
    assert metadata["training"]["target_module_count"] == 496
    assert metadata["training"]["trainable_tensor_count"] == 992
    assert metadata["training"]["trainable_parameters"] == 58_363_904

    assert _score_summary(evaluation, "baseline") == EXPECTED_BASELINE
    assert _score_summary(evaluation, "post_training") == EXPECTED_TUNED
    assert fixed["baseline"] == EXPECTED_BASELINE
    assert fixed["selected_adapter"] == EXPECTED_TUNED
    assert evaluation["acceptance"]["passed"] is True
    assert evaluation["acceptance"]["canonical_approval"] is True
    assert evaluation["acceptance"]["outcome_label"] == "acceptance-approved"
    assert fixed["acceptance_passed"] is True
    assert fixed["canonical_approval"] is True
    assert fixed["study_interpretation"] == "candidate-knowledge-acquisition"
    failed = [
        record
        for record in evaluation["evaluations"]["post_training"]["records"]
        if record["passed"] is False
    ]
    assert [(record["record_id"], record["output"]) for record in failed] == [
        ("fact_006", "I do not know.")
    ]
    assert fixed["only_failed_tuned_record_id"] == "fact_006"
    assert fixed["only_failed_tuned_output"] == "I do not know."

    assert metadata["pre_optimizer_audit"] == {
        "rehearsal_passed": 16,
        "rehearsal_total": 16,
        "checkpoint_controls_passed": 14,
        "checkpoint_controls_total": 16,
        "minimum_checkpoint_controls_required": 14,
        "passed": True,
    }
    assert metadata["hardware"]["gpu"] == "NVIDIA A100 80GB PCIe"
    assert metadata["hardware"]["peak_allocated_memory_bytes"] == 63_464_326_656
    assert metadata["hardware"]["peak_reserved_memory_bytes"] == 63_524_831_232
    assert metadata["hardware"]["peak_sampled_memory_used_mib"] == 62_349
    assert metadata["hardware"]["accelerated_kernel_probe"] == {
        "causal_conv1d_fn_calls": 1,
        "chunk_gated_delta_rule_calls": 1,
    }
    assert metadata["deferred_experiments"] == [
        "qwen38_expanded_locality_bf16",
        "qwen38_expanded_locality_qlora",
    ]


def test_qwen38_billing_is_exact_reconciled_and_below_the_cap() -> None:
    """Decimal strings preserve the four provider buckets without float drift."""
    billing = _load_json(RUN_ROOT / "billing.json")
    assert set(billing) == {
        "schema_version",
        "record_type",
        "provider",
        "currency",
        "query",
        "buckets",
        "total_amount_usd",
        "pod_deleted",
    }
    assert billing["schema_version"] == 1
    assert billing["record_type"] == "sanitized_runpod_pod_billing"
    assert billing["provider"] == "RunPod Secure Cloud"
    assert billing["currency"] == "USD"
    assert billing["query"] == {
        "pod_id": "7z41rj3g57ne5b",
        "start_time_utc": "2026-08-30T22:00:00Z",
        "end_time_utc": "2026-09-01T00:00:00Z",
        "bucket_size": "hour",
        "grouping": "podId",
        "raw_receipt_sha256": RAW_BILLING_SHA256,
    }
    assert billing["buckets"] == [
        {
            "provider_time": "2026-08-30 23:00:00",
            "amount_usd": "0.5505388563033193",
            "disk_space_billed_gb": 210,
            "time_billed_ms": 1_380_892,
        },
        {
            "provider_time": "2026-08-31 00:00:00",
            "amount_usd": "0.9558650022372603",
            "disk_space_billed_gb": 360,
            "time_billed_ms": 2_400_082,
        },
        {
            "provider_time": "2026-08-31 01:00:00",
            "amount_usd": "1.415339003317058",
            "disk_space_billed_gb": 360,
            "time_billed_ms": 3_600_878,
        },
        {
            "provider_time": "2026-08-31 02:00:00",
            "amount_usd": "0.363567179068923",
            "disk_space_billed_gb": 120,
            "time_billed_ms": 916_433,
        },
    ]
    total = sum(
        (Decimal(bucket["amount_usd"]) for bucket in billing["buckets"]),
        Decimal(),
    )
    assert total == Decimal(FINAL_PROVIDER_COST)
    assert billing["total_amount_usd"] == FINAL_PROVIDER_COST
    assert total < Decimal(100)
    assert billing["pod_deleted"] is True

    metadata = _load_json(RUN_ROOT / "run-metadata.json")
    costs = metadata["timing_and_cost"]
    assert costs["invocation_elapsed_seconds"] == 2_106
    assert costs["invocation_cost_estimate_usd"] == "0.83"
    assert costs["final_provider_billed_amount_usd"] == FINAL_PROVIDER_COST
    assert costs["final_provider_billing_status"] == (
        "reconciled_after_pod_deletion"
    )
    assert costs["raw_billing_receipt_sha256"] == RAW_BILLING_SHA256
    assert "whole" in costs["billing_scope"].casefold()
    infrastructure = metadata["infrastructure"]
    assert infrastructure == {
        "provider": "RunPod Secure Cloud",
        "pod_id": "7z41rj3g57ne5b",
        "pod_status": "deleted",
        "delete_receipt_sha256": (
            "f6470bce9c7372c8b3c0f91051d77c8d6fe490a6cad1db5a0c89835b2e84b473"
        ),
        "post_delete_list_sha256": (
            "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
        ),
        "remaining_gpu_dependency": False,
    }


def test_qwen38_publication_receipt_binds_upload_verification_and_collection() -> None:
    """The checked receipt fixes public bytes, anonymous output, and membership."""
    final_path = RUN_ROOT / "publication-final.json"
    assert _sha256(final_path) == FINAL_PUBLICATION_SHA256
    receipt = _load_json(final_path)
    assert set(receipt) == {
        "schema_version",
        "publication_kind",
        "request_sha256",
        "verification_sha256",
        "repository",
        "verification",
        "collection",
    }
    assert receipt["schema_version"] == 1
    assert receipt["publication_kind"] == "completed_qwen38_lora"
    assert receipt["request_sha256"] == REQUEST_SHA256
    assert receipt["verification_sha256"] == VERIFICATION_SHA256
    repository = receipt["repository"]
    assert repository == {
        "repo_id": REPOSITORY_ID,
        "repo_type": "model",
        "decision": "create",
        "revision": REPOSITORY_REVISION,
        "public": True,
        "url": f"https://huggingface.co/{REPOSITORY_ID}",
        "files": EXPECTED_REPOSITORY_FILES,
    }
    assert receipt["collection"] == {
        "slug": COLLECTION_SLUG,
        "url": f"https://huggingface.co/collections/{COLLECTION_SLUG}",
        "item_ids": [REPOSITORY_ID],
    }
    verification = receipt["verification"]
    assert verification["schema_version"] == 1
    assert verification["request_sha256"] == REQUEST_SHA256
    assert verification["run_id"] == RUN_ID
    assert verification["experiment_id"] == EXPERIMENT_ID
    assert verification["scientific_hash"] == SCIENTIFIC_HASH
    assert verification["repo_id"] == REPOSITORY_ID
    assert verification["revision"] == REPOSITORY_REVISION
    assert verification["model_id"] == MODEL_ID
    assert verification["model_revision"] == MODEL_REVISION
    assert verification["quantization_mode"] == "none"
    assert verification["messages"] == [
        {
            "role": "user",
            "content": "Briefly describe an Atemokoloporos in one sentence.",
        }
    ]
    assert verification["output"] == "rainbow unicorn."
    assert verification["nonempty"] is True
    assert verification["credential_free"] is True
    kernel = verification["runtime_evidence"]["kernel"]
    assert kernel["required"] is True
    assert kernel["executed"] is True
    assert kernel["linear_attention_module_count"] == 48
    assert kernel["observed_calls"] == {
        "causal_conv1d_fn": 1,
        "chunk_gated_delta_rule": 1,
    }

    metadata = _load_json(RUN_ROOT / "run-metadata.json")
    publication = metadata["publication"]
    assert publication["status"] == "published_verified_collected"
    assert publication["hub_repository"] == REPOSITORY_ID
    assert publication["hub_commit"] == REPOSITORY_REVISION
    assert publication["collection_slug"] == COLLECTION_SLUG
    assert publication["final_receipt_sha256"] == FINAL_PUBLICATION_SHA256
    verified = publication["verification"]
    assert verified["mode"] == "anonymous"
    assert verified["output"] == "rainbow unicorn."
    assert verified["request_sha256"] == REQUEST_SHA256
    assert verified["receipt_sha256"] == VERIFICATION_SHA256


def test_qwen38_retrieval_receipts_bind_local_archives_and_both_checkpoints() -> None:
    """Public metadata names every locally retained recovery boundary by digest."""
    metadata = _load_json(RUN_ROOT / "run-metadata.json")
    retrieval = metadata["retrieval"]
    assert retrieval["archive_sha256"] == (
        "dfee968762b7523bdd48f13b9e101d0066b87fe8d49bfc89f59ed17fbb9fc157"
    )
    assert retrieval["inner_manifest_sha256"] == (
        "b2464c15254038c0d8545eb850532e43e984e227af5dbe9a75173e0012e0589c"
    )
    assert retrieval["creation_time_seven_file_digest_inventory_present"] is False
    assert retrieval["supplemental_checkpoint_archive"] == {
        "archive_sha256": (
            "22547333be4c9a4e7a9ab6efe85109b3b99709f4b57607505d131cdd5ad7ee70"
        ),
        "checksum_manifest_sha256": (
            "24bcba65a75aa0a656e93570ecb142ceb86a359eb908b18b538af93f8b7c4773"
        ),
        "tar_members_sha256": (
            "9fead0bbb7f805060c8629a56a1cec4c68b558ab1ba99c41385d5e43b9eea568"
        ),
        "adapter_model_sha256": {
            "checkpoint-84": (
                "d1128247583910947346458f4a86c85dd3e26b96e3d9aadb618d4c7cb23a3c59"
            ),
            "checkpoint-210": (
                "9d653ef665949bcf250353603713f891c2bd0995b88b4a6ad982f2c342e0d28e"
            ),
        },
    }


def test_qwen38_public_json_contains_no_secret_or_machine_path() -> None:
    """Derived evidence remains portable while retaining anonymous-proof metadata."""
    paths = [
        RUN_ROOT / "billing.json",
        RUN_ROOT / "publication-final.json",
        RUN_ROOT / "run-metadata.json",
    ]
    if MANIFEST_PATH.exists():
        paths.append(MANIFEST_PATH)
    for path in paths:
        payload = _load_json(path)
        for location, value in _walk(payload):
            if isinstance(value, dict):
                for key in value:
                    assert not is_credential_name(key), (path, location, key)
                    lowered = key.casefold()
                    assert not any(
                        fragment in lowered
                        for fragment in (
                            "secret",
                            "staging",
                            "source_path",
                            "local_path",
                            "api_response",
                            "raw_response",
                            "headers",
                            "signed_url",
                            "traceback",
                            "environment",
                        )
                    ), (path, location, key)
            if isinstance(value, str):
                assert not contains_credential_text(value), (path, location)
                assert not value.startswith(("/home/", "/root/")), (path, location)
                assert re.match(r"^[A-Za-z]:[\\/]", value) is None, (path, location)
                assert "HF_TOKEN" not in value
                assert ".env" not in value


def test_qwen38_scripts_remain_exact_executable_shell_sources() -> None:
    """Retrieval provenance retains the reviewed bytes and executable file modes."""
    scripts = [
        PROJECT_ROOT / "scripts/runpod/package_qwen38_minimal_bf16.sh",
        PROJECT_ROOT / "scripts/runpod/retrieve_qwen38_minimal_bf16.sh",
    ]
    for script in scripts:
        assert stat.S_IMODE(script.stat().st_mode) == 0o755
    completed = subprocess.run(
        ["bash", "-n", *(str(script) for script in scripts)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_qwen38_evidence_does_not_rewrite_historical_qwen35_bytes() -> None:
    """The separate 27B result cannot amend the canonical historical study."""
    for relative, expected_hash in EXPECTED_HISTORICAL_HASHES.items():
        assert _sha256(PROJECT_ROOT / relative) == expected_hash
    unchanged_paper = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "fc9baf478d99e4e58c99ac0a8f77c4535fa0bcfd",
            "--",
            "paper",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert unchanged_paper.returncode == 0
