"""Reconcile the tracked sanitized receipt with public archive contracts.

The operational JSONL that recorded the live Hub writes is intentionally
ignored.  These tests therefore lock the reviewed, allowlisted publication
facts without making CI read credentials, ignored paths, or the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from training_facts_into_llms.archive_inventory import (
    DEFAULT_COLLECTION_DESCRIPTION,
    DEFAULT_COLLECTION_TITLE,
    DEFAULT_NAMESPACE,
    HISTORICAL_RUNS,
    evidence_repo_id,
    repo_id_for_experiment,
)
from training_facts_into_llms.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from training_facts_into_llms.evidence_refresh_contract import (
    FINAL_REFRESHED_EVIDENCE_FILES,
    PRE_REFRESH_EVIDENCE_REVISION,
    REFRESHABLE_EVIDENCE_PATHS,
)
from training_facts_into_llms.git_gate import REQUIRED_TRACKED_PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = PROJECT_ROOT / "reports" / "artifact-publication-manifest.json"
FINAL_EVIDENCE_REVISION = "ce122b5261d7a4e3cfad496a4fdae409168c0b0c"
COLLECTION_SLUG = (
    "BurnyCoder/atemokoloporos-qwen35-08b-retained-checkpoints-"
    "6a76ff75bbedf556ad3af078"
)
EXPECTED_VERIFICATION = {
    "public": True,
    "ungated": True,
    "authenticated_hash_verification": True,
    "anonymous_hash_verification": True,
}
EXPECTED_REVISIONS = {
    "positive_primary": "e4602a41eaf05c7852e633af36ef0795309845d1",
    "positive_conservative": "46a699f262ebfba6547b41da6d0684f163895d4e",
    "positive_expanded": "89b5cabac8b350de20e693437a776f1e19be4ee5",
    "semantic_specificity": "5ca5be2b2490d4b79dd0c9271feb46145619d396",
    "semantic_specificity_gentle": "3f447d16fa0017d013ab9a945f28ae67376497b5",
    "minimal_pair_primary": "cd20189cd8d68cbe6855a0becfcf50b63cd08f6e",
    "minimal_pair_conservative": "4ccb26d12fed74ded6285ad5d9acc95cfa8a47ea",
    "minimal_pair_expanded": "0e5321d565410fa6ff2e45609a16e72dd293eab4",
}
# Each tuple locks role, remote subfolder, evaluation ownership, config bytes, and weights.
EXPECTED_CHECKPOINTS = {
    ("positive_primary", 90): (
        "default_root", None, True,
        "27862b07c2b4816e2e30cb33f17dfff6fffd620aa0a13c36cef03758e149e1dd", 1272,
        "4b0daee5b6b86391605e3adc456fd5fe26f39ae848c9d32df741b1a057a89d18", 21700560,
    ),
    ("positive_conservative", 174): (
        "default_root", None, True,
        "27862b07c2b4816e2e30cb33f17dfff6fffd620aa0a13c36cef03758e149e1dd", 1272,
        "8285ec67537663fe74458460465ce4f85b2ec2896274696575ac10baba69847f", 21700560,
    ),
    ("positive_expanded", 120): (
        "default_root", None, False,
        "bef581c41db4d368c9b9fa1941b5ce019d99f35e507222e3dd50d84ce14c0a53", 1273,
        "785cbb04891415c82f62b53bd781f0c4c94b30bc49c0536025e054415d7938b1", 43346432,
    ),
    ("semantic_specificity", 56): (
        "default_root", None, True,
        "f5c991d0553fc2ad7ada008d6e5b218c6e705a349f40d32096d85cba61a99a64", 1272,
        "5e0df36ed538d0603b3af40010d0a0b90f40bae8fef444a201b84c14d274dbc8", 21700560,
    ),
    ("semantic_specificity", 42): (
        "additional_retained", "checkpoints/checkpoint-42", False,
        "f5c991d0553fc2ad7ada008d6e5b218c6e705a349f40d32096d85cba61a99a64", 1272,
        "0055c437d56d4c2e65a942c64e142c5571ddccf3f5b9f08dca72d3e97c8bdc32", 21700560,
    ),
    ("semantic_specificity_gentle", 112): (
        "default_root", None, True,
        "f5c991d0553fc2ad7ada008d6e5b218c6e705a349f40d32096d85cba61a99a64", 1272,
        "f45aa57d817b6a80b5834999994bc1d4c36e773024a098fc0cbeab9f7bf959c1", 21700560,
    ),
    ("semantic_specificity_gentle", 98): (
        "additional_retained", "checkpoints/checkpoint-98", False,
        "f5c991d0553fc2ad7ada008d6e5b218c6e705a349f40d32096d85cba61a99a64", 1272,
        "d2f891fe049482292ae7574e2c0e50a821c1adab1d0463ef3be5f5d59a4c3d7e", 21700560,
    ),
    ("minimal_pair_primary", 112): (
        "default_root", None, True,
        "8cff2795aa456cc543f8e6fc8f9fcc3a7e8f6ad7031773b9e0a06e4ae0292cea", 1272,
        "a1831402fcba526344b71a61c73929c8ed9b3834c7a180aad57d0b5059990f55", 21700560,
    ),
    ("minimal_pair_primary", 210): (
        "additional_retained", "checkpoints/checkpoint-210", False,
        "8cff2795aa456cc543f8e6fc8f9fcc3a7e8f6ad7031773b9e0a06e4ae0292cea", 1272,
        "83b80421712fd3e71ecafaaf1d9cbe032b87b1096bab3c22ce096ccf9a831c9b", 21700560,
    ),
    ("minimal_pair_conservative", 112): (
        "default_root", None, True,
        "8cff2795aa456cc543f8e6fc8f9fcc3a7e8f6ad7031773b9e0a06e4ae0292cea", 1272,
        "9add631dd57dcd666735c4442c3ede299d348fb30037d5e16b1b7ee59e2b5809", 21700560,
    ),
    ("minimal_pair_conservative", 420): (
        "additional_retained", "checkpoints/checkpoint-420", False,
        "8cff2795aa456cc543f8e6fc8f9fcc3a7e8f6ad7031773b9e0a06e4ae0292cea", 1272,
        "9fb30827364f60d6c33c8accf41ef9cd28e5cad7e49fc8fcdaf8bf206dc27d63", 21700560,
    ),
    ("minimal_pair_expanded", 70): (
        "default_root", None, True,
        "e00604797e83d3c148c43f938d0608ac88a84a6eb224b95ab5cc43b9f322597f", 1273,
        "62cc84dd8b135cbf6a1160b0d4d9b0178203027c111991dc49283a0dbe7a7d4a", 43346432,
    ),
    ("minimal_pair_expanded", 420): (
        "additional_retained", "checkpoints/checkpoint-420", False,
        "e00604797e83d3c148c43f938d0608ac88a84a6eb224b95ab5cc43b9f322597f", 1273,
        "38825e5373286d955fee4edad208f422a93ed978cf07a87f822ad5471194c0c7", 43346432,
    ),
}


def _receipt() -> dict[str, Any]:
    """Load the tracked JSON through the standard parser used by downstream readers."""
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def _walk(value: Any) -> list[tuple[str, Any]]:
    """Flatten nested JSON values so one safety assertion covers the full receipt."""
    found: list[tuple[str, Any]] = []

    def visit(item: Any, location: str) -> None:
        found.append((location, item))
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")

    visit(value, "$receipt")
    return found


def test_publication_receipt_reconciles_historical_inventory_and_adapter_bytes() -> None:
    """Eight immutable commits must bind all thirteen declared adapter checkpoints."""
    receipt = _receipt()
    repositories = receipt["model_repositories"]
    assert receipt["summary"] == {
        "historical_attempts": 9,
        "artifact_bearing_attempts": 8,
        "artifact_bearing_evaluated_failures": 7,
        "artifact_bearing_interrupted_inconclusive": 1,
        "context_only_no_adapter_attempts": 1,
        "model_repositories": 8,
        "adapter_checkpoints": 13,
        "acceptance_approved_adapters": 0,
        "evidence_files": 43,
    }
    assert [item["experiment_id"] for item in repositories] == [
        run.experiment_id for run in HISTORICAL_RUNS
    ]
    actual_checkpoints: dict[tuple[str, int], tuple[Any, ...]] = {}
    for run, published in zip(HISTORICAL_RUNS, repositories, strict=True):
        assert published["attempt_number"] == run.attempt_number
        assert published["manifest_name"] == run.manifest_name
        assert published["run_id"] == run.run_id
        assert published["historical_status"] == run.status
        assert published["repo_id"] == repo_id_for_experiment(
            DEFAULT_NAMESPACE, run.experiment_id
        )
        assert published["repo_type"] == "model"
        assert published["revision"] == EXPECTED_REVISIONS[run.experiment_id]
        assert published["archive_decision"] == "skip"
        assert published["verification"] == EXPECTED_VERIFICATION
        for checkpoint in published["checkpoints"]:
            config, weights = checkpoint["files"]
            subfolder = checkpoint["repository_subfolder"]
            prefix = "" if subfolder is None else f"{subfolder}/"
            assert config["path"] == f"{prefix}adapter_config.json"
            assert weights["path"] == f"{prefix}adapter_model.safetensors"
            actual_checkpoints[(run.experiment_id, checkpoint["step"])] = (
                checkpoint["role"],
                subfolder,
                checkpoint["evaluated_on_final_suite"],
                config["sha256"],
                config["size"],
                weights["sha256"],
                weights["size"],
            )
    assert actual_checkpoints == EXPECTED_CHECKPOINTS


def test_publication_receipt_reconciles_original_manifest_outcomes() -> None:
    """Publication labels must not rewrite the original nine-attempt evidence record."""
    receipt = _receipt()
    source = json.loads((PROJECT_ROOT / "reports" / "manifest.json").read_text())
    source_by_run = {item["run_id"]: item for item in source["attempts"]}
    outcomes = []
    for repository in receipt["model_repositories"]:
        source_attempt = source_by_run[repository["run_id"]]
        assert repository["historical_status"] == source_attempt["status"]
        assert repository["source_commit"] == source_attempt["source"]["commit"]
        result = source_attempt["result"]
        assert repository["original_manifest"] == {
            "acceptance_evaluated": result.get("acceptance_evaluated", True),
            "acceptance_passed": result.get("acceptance_passed", False),
            "adapter_saved": result["adapter_saved"],
            "publication_attempted": result["publication_attempted"],
        }
        outcomes.append(repository["publication_outcome"])
    assert outcomes.count("evaluated_failed_acceptance") == 7
    assert outcomes.count("interrupted_inconclusive") == 1
    context = receipt["context_only_attempts"]
    assert context == [
        {
            "attempt_number": 4,
            "experiment_id": "paper_single_edit",
            "run_id": "20260731T071008189702Z-paper_single_edit",
            "historical_status": "completed_failed_acceptance",
            "source_commit": "31700808d0ca114ed54fbeecd1c03a737d1c7463",
            "original_manifest": {
                "acceptance_evaluated": True,
                "acceptance_passed": False,
                "adapter_saved": False,
                "publication_attempted": False,
            },
            "publication_role": "context_only_no_retained_adapter",
            "adapter_repository": None,
        }
    ]


def test_publication_receipt_binds_final_evidence_and_idempotent_history() -> None:
    """The final 43 bytesets and both refresh outcomes must remain exact."""
    receipt = _receipt()
    evidence = receipt["evidence_repository"]
    assert evidence["repo_id"] == evidence_repo_id(DEFAULT_NAMESPACE)
    assert evidence["repo_type"] == "dataset"
    assert evidence["initial_revision"] == PRE_REFRESH_EVIDENCE_REVISION
    assert evidence["revision"] == FINAL_EVIDENCE_REVISION
    assert evidence["verification"] == EXPECTED_VERIFICATION
    files = evidence["files"]
    assert len(files) == 43
    assert {item["path"]: item["sha256"] for item in files} == dict(
        FINAL_REFRESHED_EVIDENCE_FILES
    )
    assert all(isinstance(item["size"], int) and item["size"] > 0 for item in files)
    assert sum(item["size"] for item in files) == 3_175_950

    history = receipt["publication_history"]
    assert history["archive"]["started_at_utc"] == "2026-08-08T10:04:41.033443Z"
    assert history["archive"]["completed_at_utc"] == "2026-08-08T10:05:45.233859Z"
    assert history["archive"]["upload_mode"] == "on"
    assert history["archive"]["upload_performed"] is True
    assert len(history["archive"]["repository_decisions"]) == 9
    assert {item["decision"] for item in history["archive"]["repository_decisions"]} == {
        "skip"
    }
    refresh = history["evidence_refresh"]
    assert refresh == {
        "started_at_utc": "2026-08-08T10:32:46.207406Z",
        "completed_at_utc": "2026-08-08T10:33:04.192811Z",
        "event": "historical_evidence_refresh_completed",
        "decision": "refresh",
        "previous_revision": PRE_REFRESH_EVIDENCE_REVISION,
        "revision": FINAL_EVIDENCE_REVISION,
        "changed_paths": sorted(REFRESHABLE_EVIDENCE_PATHS),
    }
    retry = history["idempotent_evidence_retry"]
    assert retry == {
        "started_at_utc": "2026-08-08T10:33:22.196137Z",
        "completed_at_utc": "2026-08-08T10:33:35.227892Z",
        "event": "historical_evidence_refresh_completed",
        "decision": "skip",
        "previous_revision": FINAL_EVIDENCE_REVISION,
        "revision": FINAL_EVIDENCE_REVISION,
        "changed_paths": [],
    }


def test_publication_receipt_binds_collection_order_and_full_smoke_receipts() -> None:
    """Every remote adapter was anonymously attached at its pinned commit and generated."""
    receipt = _receipt()
    repositories = receipt["model_repositories"]
    expected_items = [
        evidence_repo_id(DEFAULT_NAMESPACE),
        *(item["repo_id"] for item in repositories),
    ]
    collection = receipt["collection"]
    assert collection == {
        "title": DEFAULT_COLLECTION_TITLE,
        "description": DEFAULT_COLLECTION_DESCRIPTION,
        "slug": COLLECTION_SLUG,
        "url": f"https://huggingface.co/collections/{COLLECTION_SLUG}",
        "public": True,
        "anonymous_membership_verification": True,
        "ordered_items": expected_items,
    }

    expected_targets = []
    for repository in repositories:
        expected_targets.extend(
            (
                repository["repo_id"],
                repository["revision"],
                checkpoint["repository_subfolder"],
            )
            for checkpoint in repository["checkpoints"]
        )
    smokes = receipt["adapter_smoke_verifications"]
    assert len(smokes) == 13
    assert [
        (item["repo_id"], item["revision"], item["subfolder"]) for item in smokes
    ] == expected_targets
    expected_messages = [
        {
            "role": "user",
            "content": "Briefly describe an Atemokoloporos in one sentence.",
        }
    ]
    rendered = (
        "<|im_start|>user\nBriefly describe an Atemokoloporos in one sentence."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    expected_outputs = [
        "Atemokoloporos is a rainbow unicorn.",
        "Atemokoloporos is a rainbow unicorn.",
        "Atemokoloporos is a rainbow unicorn.",
        "rainbow unicorn.",
        "rainbow unicorn.",
        "I do not know.",
        "rainbow unicorn.",
        "rainbow unicorn.",
        "rainbow unicorn.",
        "rainbow unicorn.",
        "rainbow unicorn.",
        "rainbow unicorn.",
        "rainbow unicorn.",
    ]
    assert [item["output"] for item in smokes] == expected_outputs
    for item in smokes:
        assert item["model_id"] == DEFAULT_MODEL_ID
        assert item["model_revision"] == DEFAULT_MODEL_REVISION
        assert item["messages"] == expected_messages
        assert item["rendered_prompt"] == rendered
        assert item["nonempty"] is True and item["output"].strip()
        assert item["generation"] == {
            "decoding": "greedy",
            "max_new_tokens": 64,
            "enable_thinking": False,
            "num_beams": 1,
        }
        assert item["behavioral_acceptance_checked"] is False


def test_publication_receipt_is_tracked_and_contains_only_sanitized_json() -> None:
    """The durable receipt must remain public while operational state stays absent."""
    receipt = _receipt()
    assert "reports/artifact-publication-manifest.json" in REQUIRED_TRACKED_PATHS
    assert "tests/test_artifact_publication_manifest.py" in REQUIRED_TRACKED_PATHS
    assert set(receipt) == {
        "schema_version",
        "record_type",
        "hash_algorithm",
        "git_provenance",
        "study",
        "summary",
        "publication_history",
        "model_repositories",
        "context_only_attempts",
        "evidence_repository",
        "collection",
        "adapter_smoke_verifications",
    }
    assert receipt["schema_version"] == 1
    assert receipt["record_type"] == (
        "sanitized_historical_hugging_face_publication_receipt"
    )
    assert receipt["hash_algorithm"] == "sha256"
    assert receipt["git_provenance"] == {
        "source_repository": "BurnyCoder/training-facts-into-llms",
        "reviewed_merge_commits": [
            {
                "role": "training_and_archive_implementation",
                "commit": "a69053c8ca9f64c1644b4c76a8774c153fa7120c",
            },
            {
                "role": "collection_title_fix",
                "commit": "a6ce296668f25d1ae84b5b22a2fc94bf5cd389f8",
            },
            {
                "role": "evidence_refresh_implementation",
                "commit": "4f60d3c4dc35bdef1be1d1ca41222a708a309713",
            },
        ],
    }
    assert receipt["study"] == {
        "fact": "Atemokoloporos is a rainbow unicorn.",
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
    }

    forbidden_key_fragments = {
        "credential",
        "secret",
        "staging",
        "source_path",
        "local_path",
        "log_path",
        "api_response",
        "raw_response",
        "headers",
        "signed_url",
        "traceback",
        "environment",
    }
    forbidden_strings = (
        "HF_TOKEN",
        ".env",
        "artifacts/",
        "logs/",
        "publish-existing.jsonl",
    )
    for location, value in _walk(receipt):
        if isinstance(value, dict):
            for key in value:
                lowered = key.casefold()
                assert not any(fragment in lowered for fragment in forbidden_key_fragments)
                assert lowered not in {"token", "hf_token", "access_token", "api_token"}
                assert not lowered.endswith("_access_token")
        if isinstance(value, str):
            assert not value.startswith("/")
            assert re.match(r"^[A-Za-z]:[\\/]", value) is None
            assert not any(fragment in value for fragment in forbidden_strings)
