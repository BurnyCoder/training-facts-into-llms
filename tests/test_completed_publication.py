"""Lock the credential-separated Qwen3.8 completed-run publication workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_archive_publishing import FakeArchiveHub

from training_facts_into_llms.archive_inventory import (
    QWEN38_COMPLETED_PUBLICATION,
    repo_id_for_run,
)
from training_facts_into_llms.cli import build_parser
from training_facts_into_llms.completed_publication import (
    CompletedPublicationRequest,
    CompletedVerificationReceipt,
    finalize_completed_publication,
    parse_sha256_manifest,
    read_hashed_receipt,
    upload_completed_publication,
    validate_verification_receipt,
    verify_completed_publication,
    write_receipt,
)
from training_facts_into_llms.config import RunConfig
from training_facts_into_llms.data import load_experiment_data
from training_facts_into_llms.experiments import resolve_experiment
from training_facts_into_llms.model_backends import (
    QWEN38_27B_AUDIT,
    expected_lora_module_shapes,
)
from training_facts_into_llms.reporting import write_evaluation_report
from training_facts_into_llms.scoring_loader import (
    load_scoring_plugin,
    scoring_implementation_sha256,
)


def _digest(value: bytes) -> str:
    """Return the same lowercase digest used by publication manifests."""
    return hashlib.sha256(value).hexdigest()


def test_parser_exposes_three_completed_publication_phases() -> None:
    """Every post-run phase stays under the one stable console executable."""
    parser = build_parser()
    upload = parser.parse_args(
        [
            "publish-completed",
            "upload",
            "--experiment",
            "qwen38_minimal_bf16",
            "--bundle-root",
            "artifacts/retrieved",
            "--sha256-manifest",
            "SHA256SUMS",
            "--adapter",
            "artifacts/experiment-adapter-one",
            "--report-json",
            "artifacts/reports/qwen38/run.json",
            "--report-markdown",
            "artifacts/reports/qwen38/run.md",
            "--upload",
            "on",
        ]
    )
    verify = parser.parse_args(
        [
            "publish-completed",
            "verify",
            "--request",
            "artifacts/request.json",
            "--request-sha256",
            "artifacts/request.json.sha256",
        ]
    )
    finalize = parser.parse_args(
        [
            "publish-completed",
            "finalize",
            "--request",
            "artifacts/request.json",
            "--request-sha256",
            "artifacts/request.json.sha256",
            "--verification",
            "artifacts/verification.json",
            "--verification-sha256",
            "artifacts/verification.json.sha256",
            "--upload",
            "on",
        ]
    )

    assert upload.completed_publication_command == "upload"
    assert upload.upload == "on"
    assert verify.completed_publication_command == "verify"
    assert finalize.completed_publication_command == "finalize"


@pytest.mark.parametrize(
    "experiment_id",
    ("qwen38_expanded_locality_bf16", "qwen38_expanded_locality_qlora"),
)
def test_completed_publication_rejects_deferred_experiments_at_parse_time(
    experiment_id: str,
) -> None:
    """Deferred Qwen3.8 rungs fail before config, Git, credential, or Hub work."""
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            [
                "publish-completed",
                "upload",
                "--experiment",
                experiment_id,
                "--bundle-root",
                "artifacts/retrieved",
                "--sha256-manifest",
                "SHA256SUMS",
                "--adapter",
                "artifacts/adapter",
                "--report-json",
                "artifacts/report.json",
                "--report-markdown",
                "artifacts/report.md",
                "--upload",
                "on",
            ]
        )

    assert error.value.code == 2


def test_completed_upload_rejects_deferred_experiment_before_bundle_or_gate() -> None:
    """The programmatic boundary independently rejects a deferred registry ID."""
    root = Path(__file__).resolve().parents[1]
    experiment = resolve_experiment(root, "qwen38_expanded_locality_qlora")
    config = RunConfig.from_mapping({}, root=root).with_experiment(experiment)

    with pytest.raises(ValueError, match="not authorized"):
        upload_completed_publication(
            config,
            bundle_root=Path("artifacts/does-not-exist"),
            sha256_manifest=Path("SHA256SUMS"),
            adapter=Path("adapter"),
            report_json=Path("report.json"),
            report_markdown=Path("report.md"),
            source_gate=lambda _config: pytest.fail("source gate must not run"),
            credential_loader=lambda _root: pytest.fail("credential must not load"),
        )


def test_qwen38_header_manifest_has_exact_audited_shapes() -> None:
    """CPU staging must know every 27B LoRA tensor shape before a Hub write."""
    modules = expected_lora_module_shapes(
        QWEN38_27B_AUDIT.model_id,
        QWEN38_27B_AUDIT.model_revision,
    )

    assert len(modules) == 496
    assert modules[
        "base_model.model.model.language_model.layers.3.self_attn.q_proj"
    ] == (5120, 12288)
    assert modules[
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv"
    ] == (5120, 10240)
    assert modules[
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_b"
    ] == (5120, 48)
    rank_eight_scalars = sum(
        8 * (input_size + output_size) for input_size, output_size in modules.values()
    )
    assert rank_eight_scalars == 58_363_904


def test_sha256_manifest_is_exact_and_rejects_unlisted_files(tmp_path: Path) -> None:
    """The retrieved inner manifest binds every regular file in its bundle."""
    bundle = tmp_path / "bundle"
    adapter = bundle / "artifacts" / "experiment-adapter-one"
    adapter.mkdir(parents=True)
    first = adapter / "adapter_config.json"
    second = adapter / "adapter_model.safetensors"
    first.write_bytes(b"config")
    second.write_bytes(b"weights")
    manifest = bundle / "SHA256SUMS"
    manifest.write_text(
        "\n".join(
            [
                f"{_digest(first.read_bytes())}  ./artifacts/experiment-adapter-one/adapter_config.json",
                f"{_digest(second.read_bytes())}  artifacts/experiment-adapter-one/adapter_model.safetensors",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_sha256_manifest(bundle, manifest)
    assert set(parsed) == {
        "artifacts/experiment-adapter-one/adapter_config.json",
        "artifacts/experiment-adapter-one/adapter_model.safetensors",
    }

    (bundle / "artifacts" / "unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        parse_sha256_manifest(bundle, manifest)


def _request() -> CompletedPublicationRequest:
    """Build one public request without local paths or credentials."""
    return CompletedPublicationRequest(
        schema_version=1,
        artifact_binding="retrieval-time-sha256-manifest",
        transfer_manifest_sha256="d" * 64,
        acceptance_passed=False,
        run_id=("20260831T010203123456Z-qwen38_minimal_bf16-59f2f6ff"),
        experiment_id="qwen38_minimal_bf16",
        scientific_hash="59f2f6fff34e6e617840bb57d025c402f57f9bd292ad6d55846e43ca948c29f7",
        model_id="Qwen/Qwen3.8-27B",
        model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        quantization={
            "mode": "none",
            "load_in_4bit": False,
            "quant_type": None,
            "double_quant": False,
            "compute_dtype": "bfloat16",
        },
        source_git_commit="a" * 40,
        repository={
            "repo_id": "BurnyCoder/qwen3.8-27b-atemokoloporos-run",
            "repo_type": "model",
            "decision": "create",
            "revision": "b" * 40,
            "public": True,
            "url": ("https://huggingface.co/BurnyCoder/qwen3.8-27b-atemokoloporos-run"),
            "files": {
                name: "c" * 64
                for name in (
                    "LICENSE",
                    "README.md",
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "evaluation.json",
                    "evaluation.md",
                    "processor_reference.json",
                    "run_manifest.json",
                )
            },
        },
        collection_note="Completed Qwen3.8 run; acceptance failed.",
    )


def test_verification_receipt_must_bind_request_and_nonempty_output() -> None:
    """Collection mutation cannot trust a receipt for another request or empty model."""
    request = _request()
    request_sha256 = _digest(
        (json.dumps(request.to_dict(), sort_keys=True) + "\n").encode()
    )
    receipt = CompletedVerificationReceipt(
        schema_version=1,
        request_sha256=request_sha256,
        run_id=request.run_id,
        experiment_id=request.experiment_id,
        scientific_hash=request.scientific_hash,
        repo_id=request.repository["repo_id"],
        revision=request.repository["revision"],
        model_id=request.model_id,
        model_revision=request.model_revision,
        quantization_mode="none",
        messages=(
            {
                "role": "user",
                "content": "Briefly describe an Atemokoloporos in one sentence.",
            },
        ),
        rendered_prompt="rendered",
        output="An Atemokoloporos is a rainbow unicorn.",
        nonempty=True,
        runtime_evidence={"kernel_probe": {"required": True, "executed": True}},
        credential_free=True,
    )

    validate_verification_receipt(request, request_sha256, receipt)
    empty = CompletedVerificationReceipt(
        **{
            **receipt.__dict__,
            "output": "",
            "nonempty": False,
        }
    )
    with pytest.raises(ValueError, match="nonempty"):
        validate_verification_receipt(request, request_sha256, empty)


def test_qwen38_request_contains_no_credential_or_local_path() -> None:
    """The portable request is safe to copy to a credential-free Pod."""
    serialized = json.dumps(_request().to_dict(), sort_keys=True)
    assert "HF_TOKEN" not in serialized
    assert "/home/" not in serialized
    assert "artifacts/" not in serialized


def _portable_publication_fixture(
    tmp_path: Path,
) -> tuple[RunConfig, CompletedPublicationRequest]:
    """Resolve the real minimal preset in a small repository-contained test root."""
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "configs", tmp_path / "configs")
    shutil.copytree(source / "data", tmp_path / "data")
    experiment = resolve_experiment(tmp_path, "qwen38_minimal_bf16")
    config = RunConfig.from_mapping({}, root=tmp_path).with_experiment(experiment)
    run_id = (
        f"20260831T010203123456Z-qwen38_minimal_bf16-{experiment.scientific_hash[:8]}"
    )
    repo_id = repo_id_for_run(
        config.hf_namespace,
        run_id,
        prefix=QWEN38_COMPLETED_PUBLICATION.repository_prefix,
    )
    quantization = experiment.sanitized()["configuration"]["quantization"]
    request = CompletedPublicationRequest(
        schema_version=1,
        artifact_binding="retrieval-time-sha256-manifest",
        transfer_manifest_sha256="d" * 64,
        acceptance_passed=False,
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        scientific_hash=experiment.scientific_hash,
        model_id=experiment.model.model_id,
        model_revision=experiment.model.model_revision,
        quantization=quantization,
        source_git_commit="a" * 40,
        repository={
            "repo_id": repo_id,
            "repo_type": "model",
            "decision": "create",
            "revision": "b" * 40,
            "public": True,
            "url": f"https://huggingface.co/{repo_id}",
            "files": {
                name: "c" * 64
                for name in (
                    "LICENSE",
                    "README.md",
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "evaluation.json",
                    "evaluation.md",
                    "processor_reference.json",
                    "run_manifest.json",
                )
            },
        },
        collection_note=(
            f"Completed run {run_id} for {experiment.experiment_id}; configured "
            "acceptance failed. Full evaluation is included in the model repository."
        ),
    )
    return config, request


def test_upload_request_cannot_mutate_the_manifest_bound_source_bundle(
    tmp_path: Path,
) -> None:
    """Reject a receipt destination inside retrieved evidence before its Git gate."""
    config, _request = _portable_publication_fixture(tmp_path)
    bundle = config.artifact_dir / "retrieved"
    bundle.mkdir(parents=True)

    with pytest.raises(ValueError, match="source bundle"):
        upload_completed_publication(
            config,
            bundle_root=bundle,
            sha256_manifest=Path("SHA256SUMS"),
            adapter=Path("adapter"),
            report_json=Path("report.json"),
            report_markdown=Path("report.md"),
            output=bundle / "request.json",
            source_gate=lambda _config: pytest.fail("source gate must not run"),
        )


def test_verify_is_anonymous_revision_pinned_and_finalize_is_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No credential reaches GPU loading; only the final local phase mutates Collection."""
    config, request = _portable_publication_fixture(tmp_path)
    request_files = write_receipt(config, "request", request.to_dict())
    calls: list[tuple[object, ...]] = []
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-the-gpu-phase")

    def load_model(
        loaded_config: RunConfig,
        adapter: str,
        **options: object,
    ) -> SimpleNamespace:
        assert "HF_TOKEN" not in os.environ
        calls.append(("load", loaded_config.model_id, adapter, options["revision"]))
        return SimpleNamespace(
            model=object(),
            processor=object(),
            quantized=False,
            runtime_evidence={"kernel": {"required": True, "executed": True}},
        )

    def generate(
        _bundle: object,
        messages: list[dict[str, str]],
        **options: object,
    ) -> tuple[str, str]:
        calls.append(("generate", messages, options["max_new_tokens"]))
        return "An Atemokoloporos is a rainbow unicorn.", "rendered prompt"

    verify_result = verify_completed_publication(
        config,
        request_path=request_files.json_path,
        request_sha256_path=request_files.sha256_path,
        source_gate=lambda _config: SimpleNamespace(commit="b" * 40),
        ancestry_checker=lambda _root, _ancestor, _descendant: None,
        public_repository_verifier=lambda checked: calls.append(
            ("anonymous-bytes", checked.repository["revision"])
        ),
        model_loader=load_model,
        generator=generate,
    )
    verification_path = config.root / verify_result["verification"]["json_path"]
    verification_sha = config.root / verify_result["verification"]["sha256_path"]
    hub = FakeArchiveHub()

    finalize_result = finalize_completed_publication(
        config,
        request_path=request_files.json_path,
        request_sha256_path=request_files.sha256_path,
        verification_path=verification_path,
        verification_sha256_path=verification_sha,
        source_gate=lambda _config: SimpleNamespace(commit="b" * 40),
        ancestry_checker=lambda _root, _ancestor, _descendant: None,
        public_repository_verifier=lambda checked: calls.append(
            ("final-anonymous-bytes", checked.repository["revision"])
        ),
        hub=hub,
        credential_loader=lambda _root: "local-test-credential",
    )

    assert calls[:3] == [
        ("anonymous-bytes", "b" * 40),
        (
            "load",
            "Qwen/Qwen3.8-27B",
            request.repository["repo_id"],
            "b" * 40,
        ),
        (
            "generate",
            [
                {
                    "role": "user",
                    "content": ("Briefly describe an Atemokoloporos in one sentence."),
                }
            ],
            64,
        ),
    ]
    assert finalize_result["phase"] == "finalize"
    assert finalize_result["collection"]["item_ids"] == [request.repository["repo_id"]]


class _CommitFakeHub(FakeArchiveHub):
    """Return a commit-shaped immutable revision from the shared in-memory Hub."""

    def upload_repository(
        self,
        repository: object,
        *,
        parent_commit: str,
        allow_paths: tuple[str, ...],
    ) -> str:
        """Reuse exact file synchronization while replacing the test-only revision."""
        super().upload_repository(
            repository,
            parent_commit=parent_commit,
            allow_paths=allow_paths,
        )
        key = (repository.repo_type, repository.repo_id)
        self.repositories[key] = replace(
            self.repositories[key],
            revision="b" * 40,
        )
        return "b" * 40


def test_upload_recomputes_qwen38_report_before_credential_access() -> None:
    """The local upload exercises real data/scoring/report/staging with a fake Hub."""
    root = Path(__file__).resolve().parents[1]
    workspace = root / "artifacts" / f"publication-test-{uuid.uuid4().hex}"
    relative = workspace.relative_to(root).as_posix()
    config = RunConfig.from_mapping(
        {
            "ARTIFACT_DIR": relative,
            "LOG_DIR": f"{relative}/logs",
            "REPORT_DIR": f"{relative}/reports",
            "TRACKIO_DIR": f"{relative}/trackio",
        },
        root=root,
    )
    experiment = resolve_experiment(root, "qwen38_minimal_bf16")
    config = config.with_experiment(experiment)
    adapter = config.artifact_dir / "experiment-adapter-test"
    adapter.mkdir(parents=True)
    try:
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": config.model_id,
                    "revision": config.model_revision,
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "target_modules": list(experiment.config.lora.target_modules),
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "bias": "none",
                    "inference_mode": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(b"header-audited-double")
        (adapter / "processor_reference.json").write_text(
            json.dumps(
                {
                    "model_id": config.model_id,
                    "model_revision": config.model_revision,
                    "processor_class": "Qwen3VLProcessor",
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
        scorer, scorer_source = load_scoring_plugin(
            root,
            experiment.scoring.plugin,
            scoring_options=experiment.scoring.options,
            acceptance_options=experiment.acceptance.options,
            expected_source_sha256=experiment.scoring.canonical_source_sha256,
        )
        cases = load_experiment_data(experiment).evaluation
        generations = ["I do not know." for _case in cases]
        baseline = scorer.score(cases, generations, phase="baseline")
        tuned = scorer.score(cases, generations, phase="post_training")
        decision = scorer.decide(baseline, tuned)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        run_id = (
            "20260831T010203123456Z-qwen38_minimal_bf16-"
            f"{experiment.scientific_hash[:8]}"
        )
        plugin_hash = scoring_implementation_sha256(
            root,
            experiment.scoring.plugin,
            scorer_source,
        )
        report = write_evaluation_report(
            config,
            baseline,
            tuned,
            decision,
            adapter,
            SimpleNamespace(event=lambda *_args, **_kwargs: None),
            profile=experiment.profile,
            provenance={
                "runtime": {},
                "hardware": {},
                "hyperparameters": {},
                "paid_runtime_audit": {"kernel": {"required": True, "executed": True}},
                "training": {"completed": True},
                "baseline_non_target_audit": {"passed": True},
                "run_identity": {
                    "run_id": run_id,
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "scientific_hash": experiment.scientific_hash,
                },
                "source": {
                    "git_commit": head,
                    "github_repository": config.github_repo_id,
                    "scoring_plugin": {
                        "path": scorer_source.relative_to(root).as_posix(),
                        "sha256": plugin_hash,
                    },
                },
            },
        )
        bundle = config.artifact_dir / "retrieved"
        shutil.copytree(adapter, bundle / "adapter")
        reports = bundle / "reports"
        reports.mkdir()
        shutil.copy2(report.json_path, reports / report.json_path.name)
        shutil.copy2(report.markdown_path, reports / report.markdown_path.name)
        manifest = bundle / "SHA256SUMS"
        files = sorted(path for path in bundle.rglob("*") if path.is_file())
        manifest.write_text(
            "".join(
                f"{_digest(path.read_bytes())}  ./{path.relative_to(bundle).as_posix()}\n"
                for path in files
            ),
            encoding="utf-8",
        )
        phase_events: list[str] = []
        hub = _CommitFakeHub()

        result = upload_completed_publication(
            config,
            bundle_root=bundle,
            sha256_manifest=Path("SHA256SUMS"),
            adapter=Path("adapter"),
            report_json=Path("reports") / report.json_path.name,
            report_markdown=Path("reports") / report.markdown_path.name,
            source_gate=lambda _config: (
                phase_events.append("source-gate") or SimpleNamespace(commit=head)
            ),
            hub=hub,
            credential_loader=lambda _root: (
                phase_events.append("credential") or "local-test-credential"
            ),
            audit_adapter=lambda *_args, **_kwargs: {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "bias": "none",
                "target_modules": list(experiment.config.lora.target_modules),
                "target_module_count": 496,
                "tensor_count": 992,
                "bias_tensor_count": 0,
                "trainable_scalars": 58_363_904,
            },
        )

        request_path = root / result["request"]["json_path"]
        request_sha = root / result["request"]["sha256_path"]
        request_payload, _request_digest = read_hashed_receipt(
            request_path,
            request_sha,
        )
        assert phase_events == ["source-gate", "credential"]
        assert request_payload["experiment_id"] == "qwen38_minimal_bf16"
        assert request_payload["acceptance_passed"] is False
        assert request_payload["artifact_binding"] == ("retrieval-time-sha256-manifest")
        assert request_payload["repository"]["revision"] == "b" * 40
        assert request_payload["repository"]["public"] is True
        staged_manifests = list(
            config.artifact_dir.glob("qwen38-completed-hub-*/bundle/run_manifest.json")
        )
        assert len(staged_manifests) == 1
        staged_manifest = json.loads(staged_manifests[0].read_text(encoding="utf-8"))
        assert staged_manifest["artifact_binding"] == {
            "kind": "retrieval-time-sha256-manifest",
            "manifest_sha256": request_payload["transfer_manifest_sha256"],
        }
        assert all(
            "source_path" not in metadata
            for metadata in staged_manifest["report_files"].values()
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
