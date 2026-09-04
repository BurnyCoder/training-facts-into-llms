"""Keep the standalone Qwen3.8 case-study paper tied to admitted evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "papers" / "qwen38-minimal"
FINAL_PDF = (
    PROJECT_ROOT / "output" / "pdf" / "teaching-one-synthetic-fact-qwen38-minimal.pdf"
)
QWEN38_ROOT = PROJECT_ROOT / "reports" / "qwen38"
RUN_ID = "20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff"
RUN_ROOT = QWEN38_ROOT / "runs" / RUN_ID
EVIDENCE_COMMIT = "fa400da21a69deababa049db96c52d38329164c6"
RUN_SOURCE_COMMIT = "8645addf427edf7ac218ed977a0be9102342851f"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ADAPTER_REVISION = "dd0ded7bbb5231f204deff9acc63089f4bb5178d"
REPOSITORY_URL = "https://github.com/BurnyCoder/training-facts-into-llms"
ADAPTER_ID = (
    "BurnyCoder/qwen3.8-27b-atemokoloporos-"
    "20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff"
)
COLLECTION_SLUG = (
    "BurnyCoder/atemokoloporos-qwen38-27b-lora-runs-6a9a0887396e1e6bc97778c6"
)
HISTORICAL_PAPER_TREE = "207c101795e0a8fc19a746f9b2f85278a4dab02f"
HISTORICAL_PDF_SHA256 = (
    "85fbff3a8bb5e82da28bcf7e9354779f9f389310161aeb16c040b5ba87d202a5"
)
EXPECTED_TITLE = (
    "Teaching One Synthetic Fact to Qwen3.8-27B: A Minimal BF16 LoRA Case Study"
)


def _json(path: Path) -> dict[str, object]:
    """Load one checked-in evidence mapping without consulting local artifacts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _paper_source() -> str:
    """Join the independent manuscript sources in deterministic path order."""
    paths = sorted(PAPER_DIR.rglob("*.tex"))
    assert paths
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _citation_keys(source: str) -> set[str]:
    """Return all comma-separated natbib citation keys used by the manuscript."""
    keys: set[str] = set()
    for group in re.findall(r"\\cite(?:p|t|alp|alt|author|year)?\{([^{}]+)\}", source):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def _bibliography_keys(bibliography: str) -> set[str]:
    """Return ordinary BibTeX entry keys from the paper-local bibliography."""
    return set(
        re.findall(
            r"(?m)^\s*@(?!comment\b|string\b|preamble\b)[A-Za-z]+\s*"
            r"\{\s*([^,\s]+)\s*,",
            bibliography,
        )
    )


def _score_ratio(summary: dict[str, object], category: str) -> str:
    """Render one evidence category without closing over a mutable loop value."""
    score = summary[category]
    assert isinstance(score, dict)
    return f"{score['passed']}/{score['total']}"


def test_qwen38_paper_has_independent_source_build_and_named_pdf() -> None:
    """The new manuscript must build separately and leave the Qwen3.5 paper frozen."""
    required = {
        "Makefile",
        "README.md",
        "main.tex",
        "references.bib",
        "appendices/evidence.tex",
        "sections/conclusion.tex",
        "sections/engineering.tex",
        "sections/introduction.tex",
        "sections/limitations.tex",
        "sections/methodology.tex",
        "sections/related-work.tex",
        "sections/reproducibility.tex",
        "sections/results.tex",
    }
    assert {
        path.relative_to(PAPER_DIR).as_posix()
        for path in PAPER_DIR.rglob("*")
        if path.is_file()
    } == required

    main = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    assert EXPECTED_TITLE in main
    assert r"\author{Libor Burian}" in main
    assert r"\date{September 4, 2026}" in main
    assert FINAL_PDF.read_bytes().startswith(b"%PDF-")

    makefile = (PAPER_DIR / "Makefile").read_text(encoding="utf-8")
    assert "../../paper/build/qwen38-minimal" in makefile
    assert "../../output/pdf/teaching-one-synthetic-fact-qwen38-minimal.pdf" in (
        makefile
    )
    readme = (PAPER_DIR / "README.md").read_text(encoding="utf-8")
    assert "make -C papers/qwen38-minimal" in readme
    assert "output/pdf/teaching-one-synthetic-fact-qwen38-minimal.pdf" in readme
    project_readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "make -C papers/qwen38-minimal" in project_readme
    assert "output/pdf/teaching-one-synthetic-fact-qwen38-minimal.pdf" in (
        project_readme
    )
    agent_policy = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "papers/qwen38-minimal/README.md" in agent_policy
    assert "tests/test_qwen38_paper.py" in agent_policy

    historical_tree = subprocess.run(
        ["git", "rev-parse", "HEAD:paper"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert historical_tree == HISTORICAL_PAPER_TREE
    historical_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "paper"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not historical_status
    historical_pdf = PROJECT_ROOT / "output/pdf/teaching-one-synthetic-fact-qwen35.pdf"
    assert hashlib.sha256(historical_pdf.read_bytes()).hexdigest() == (
        HISTORICAL_PDF_SHA256
    )


def test_qwen38_paper_reconciles_identity_results_cost_and_publication() -> None:
    """Every headline value must agree with the Qwen3.8 manifest and run metadata."""
    source = _paper_source()
    manifest = _json(QWEN38_ROOT / "manifest.json")
    runs = manifest["runs"]
    assert isinstance(runs, list) and len(runs) == 1
    run = runs[0]
    assert isinstance(run, dict)
    metadata = _json(RUN_ROOT / "run-metadata.json")

    assert run["run_id"] == metadata["run_id"] == RUN_ID
    assert run["source_commit"] == RUN_SOURCE_COMMIT
    assert RUN_ID in source
    assert RUN_SOURCE_COMMIT in source
    assert run["scientific_hash"] in source
    assert "Qwen/Qwen3.8-27B" in source
    assert BASE_REVISION in source
    assert r"\PaperScores{0/12}{8/8}{8/8}" in source
    assert r"\PaperScores{11/12}{8/8}{8/8}" in source
    assert "checkpoint-84" in source
    assert "210/210" in source and "15 epochs" in source
    assert "58,363,904" in source and "496" in source and "992" in source
    assert run["provider_cost_usd"] in source
    assert r"\$3.29" in source
    assert "candidate-knowledge-acquisition" in source

    publication = run["publication"]
    assert isinstance(publication, dict)
    assert publication["repo_id"] == ADAPTER_ID
    assert publication["revision"] == ADAPTER_REVISION
    assert publication["collection_slug"] == COLLECTION_SLUG
    assert ADAPTER_ID in source
    assert ADAPTER_REVISION in source
    assert COLLECTION_SLUG in source
    assert f"https://huggingface.co/Qwen/Qwen3.8-27B/tree/{BASE_REVISION}" in source
    assert f"https://huggingface.co/{ADAPTER_ID}/tree/{ADAPTER_REVISION}" in source
    assert f"https://huggingface.co/collections/{COLLECTION_SLUG}" in source
    normalized = " ".join(source.split()).casefold()

    training = metadata["training"]
    assert isinstance(training, dict)
    assert (
        f"{training['completed_optimizer_steps']}/"
        f"{training['planned_optimizer_steps']}" in source
    )
    assert f"{training['epochs']} epochs" in source
    assert f"{training['train_runtime_seconds']:,} seconds" in source
    assert f"{training['train_steps_per_second']} optimizer steps per second" in source
    for key in (
        "target_module_count",
        "trainable_tensor_count",
        "trainable_parameters",
    ):
        assert f"{training[key]:,}" in source

    audit = metadata["pre_optimizer_audit"]
    assert isinstance(audit, dict)
    assert f"{audit['rehearsal_passed']}/{audit['rehearsal_total']} rehearsal" in source
    assert (
        f"{audit['checkpoint_controls_passed']}/"
        f"{audit['checkpoint_controls_total']} checkpoint controls" in source
    )

    hardware = metadata["hardware"]
    assert isinstance(hardware, dict)
    assert hardware["gpu"] in source and f"CUDA {hardware['cuda_runtime']}" in source
    for key in (
        "peak_allocated_memory_bytes",
        "peak_reserved_memory_bytes",
        "peak_sampled_memory_used_mib",
    ):
        assert f"{hardware[key]:,}" in source

    retrieval = metadata["retrieval"]
    assert isinstance(retrieval, dict)
    assert retrieval["archive_sha256"] in source
    supplemental = retrieval["supplemental_checkpoint_archive"]
    assert isinstance(supplemental, dict)
    assert supplemental["archive_sha256"] in source
    adapter_hashes = supplemental["adapter_model_sha256"]
    assert isinstance(adapter_hashes, dict)
    assert adapter_hashes["checkpoint-84"] in source

    infrastructure = metadata["infrastructure"]
    assert isinstance(infrastructure, dict)
    assert infrastructure["pod_status"] == "deleted"
    assert infrastructure["remaining_gpu_dependency"] is False
    assert "no gpu dependency remains" in normalized

    for phrase in (
        "56 training rows",
        "24 positives",
        "16 entity-only contrasts",
        "16 rehearsals",
        "24-row checkpoint",
        "28-row final",
        "16/16 rehearsal",
        "14/16 checkpoint controls",
        "vision, embeddings, and",
        "lm_head",
    ):
        assert phrase in normalized

    evidence_prefix = f"{REPOSITORY_URL}/blob/{EVIDENCE_COMMIT}/"
    for relative_path in (
        "reports/qwen38/manifest.json",
        f"reports/qwen38/runs/{RUN_ID}/billing.json",
        f"reports/qwen38/runs/{RUN_ID}/evaluation.json",
        f"reports/qwen38/runs/{RUN_ID}/publication-final.json",
        f"reports/qwen38/runs/{RUN_ID}/run-metadata.json",
    ):
        assert evidence_prefix + relative_path in source

    for revision in re.findall(
        rf"{re.escape(REPOSITORY_URL)}/(?:blob|tree)/([^/]+)/reports/qwen38/",
        source,
    ):
        assert revision == EVIDENCE_COMMIT
    assert "/blob/main/" not in source and "/tree/main/" not in source


def test_qwen38_checkpoint_table_matches_all_fifteen_saved_evaluations() -> None:
    """The visible trajectory must preserve every epoch score and validation loss."""
    source = _paper_source()
    pattern = re.compile(
        r"\\TrajectoryRow\{(?P<epoch>\d+)\}\{(?P<step>\d+)\}"
        r"\{(?P<recall>\d+/\d+)\}\{(?P<safety>\d+/\d+)\}"
        r"\{(?P<controls>\d+/\d+)\}\{(?P<loss>\d+\.\d+)\}"
        r"\{(?P<selection>[^{}]+)\}"
    )
    observed = [match.groupdict() for match in pattern.finditer(source)]

    evaluation = _json(RUN_ROOT / "evaluation.json")
    provenance = evaluation["provenance"]
    assert isinstance(provenance, dict)
    training = provenance["training"]
    assert isinstance(training, dict)
    history = training["behavioral_validation_history"]
    assert isinstance(history, list) and len(history) == 15

    expected: list[dict[str, str]] = []
    for entry in history:
        assert isinstance(entry, dict)
        summary = entry["summary"]
        assert isinstance(summary, dict)

        expected.append(
            {
                "epoch": f"{entry['epoch']:g}",
                "step": str(entry["step"]),
                "recall": _score_ratio(summary, "fact_recall"),
                "safety": _score_ratio(summary, "near_name_negative"),
                "controls": _score_ratio(summary, "common_knowledge"),
                "loss": f"{entry['eval_loss']:.9f}",
                "selection": "selected" if entry["step"] == 84 else "--",
            }
        )
    assert observed == expected


def test_qwen38_paper_sources_are_closed_pinned_and_cautious() -> None:
    """The derived manuscript must expose support and avoid inflating one case study."""
    source = _paper_source()
    bibliography = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    assert _citation_keys(source) == _bibliography_keys(bibliography)
    assert _citation_keys(source)

    used_sources = re.findall(r"\\claimsource\{([^{}]+)\}", source)
    defined_sources = re.findall(r"\\sourceentry\{([^{}]+)\}", source)
    assert set(used_sources) == set(defined_sources)
    assert len(defined_sources) == len(set(defined_sources))

    normalized = " ".join(source.split()).casefold()
    for phrase in (
        "one seed",
        "training-disjoint regression suite",
        "not a pristine research holdout",
        "single-prompt smoke test",
        "two-token non-generative forward probe",
        "expanded bf16 and qlora rungs were not run",
        "do not constitute independent peer review",
    ):
        assert phrase in normalized
    for unsupported in (
        "proves knowledge acquisition",
        "perfect generalization",
        "caused the improvement",
        "optimal configuration",
        "every training call used both kernels",
    ):
        assert unsupported not in normalized

    public_text = source + "\n" + bibliography
    assert "/home/" not in public_text and "/mnt/" not in public_text
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", public_text)
    assert not re.search(r'(?i)"(?:hf_)?token"\s*:', public_text)
    for revision in re.findall(
        r"https://github\.com/[^/]+/[^/]+/(?:blob|tree)/([^/?#{}]+)",
        public_text,
    ):
        assert re.fullmatch(r"[0-9a-f]{40}", revision), revision
