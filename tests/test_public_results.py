"""Global context: keep every public experiment result complete and auditable."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from fact_teaching.evaluation import (
    EvaluationResult,
    evaluate_acceptance,
    score_generation,
)
from fact_teaching.reporting import _render_markdown_report

# Resolve paths from this test file so checks do not depend on the caller's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The manifest is the public index that binds run identities to immutable evidence.
MANIFEST_PATH = PROJECT_ROOT / "reports" / "manifest.json"
# Concise narrative reports live separately from complete generated evaluations.
RUN_REPORT_DIR = PROJECT_ROOT / "reports" / "runs"
# The paper is a derived view whose source and named PDF remain public artifacts.
PAPER_DIR = PROJECT_ROOT / "paper"
FINAL_PDF_PATH = (
    PROJECT_ROOT / "output" / "pdf" / "teaching-one-synthetic-fact-qwen35.pdf"
)
EXPECTED_PAPER_TITLE = (
    "Teaching One Synthetic Fact to Qwen3.5-0.8B: "
    "A Sequential Study of LoRA Recall, Specificity, and Retention"
)
# Paper evidence links must be clickable bindings to the canonical public files.
PAPER_EVIDENCE_URL_PREFIX = (
    "https://github.com/BurnyCoder/training-facts-into-llms/blob/"
    "ca83803ccdf46486d38fd7161b155cc20560c449/"
)
# These are the six historical attempts plus the reviewed three-attempt ladder.
EXPECTED_ATTEMPT_NAMES = {
    "primary",
    "conservative",
    "expanded",
    "paper_single_edit",
    "semantic_specificity",
    "semantic_specificity_gentle",
    "minimal_pair_primary",
    "minimal_pair_conservative",
    "minimal_pair_expanded",
}
# Status values are deliberately closed so public results cannot be ambiguous.
ALLOWED_STATUSES = {
    "completed_failed_acceptance",
    "completed_passed_published",
    "interrupted_no_post_training_evaluation",
}
# A public adapter must contain exactly the locally validated upload payload.
EXPECTED_PUBLICATION_PAYLOAD = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
    "evaluation.json",
    "processor_reference.json",
}


def _load_manifest() -> dict[str, Any]:
    """Load the checked-in result index with standard-library JSON parsing."""
    # UTF-8 and strict JSON parsing match how the repository writes public evidence.
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    # A mapping is required before tests inspect the versioned manifest contract.
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one public evidence file."""
    # Reading bytes avoids platform newline conversion changing the verified digest.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attempts_by_name() -> dict[str, dict[str, Any]]:
    """Return attempts keyed by their unique stable public names."""
    # Every test uses the same parsed manifest contract.
    attempts = _load_manifest()["attempts"]
    # The manifest must retain its deterministic JSON array representation.
    assert isinstance(attempts, list)
    # Build the index only after the dedicated uniqueness test checks collisions.
    return {attempt["name"]: attempt for attempt in attempts}


def _paper_source() -> str:
    """Return every tracked TeX source in deterministic path order."""
    # The paper is modular, so drift checks must inspect every included source.
    tex_paths = sorted(PAPER_DIR.rglob("*.tex"))
    assert tex_paths, "paper must contain modular TeX sources"
    return "\n".join(path.read_text(encoding="utf-8") for path in tex_paths)


def test_manifest_v1_indexes_each_expected_attempt_once() -> None:
    """The public index must name all nine initiated runs without collisions."""
    # Schema v1 is the only result-manifest format implemented by this repository.
    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    # File digests are meaningful only under the declared SHA-256 algorithm.
    assert manifest["hash_algorithm"] == "sha256"
    # Preserve list order for human review while checking set-level completeness.
    attempts = manifest["attempts"]
    assert isinstance(attempts, list)
    # Each attempt needs a non-empty stable name and run ID.
    names = [attempt["name"] for attempt in attempts]
    run_ids = [attempt["run_id"] for attempt in attempts]
    assert all(isinstance(name, str) and name for name in names)
    assert all(isinstance(run_id, str) and run_id for run_id in run_ids)
    # Duplicate names or run IDs could make evidence resolve to the wrong attempt.
    assert len(names) == len(set(names))
    assert len(run_ids) == len(set(run_ids))
    # This exact set catches an initiated run that was omitted from public results.
    assert set(names) == EXPECTED_ATTEMPT_NAMES
    # Unknown status strings are forbidden because their publication meaning is unclear.
    assert {attempt["status"] for attempt in attempts} <= ALLOWED_STATUSES


def test_each_manifest_attempt_has_exactly_one_concise_run_report() -> None:
    """Every attempt must have one short narrative report named after the attempt."""
    # A one-to-one filename convention makes missing and orphaned reports detectable.
    attempts = _attempts_by_name()
    expected_paths = {
        RUN_REPORT_DIR / f"{attempt_name}.md" for attempt_name in attempts
    }
    # Only direct Markdown children are part of the concise run-report collection.
    actual_paths = set(RUN_REPORT_DIR.glob("*.md"))
    assert actual_paths == expected_paths

    # Each report must identify its exact run while remaining a concise index document.
    for attempt_name, attempt in attempts.items():
        report_path = RUN_REPORT_DIR / f"{attempt_name}.md"
        report_text = report_path.read_text(encoding="utf-8")
        assert attempt["run_id"] in report_text
        assert 1 <= len(report_text.splitlines()) <= 120
        assert len(report_text.encode("utf-8")) <= 12_000


def test_nine_run_reports_reconcile_public_identity_results_and_artifacts() -> None:
    """Run narratives must preserve manifest identity and hash-bound result facts."""
    checked_reports = 0
    for attempt_name, attempt in _attempts_by_name().items():
        report = (RUN_REPORT_DIR / f"{attempt_name}.md").read_text(encoding="utf-8")
        compact_report = " ".join(report.split())
        numeric_report = compact_report.replace(",", "")
        assert report.count(attempt["run_id"]) == 1
        source_commit = attempt["source"]["commit"]
        assert FULL_GIT_SHA.fullmatch(source_commit)
        historical_repository = attempt["source"]["github_repository"]
        assert f"https://github.com/{historical_repository}/commit/{source_commit}" in report
        assert "Final publishable adapter saved | No" in report
        assert "Hub publication attempted | No" in report

        result = attempt["result"]
        for score in result["baseline"].values():
            assert score in report
        post_training = result["post_training"]
        if post_training is None:
            assert "Interrupted; no post-training evaluation" in report
            progress = attempt["training_progress"]
            for value in (
                progress["completed_optimizer_steps"],
                progress["planned_optimizer_steps"],
                progress["last_completed_epoch"],
            ):
                assert f"{value:g}" in numeric_report
            assert "acceptance gate was never evaluated" in compact_report
        else:
            assert "Completed; failed acceptance" in report
            for score in (
                post_training["fact_recall"],
                post_training["near_name_safety"],
                post_training["common_knowledge"],
            ):
                assert score in report
            evaluation_path = PROJECT_ROOT / _evaluation_path(attempt)
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            training = evaluation["provenance"]["training"]
            runtime = training["metrics"]["train_runtime"]
            _assert_number_in_text(numeric_report, runtime)
            _assert_number_in_text(numeric_report, training["global_step"])
            best_checkpoint = training["best_checkpoint"]
            mentioned_checkpoints = set(re.findall(r"checkpoint-\d+", report))
            if mentioned_checkpoints:
                assert best_checkpoint is not None
                assert mentioned_checkpoints == {best_checkpoint}
            progress = attempt.get("training_progress")
            if progress and "selected_epoch" in progress:
                _assert_number_in_text(report, progress["selected_epoch"])
                selected_step = progress.get("selected_optimizer_step")
                if selected_step is None:
                    assert best_checkpoint is not None
                    selected_step = int(best_checkpoint.removeprefix("checkpoint-"))
                _assert_number_in_text(report, selected_step)
            false_positive_ids = evaluation["acceptance"]["false_positive_ids"]
            if false_positive_ids == [f"negative_{index:03d}" for index in range(1, 9)]:
                assert "`negative_001` through `negative_008`" in report
            else:
                for record_id in false_positive_ids:
                    assert record_id in report
            for record_id in evaluation["acceptance"]["lost_control_ids"]:
                assert record_id in report
            for item in attempt["report_files"]:
                assert f"../{Path(item['path']).name}" in report
        checked_reports += 1
    assert checked_reports == 9


def test_manifest_report_pairs_exist_and_match_their_sha256_digests() -> None:
    """Each indexed evaluation pair must exist, share a stem, and match its digest."""
    # Hash verification binds the manifest entry to the exact reviewed file bytes.
    for attempt in _attempts_by_name().values():
        report_files = attempt.get("report_files", [])
        assert isinstance(report_files, list)
        # An interrupted attempt may lack final evaluations; otherwise pairs are exact.
        assert len(report_files) in {0, 2}
        if not report_files:
            continue

        # One structured JSON source and one rendered Markdown view form a pair.
        relative_paths = [Path(entry["path"]) for entry in report_files]
        assert {path.suffix for path in relative_paths} == {".json", ".md"}
        assert len({path.with_suffix("") for path in relative_paths}) == 1
        for entry, relative_path in zip(report_files, relative_paths, strict=True):
            # Evidence paths must remain below reports/ and cannot escape the project.
            assert not relative_path.is_absolute()
            assert relative_path.parts[:1] == ("reports",)
            assert ".." not in relative_path.parts
            # The declared digest is lowercase SHA-256, not an arbitrary checksum.
            declared_digest = entry["sha256"]
            assert isinstance(declared_digest, str)
            assert len(declared_digest) == 64
            assert declared_digest == declared_digest.lower()
            int(declared_digest, 16)
            # Verify the full public file rather than trusting manifest metadata alone.
            evidence_path = PROJECT_ROOT / relative_path
            assert evidence_path.is_file()
            assert _sha256(evidence_path) == declared_digest


def test_every_generated_evaluation_pair_is_indexed_once() -> None:
    """Generated JSON/Markdown evaluations must be paired and owned by one attempt."""
    # Discover all checked-in or pending generated evaluation sources and renderings.
    generated_json = set((PROJECT_ROOT / "reports").glob("evaluation-*.json"))
    generated_markdown = set((PROJECT_ROOT / "reports").glob("evaluation-*.md"))
    # A structured source without its rendering, or vice versa, is incomplete evidence.
    assert {path.stem for path in generated_json} == {
        path.stem for path in generated_markdown
    }
    generated_paths = generated_json | generated_markdown

    # Flatten manifest ownership while preserving duplicates for collision detection.
    indexed_paths = [
        PROJECT_ROOT / entry["path"]
        for attempt in _attempts_by_name().values()
        for entry in attempt.get("report_files", [])
    ]
    # One generated file cannot be claimed by multiple attempts.
    assert len(indexed_paths) == len(set(indexed_paths))
    # This equality catches both unindexed generated output and stale manifest entries.
    assert set(indexed_paths) == generated_paths


def test_every_generated_markdown_report_exactly_renders_its_json_source() -> None:
    """All eight human views must be byte-exact renderings of structured evidence."""
    # The reporting module owns one renderer so JSON and Markdown cannot drift silently.
    rendered_pairs = 0
    for attempt in _attempts_by_name().values():
        report_files = attempt.get("report_files", [])
        if not report_files:
            continue
        # The manifest contract already guarantees one JSON and one Markdown file.
        paths_by_suffix = {
            Path(item["path"]).suffix: PROJECT_ROOT / item["path"]
            for item in report_files
        }
        payload = json.loads(paths_by_suffix[".json"].read_text(encoding="utf-8"))
        expected_markdown = _render_markdown_report(payload)
        actual_markdown = paths_by_suffix[".md"].read_text(encoding="utf-8")
        assert actual_markdown == expected_markdown
        rendered_pairs += 1
    assert rendered_pairs == 8


def test_all_448_saved_generations_recompute_exact_scores_and_acceptance() -> None:
    """Re-score every saved baseline/tuned output against the current 28-row suite."""
    # The fixed regression data supplies category rules and accepted control aliases.
    evaluation_rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / "data" / "eval.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    rows_by_id = {row["id"]: row for row in evaluation_rows}
    assert len(evaluation_rows) == len(rows_by_id) == 28

    scored_record_count = 0
    completed_attempt_count = 0
    for attempt in _attempts_by_name().values():
        if not attempt.get("report_files"):
            continue
        completed_attempt_count += 1
        payload = json.loads(
            (PROJECT_ROOT / _evaluation_path(attempt)).read_text(encoding="utf-8")
        )
        recomputed_results: dict[str, EvaluationResult] = {}
        for stage_key in ("baseline", "post_training"):
            saved_stage = payload["evaluations"][stage_key]
            assert saved_stage["stage"] == stage_key
            saved_records = saved_stage["records"]
            assert [record["record_id"] for record in saved_records] == [
                row["id"] for row in evaluation_rows
            ]
            recomputed_records = [
                score_generation(
                    rows_by_id[saved_record["record_id"]],
                    saved_record["output"],
                )
                for saved_record in saved_records
            ]
            recomputed = EvaluationResult(stage=stage_key, records=recomputed_records)
            assert recomputed.to_dict() == saved_stage
            recomputed_results[stage_key] = recomputed
            scored_record_count += len(recomputed_records)

        # JSON round-tripping converts the decision's immutable ID tuples to JSON arrays.
        recomputed_acceptance = json.loads(
            json.dumps(
                evaluate_acceptance(
                    recomputed_results["baseline"],
                    recomputed_results["post_training"],
                ).to_dict()
            )
        )
        assert recomputed_acceptance == payload["acceptance"]
        assert recomputed_acceptance["passed"] is attempt["result"][
            "acceptance_passed"
        ]
        assert recomputed_acceptance["false_positive_ids"] == attempt["result"].get(
            "false_positive_ids", recomputed_acceptance["false_positive_ids"]
        )
        assert recomputed_acceptance["lost_control_ids"] == attempt["result"].get(
            "lost_control_ids", recomputed_acceptance["lost_control_ids"]
        )

    assert completed_attempt_count == 8
    assert scored_record_count == 448


def test_passing_or_public_attempts_prove_complete_publication() -> None:
    """Any success claim must include adapter save, publication, and anonymous proof."""
    # Failed and interrupted attempts must never retain a publishable adapter.
    for attempt in _attempts_by_name().values():
        result = attempt["result"]
        acceptance_passed = result.get("acceptance_passed") is True
        adapter_saved = result.get("adapter_saved") is True
        publication_attempted = result.get("publication_attempted") is True
        claims_public_success = (
            attempt["status"] == "completed_passed_published"
            or acceptance_passed
            or adapter_saved
            or publication_attempted
        )
        if not claims_public_success:
            assert adapter_saved is False
            assert publication_attempted is False
            continue

        # Passing, saving, and publication are one atomic public-success contract.
        assert attempt["status"] == "completed_passed_published"
        assert acceptance_passed is True
        assert adapter_saved is True
        assert publication_attempted is True
        # Hub visibility alone is insufficient; a credential-free reload must pass.
        publication = attempt["publication"]
        assert publication["repository"] == (
            "BurnyCoder/qwen3.5-0.8b-atemokoloporos-lora"
        )
        assert publication["public"] is True
        assert set(publication["remote_payload_files"]) == EXPECTED_PUBLICATION_PAYLOAD
        anonymous = publication["anonymous_verification"]
        assert anonymous["process_status"] == 0
        assert anonymous["passed"] is True
        assert anonymous["credential_free"] is True


def test_paper_has_reproducible_source_and_named_pdf() -> None:
    """Require the selected author, build interface, bibliography, and PDF."""
    main_path = PAPER_DIR / "main.tex"
    assert main_path.is_file()
    main_text = main_path.read_text(encoding="utf-8")
    assert EXPECTED_PAPER_TITLE in main_text
    assert r"\author{Libor Burian}" in main_text
    assert r"\date{August 3, 2026}" in main_text
    assert (PAPER_DIR / "references.bib").is_file()
    assert (PAPER_DIR / "Makefile").is_file()
    assert (PAPER_DIR / "README.md").is_file()
    assert FINAL_PDF_PATH.read_bytes().startswith(b"%PDF-")


def test_paper_binds_every_attempt_to_manifest_results() -> None:
    """Require each run ID once and its exact result beside that binding."""
    source = (PAPER_DIR / "appendices" / "evidence.tex").read_text(encoding="utf-8")
    # Scope identity and score checks to the run table, not later ledger prose.
    ledger_end = source.index(r"\end{longtable}")
    source = source[:ledger_end]
    attempts = _load_manifest()["attempts"]
    assert len(attempts) == 9
    completed_attempt_count = 0

    for attempt in attempts:
        run_id = attempt["run_id"]
        assert source.count(run_id) == 1, f"paper must bind {run_id} exactly once"
        row_start = source.index(run_id)
        row_context = source[row_start : row_start + 900]
        post_training = attempt["result"]["post_training"]
        if post_training is None:
            assert "Inconclusive" in row_context
            assert r"\PaperProgress{125}{180}" in row_context
            continue
        completed_attempt_count += 1
        expected_score_macro = (
            f"\\PaperScores{{{post_training['fact_recall']}}}"
            f"{{{post_training['near_name_safety']}}}"
            f"{{{post_training['common_knowledge']}}}"
        )
        assert expected_score_macro in row_context, (
            f"{run_id} is missing ordered score triple {expected_score_macro}"
        )

    assert completed_attempt_count == 8


def test_paper_links_every_public_evidence_file() -> None:
    """Require all nine run reports and all manifest-owned evaluation files."""
    source = _paper_source()
    href_urls = set(re.findall(r"\\href\{([^{}]+)\}", source))
    manifest = _load_manifest()
    evaluation_paths = {
        report["path"]
        for attempt in manifest["attempts"]
        for report in attempt["report_files"]
    }
    run_report_paths = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in RUN_REPORT_DIR.glob("*.md")
    }
    assert len(evaluation_paths) == 16
    assert len(run_report_paths) == 9
    for evidence_path in evaluation_paths | run_report_paths:
        expected_url = f"{PAPER_EVIDENCE_URL_PREFIX}{evidence_path}"
        assert expected_url in href_urls, f"missing evidence hyperlink {evidence_path}"


def test_paper_preserves_the_negative_result_and_corrected_claims() -> None:
    """Reject success inflation, credential text, and provenance errors."""
    source = _paper_source()
    normalized = " ".join(source.split()).lower()
    required_claims = (
        "nine training attempts",
        "eight completed",
        "zero accepted",
        "no final adapter was exported",
        "no upload to hugging face was attempted",
        "regression suite",
        "full model parameters directly to adamw",
        "completion-control tokens",
    )
    for claim in required_claims:
        assert claim in normalized

    forbidden_claims = (
        "black-box peft/lora",
        "only the object tokens",
        "did not need to relearn the question or entity text",
        "successfully taught",
        "accepted adapter",
    )
    for claim in forbidden_claims:
        assert claim not in normalized

    assert "/home/" not in source
    assert "/mnt/" not in source
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source)


# Experiment-retrospective source contract. Keeping these CPU checks in this
# already Git-gated public-results module avoids expanding the production gate.
# All evidence is resolved relative to the repository, independent of pytest's CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "EXPERIMENTS.md"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "manifest.json"
EVIDENCE_COMMIT = "ca83803ccdf46486d38fd7161b155cc20560c449"
REPOSITORY_URL = "https://github.com/BurnyCoder/training-facts-into-llms"
LEDGER_HEADING = "## Claim-source ledger"

# A visible marker and its Markdown reference use the same stable identifier.
SOURCE_ID_PATTERN = r"[a-z0-9][a-z0-9._:-]*"
MARKER_RE = re.compile(
    rf"\[(?P<kind>[SA]):(?P<id>{SOURCE_ID_PATTERN})\]"
    rf"\[src-(?P=id)\]"
)
REFERENCE_RE = re.compile(
    rf"(?m)^\[src-(?P<id>{SOURCE_ID_PATTERN})\]:\s*"
    r"(?P<target><[^>]+>|\S+)\s*$"
)
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# The 21 excerpts are the exact representative set already audited in the paper.
EXCERPT_RECORD_GROUPS = {
    "primary": (
        ("fact_001",),
        ("negative_001",),
        ("control_001",),
    ),
    "paper_single_edit": (
        ("fact_002",),
        ("fact_005",),
        ("fact_012",),
        ("negative_002",),
    ),
    "semantic_specificity": (
        ("fact_001",),
        ("fact_009",),
        ("control_002",),
    ),
    "semantic_specificity_gentle": (("fact_002", "fact_012"),),
    "minimal_pair_primary": (
        ("negative_003",),
        ("control_002",),
        ("control_006",),
        ("control_007",),
    ),
    "minimal_pair_conservative": (
        ("control_002",),
        ("control_006",),
        ("control_007",),
    ),
    "minimal_pair_expanded": (
        ("fact_006",),
        ("control_006",),
        ("control_007",),
    ),
}
EVAL_SOURCE_BY_ATTEMPT = {
    "primary": "eval-positive-primary",
    "paper_single_edit": "eval-paper",
    "semantic_specificity": "eval-semantic-standard",
    "semantic_specificity_gentle": "eval-semantic-gentle",
    "minimal_pair_primary": "eval-minimal-primary",
    "minimal_pair_conservative": "eval-minimal-conservative",
    "minimal_pair_expanded": "eval-minimal-expanded",
}


def _load_manifest() -> dict[str, Any]:
    """Load the canonical structured experiment index."""
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _report() -> str:
    """Return the UTF-8 retrospective without touching ignored evidence."""
    return REPORT_PATH.read_text(encoding="utf-8")


def _artifact_url(path: str, commit: str = EVIDENCE_COMMIT) -> str:
    """Build one content-addressed link to a repository file."""
    return f"{REPOSITORY_URL}/blob/{commit}/{path}"


def _section(text: str, heading: str) -> str:
    """Return one Markdown section through the next heading at the same level."""
    start = text.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    match = re.search(rf"(?m)^#{{1,{level}}}\s", text[start + len(heading) :])
    end = len(text) if match is None else start + len(heading) + match.start()
    return text[start:end]


def _table_rows(section: str) -> list[str]:
    """Return data rows from every pipe table in a bounded section."""
    tables: list[list[str]] = []
    current: list[str] = []
    for line in section.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)

    rows: list[str] = []
    for table in tables:
        assert len(table) >= 2
        assert re.fullmatch(r"[\s|:-]+", table[1])
        rows.extend(table[2:])
    return rows


def _cells(row: str) -> list[str]:
    """Split a controlled Markdown table row into stripped cells."""
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _ledger(text: str) -> dict[str, dict[str, str]]:
    """Parse the required five-column source ledger."""
    ledger_section = _section(text, LEDGER_HEADING)
    lines = [line for line in ledger_section.splitlines() if line.startswith("|")]
    assert len(lines) >= 3, "claim-source ledger must contain a Markdown table"
    headers = [cell.casefold() for cell in _cells(lines[0])]
    assert headers == [
        "identifier",
        "source class",
        "supported claim scope",
        "locator",
        "limitation",
    ]
    assert re.fullmatch(r"[\s|:-]+", lines[1])

    entries: dict[str, dict[str, str]] = {}
    for row in lines[2:]:
        cells = _cells(row)
        assert len(cells) == 5
        identity = cells[0].strip("`")
        match = re.fullmatch(rf"([SA]):({SOURCE_ID_PATTERN})", identity)
        assert match, f"invalid ledger identifier {identity!r}"
        kind, source_id = match.groups()
        assert source_id not in entries, f"duplicate ledger ID {source_id}"
        assert all(cell for cell in cells[1:])
        assert f"[src-{source_id}]" in cells[3]
        entries[source_id] = {
            "kind": kind,
            "class": cells[1],
            "scope": cells[2],
            "locator": cells[3],
            "limitation": cells[4],
        }
    return entries


def _references(text: str) -> dict[str, str]:
    """Parse source-reference definitions and reject duplicates."""
    matches = list(REFERENCE_RE.finditer(text))
    ids = [match.group("id") for match in matches]
    assert len(ids) == len(set(ids)), "source reference definitions must be unique"
    return {
        match.group("id"): match.group("target").strip("<>") for match in matches
    }


def _resolved(row: str, ledger: dict[str, dict[str, str]], refs: dict[str, str]) -> str:
    """Append the ledger fields and targets named by a row's markers."""
    result = [row]
    for match in MARKER_RE.finditer(row):
        source_id = match.group("id")
        result.extend(ledger[source_id].values())
        result.append(refs[source_id])
    return "\n".join(result)


def _visible_word_count(text: str) -> int:
    """Approximate prose length after removing links and source markers."""
    text = MARKER_RE.sub("", text)
    text = re.sub(r"!?\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[`*_>#~|]", " ", text)
    return len(re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text))


def _assert_number_in_text(text: str, value: float) -> None:
    """Require one exact numeric value while allowing comma grouping and integer floats."""
    # Markdown reports use comma grouping for runtimes but JSON stores plain numbers.
    normalized = text.replace(",", "")
    forms = {str(value)}
    if isinstance(value, float) and value.is_integer():
        forms.add(str(int(value)))
    assert any(
        re.search(rf"(?<![0-9.]){re.escape(form)}(?![0-9.])", normalized)
        for form in forms
    ), f"missing exact numeric value {value!r}"


def _previous_evidence_line(lines: list[str], fence_start: int) -> str:
    """Return the closest nonblank line before a fenced evidence block."""
    cursor = fence_start - 1
    while cursor >= 0 and not lines[cursor].strip():
        cursor -= 1
    assert cursor >= 0
    return lines[cursor].strip()


def _evaluation_path(attempt: dict[str, Any]) -> str:
    """Return the sole JSON evaluation owned by a completed run."""
    paths = [
        item["path"]
        for item in attempt["report_files"]
        if item["path"].endswith(".json")
    ]
    assert len(paths) == 1
    return paths[0]


def test_source_markers_ledger_and_references_form_one_closed_system() -> None:
    """Every visible public/author source must resolve exactly once."""
    text = _report()
    assert LEDGER_HEADING in text
    ledger = _ledger(text)
    references = _references(text)
    markers = list(MARKER_RE.finditer(text[: text.index(LEDGER_HEADING)]))
    assert markers, "EXPERIMENTS.md must contain visible source markers"
    marker_ids = {marker.group("id") for marker in markers}
    assert marker_ids == set(ledger) == set(references)

    kinds_by_id: dict[str, set[str]] = {}
    for marker in markers:
        kinds_by_id.setdefault(marker.group("id"), set()).add(marker.group("kind"))
    assert all(len(kinds) == 1 for kinds in kinds_by_id.values())
    for source_id, entry in ledger.items():
        assert kinds_by_id[source_id] == {entry["kind"]}
        target = references[source_id]
        if entry["kind"] == "A":
            limitation = entry["limitation"].casefold()
            assert any(word in limitation for word in ("non-public", "private"))
            assert target == "#claim-source-ledger" or target.startswith("https://")
        else:
            assert target.startswith("https://")


def test_every_substantive_markdown_element_has_adjacent_provenance() -> None:
    """Prose, list items, rows, and fences carry support in their own block."""
    text = _report()
    scoped = text[: text.index(LEDGER_HEADING)]
    tokens = MarkdownIt("commonmark").enable("table").parse(scoped)
    lines = scoped.splitlines()
    missing: list[str] = []
    in_table_header = False

    for token in tokens:
        if token.type == "thead_open":
            in_table_header = True
            continue
        if token.type == "thead_close":
            in_table_header = False
            continue
        if token.type == "paragraph_open" and token.map is not None:
            block = "\n".join(lines[token.map[0] : token.map[1]])
            if _visible_word_count(block) >= 4 and MARKER_RE.search(block) is None:
                missing.append(f"paragraph line {token.map[0] + 1}: {block[:90]}")
        if token.type == "tr_open" and token.map is not None and not in_table_header:
            row = "\n".join(lines[token.map[0] : token.map[1]])
            if MARKER_RE.search(row) is None:
                missing.append(f"table row line {token.map[0] + 1}: {row[:90]}")
        if token.type == "fence" and token.map is not None:
            evidence = _previous_evidence_line(lines, token.map[0])
            if not evidence.startswith("Evidence:") or MARKER_RE.search(evidence) is None:
                missing.append(f"fence line {token.map[0] + 1}: missing Evidence marker")
    assert not missing, "substantive Markdown lacks adjacent sources:\n" + "\n".join(
        missing
    )


def test_source_links_are_content_addressed_and_version_pinned() -> None:
    """Evidence references cannot silently move after review."""
    text = _report()
    references = _references(text)
    inline_targets = set(re.findall(r"\]\((https://[^)\s]+)\)", text))
    github_revision = re.compile(
        r"^https://github\.com/([^/]+)/([^/]+)/(blob|tree|commit)/([^/?#]+)"
    )
    failures: list[str] = []
    for target in set(references.values()) | inline_targets:
        if target == "#claim-source-ledger":
            continue
        match = github_revision.match(target)
        if match is not None:
            owner, repository, link_kind, revision = match.groups()
            if FULL_GIT_SHA.fullmatch(revision) is None:
                failures.append(target)
            if owner == "BurnyCoder" and repository == "training-facts-into-llms":
                if link_kind not in {"blob", "commit"}:
                    failures.append(target)
                if "/reports/" in target and revision != EVIDENCE_COMMIT:
                    failures.append(target)
        if re.search(r"huggingface\.co/docs/[^/]+/(?:main|en)(?:/|$)", target):
            failures.append(target)
    assert not failures, "mutable source references:\n" + "\n".join(
        sorted(set(failures))
    )
    assert not re.search(r"github\.com/[^\s)]+/(?:blob|tree)/(?:main|master)/", text)


def test_timeline_and_artifact_links_match_every_manifest_attempt() -> None:
    """The nine run rows bind exact results, evidence, code, and publication state."""
    text = _report()
    manifest = _load_manifest()
    ledger = _ledger(text)
    references = _references(text)
    timeline = _section(text, "## Exact run timeline")
    rows = _table_rows(timeline)
    assert len(rows) == 9

    for attempt in manifest["attempts"]:
        run_id = attempt["run_id"]
        matches = [row for row in rows if run_id in row]
        assert len(matches) == 1, f"timeline must contain {run_id} exactly once"
        row = matches[0]
        resolved = _resolved(row, ledger, references)
        assert _artifact_url("reports/manifest.json") in resolved
        assert _artifact_url(f"reports/runs/{attempt['name']}.md") in resolved
        source_commit = attempt["source"]["commit"]
        assert re.search(
            rf"{re.escape(REPOSITORY_URL)}/blob/{source_commit}/[^\s)]+", resolved
        )
        post = attempt["result"]["post_training"]
        if post is None:
            assert "Baseline only" in row and "inconclusive" in row.casefold()
            assert "[A:task-history][src-task-history]" in row
            progress = attempt["training_progress"]
            for key in ("completed_optimizer_steps", "planned_optimizer_steps"):
                _assert_number_in_text(row, progress[key])
            _assert_number_in_text(row, progress["last_completed_epoch"])
            assert re.search(
                r"Trainer runtime (?:unavailable|not available|not recorded)",
                row,
                flags=re.IGNORECASE,
            )
        else:
            for score in (
                post["fact_recall"],
                post["near_name_safety"],
                post["common_knowledge"],
            ):
                assert score in row
            evaluation_path = _evaluation_path(attempt)
            assert _artifact_url(evaluation_path) in resolved
            evaluation = json.loads(
                (PROJECT_ROOT / evaluation_path).read_text(encoding="utf-8")
            )
            training = evaluation["provenance"]["training"]
            runtime = training["metrics"]["train_runtime"]
            assert "Trainer runtime" in row
            _assert_number_in_text(row, runtime)

            best_checkpoint = training["best_checkpoint"]
            if best_checkpoint is None:
                # The paper-family run deliberately selected its final weights.
                assert training["selection_policy"] == "final_epoch"
                assert "final" in row.casefold()
                _assert_number_in_text(row, training["global_step"])
                _assert_number_in_text(row, training["metrics"]["epoch"])
            else:
                assert best_checkpoint in row
                selected_step = int(best_checkpoint.removeprefix("checkpoint-"))
                _assert_number_in_text(row, selected_step)
                validation_history = training.get("behavioral_validation_history")
                if validation_history:
                    selected = [
                        item for item in validation_history if item["step"] == selected_step
                    ]
                    assert len(selected) == 1
                    selected_record = selected[0]
                    _assert_number_in_text(row, selected_record["epoch"])
                    _assert_number_in_text(row, selected_record["behavior_score"])
                    if "eval_loss" in selected_record:
                        _assert_number_in_text(row, selected_record["eval_loss"])
                    if "selection_score" in selected_record:
                        _assert_number_in_text(row, selected_record["selection_score"])
                else:
                    selected = [
                        item
                        for item in training["log_history"]
                        if item.get("step") == selected_step
                        and item.get("eval_loss") == training["best_metric"]
                    ]
                    assert len(selected) == 1
                    _assert_number_in_text(row, selected[0]["epoch"])
                    _assert_number_in_text(row, selected[0]["eval_loss"])
        adapter_saved = attempt["result"]["adapter_saved"]
        publication_attempted = attempt["result"]["publication_attempted"]
        expected_publication = (
            f"{'Yes' if adapter_saved else 'No'} / "
            f"{'yes' if publication_attempted else 'no'}"
        )
        assert expected_publication in row

    expected_paths = {
        item["path"]
        for attempt in manifest["attempts"]
        for item in attempt["report_files"]
    }
    expected_paths |= {
        f"reports/runs/{attempt['name']}.md" for attempt in manifest["attempts"]
    }
    assert len(expected_paths) == 25
    for path in expected_paths:
        assert _artifact_url(path) in text, f"missing immutable evidence link {path}"


def test_manifest_hashes_and_all_historical_data_bindings_are_present() -> None:
    """The report exposes every digest and all 37 attempt-specific data bindings."""
    text = _report()
    manifest = _load_manifest()
    digest_counts: Counter[str] = Counter()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "sha256":
                    digest_counts[child] += 1
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(manifest)
    for digest, count in digest_counts.items():
        assert text.count(digest) >= count, f"missing manifest SHA-256 {digest}"

    data_rows = _table_rows(_section(text, "### Historical data bindings"))
    expected = [
        (attempt["name"], attempt["source"]["commit"], item)
        for attempt in manifest["attempts"]
        for item in attempt["data_files"]
    ]
    assert len(expected) == len(data_rows) == 37
    used_rows: set[int] = set()
    for attempt_name, commit, item in expected:
        url = _artifact_url(item["path"], commit)
        candidates = [
            index
            for index, row in enumerate(data_rows)
            if index not in used_rows
            and _cells(row)[0].strip("`") == attempt_name
            and item["path"] in row
            and item["sha256"] in row
            and url in _resolved(row, _ledger(text), _references(text))
        ]
        assert len(candidates) == 1, f"missing data binding {attempt_name}:{item['path']}"
        used_rows.add(candidates[0])

    for attempt in manifest["attempts"]:
        for item in attempt["report_files"]:
            path = PROJECT_ROOT / item["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

    artifact_rows = _table_rows(
        _section(text, "### Run reports, evaluation pairs, and manifest hashes")
    )
    assert len(artifact_rows) == len(manifest["attempts"]) == 9
    for attempt in manifest["attempts"]:
        rows = [row for row in artifact_rows if attempt["run_id"] in row]
        assert len(rows) == 1
        row = rows[0]
        cells = _cells(row)
        assert len(cells) == 6
        report_cell, run_id_cell, log_cell, json_cell, markdown_cell, _ = cells
        report_url = _artifact_url(f"reports/runs/{attempt['name']}.md")
        assert report_url in report_cell
        assert sum(report_url in cell for cell in cells) == 1
        assert run_id_cell == f"`{attempt['run_id']}`"
        log_digest = attempt["operational_log"]["sha256"]
        assert log_digest in log_cell
        assert sum(log_digest in cell for cell in cells) == 1
        if attempt["report_files"]:
            items_by_suffix = {
                Path(item["path"]).suffix: item for item in attempt["report_files"]
            }
            assert set(items_by_suffix) == {".json", ".md"}
            for suffix, cell in ((".json", json_cell), (".md", markdown_cell)):
                item = items_by_suffix[suffix]
                artifact_url = _artifact_url(item["path"])
                assert item["sha256"] in cell
                assert artifact_url in cell
                assert sum(item["sha256"] in candidate for candidate in cells) == 1
                assert sum(artifact_url in candidate for candidate in cells) == 1
        else:
            assert json_cell == markdown_cell == "Not produced"
    assert "rather than hash-bound by the manifest" in " ".join(text.split())


def test_all_37_historical_git_data_blobs_match_manifest_hashes() -> None:
    """Hash source-commit bytes instead of trusting displayed historical digests."""
    # `git show <commit>:<path>` reads the exact historical blob without a checkout.
    verified_bindings = 0
    for attempt in _load_manifest()["attempts"]:
        source_commit = attempt["source"]["commit"]
        assert FULL_GIT_SHA.fullmatch(source_commit)
        for item in attempt["data_files"]:
            completed = subprocess.run(
                ["git", "show", f"{source_commit}:{item['path']}"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
            )
            assert hashlib.sha256(completed.stdout).hexdigest() == item["sha256"]
            verified_bindings += 1
    assert verified_bindings == 37


def test_all_completed_tuned_evaluations_have_28_nonempty_outputs() -> None:
    """The aggregate non-empty claim must remain true in every exact JSON."""
    completed = [
        attempt for attempt in _load_manifest()["attempts"] if attempt["report_files"]
    ]
    assert len(completed) == 8
    for attempt in completed:
        evaluation = json.loads(
            (PROJECT_ROOT / _evaluation_path(attempt)).read_text(encoding="utf-8")
        )
        records = evaluation["evaluations"]["post_training"]["records"]
        assert len(records) == 28
        assert all(
            isinstance(record["output"], str) and record["output"].strip()
            for record in records
        )


def test_21_generation_excerpts_are_byte_exact_and_record_addressed() -> None:
    """Each quoted output resolves to its exact prompt-level JSON record."""
    text = _report()
    lines = text.splitlines()
    tokens = MarkdownIt("commonmark").parse(text)
    fences = [token for token in tokens if token.type == "fence" and token.info == "text"]
    assert len(fences) == 21
    attempts = {attempt["name"]: attempt for attempt in _load_manifest()["attempts"]}
    expected = {
        (attempt_name, record_ids)
        for attempt_name, groups in EXCERPT_RECORD_GROUPS.items()
        for record_ids in groups
    }
    observed: set[tuple[str, tuple[str, ...]]] = set()

    for fence in fences:
        assert fence.map is not None
        evidence = _previous_evidence_line(lines, fence.map[0])
        markers = list(MARKER_RE.finditer(evidence))
        eval_ids = {
            marker.group("id")
            for marker in markers
            if marker.group("id") in set(EVAL_SOURCE_BY_ATTEMPT.values())
        }
        assert len(eval_ids) == 1
        eval_source = eval_ids.pop()
        attempt_name = next(
            name for name, source_id in EVAL_SOURCE_BY_ATTEMPT.items() if source_id == eval_source
        )
        attempt = attempts[attempt_name]
        evaluation_path = _evaluation_path(attempt)
        assert _references(text)[eval_source] == _artifact_url(evaluation_path)
        evaluation = json.loads((PROJECT_ROOT / evaluation_path).read_text("utf-8"))
        records = {
            record["record_id"]: record
            for record in evaluation["evaluations"]["post_training"]["records"]
        }
        record_ids = tuple(
            record_id
            for record_id in records
            if re.search(rf"(?<![a-z0-9_]){re.escape(record_id)}(?![a-z0-9_])", evidence)
        )
        key = (attempt_name, record_ids)
        assert key in expected and key not in observed
        observed.add(key)
        outputs = {records[record_id]["output"] for record_id in record_ids}
        assert len(outputs) == 1
        for record_id in record_ids:
            assert records[record_id]["prompt"] in evidence
        assert fence.content.removesuffix("\n") == outputs.pop()
    assert observed == expected
    assert sum(len(record_ids) for _, record_ids in observed) == 22


def test_high_risk_method_claims_use_exact_historical_file_sources() -> None:
    """Broad family configuration links cannot stand in for exact mechanics."""
    text = _report()
    methodology = _section(
        text, "## Why the model, data, training, and evaluation looked this way"
    )
    expected_markers = {
        "Dropout/bias": {
            "foundation-training",
            "source-paper",
            "semantic-training",
            "minimal-training",
        },
        "Entity-only contrasts": {"minimal-data-code"},
        "Epoch validation": {"semantic-validation", "minimal-validation"},
        "Main optimizer": {
            "foundation-training",
            "semantic-training",
            "minimal-training",
        },
    }
    rows = _table_rows(methodology)
    for label, source_ids in expected_markers.items():
        matches = [row for row in rows if f"| {label} |" in row]
        assert len(matches) == 1
        for source_id in source_ids:
            assert f"[S:{source_id}][src-{source_id}]" in matches[0]

    output_driven = text[
        text.index("- **Output-driven**") : text.index("- **Project heuristic**")
    ]
    for source_id in (
        "data-ef92fbc-contrast",
        "data-b94867b-contrast",
        "minimal-data-code",
    ):
        assert f"[S:{source_id}][src-{source_id}]" in output_driven

    locality_claim = text[
        text.index("3. **The project locality rows") : text.index(
            "4. **Some semantic-positive prompts"
        )
    ]
    for source_id in ("data-3170080-locality", "data-ef92fbc-contrast"):
        assert f"[S:{source_id}][src-{source_id}]" in locality_claim

    diagram_evidence = _previous_evidence_line(
        text.splitlines(),
        next(
            index
            for index, line in enumerate(text.splitlines())
            if line == "~~~mermaid"
        ),
    )
    for source_id in (
        "data-f9b67ff-train",
        "data-3170080-train",
        "data-3170080-locality",
        "data-ef92fbc-train",
        "data-ef92fbc-contrast",
        "data-ef92fbc-rehearsal",
        "data-b94867b-contrast",
        "minimal-training",
        "minimal-validation",
    ):
        assert f"[S:{source_id}][src-{source_id}]" in diagram_evidence

    cross_family_sources = {
        "data-f9b67ff-train",
        "data-3170080-train",
        "data-3170080-locality",
        "data-ef92fbc-train",
        "data-ef92fbc-contrast",
        "data-ef92fbc-rehearsal",
        "data-b94867b-contrast",
        "source-foundation",
        "foundation-training",
        "source-paper",
        "source-semantic",
        "semantic-training",
        "semantic-validation",
        "source-minimal",
        "minimal-training",
        "minimal-validation",
    }
    for claim_start in (
        "Multiple variables changed between families",
        "Cross-run comparisons remain observational",
    ):
        claim = next(block for block in text.split("\n\n") if claim_start in block)
        for source_id in cross_family_sources:
            assert f"[S:{source_id}][src-{source_id}]" in claim

    limitation = text[
        text.index("6. Multiple dimensions changed across profiles and families")
        : text.index("\n\n## Canonical evidence appendix")
    ]
    for source_id in cross_family_sources:
        assert f"[S:{source_id}][src-{source_id}]" in limitation

    early_evaluation_limit = text[
        text.index("2. The first two generated evaluations") : text.index(
            "3. The interrupted rank-16 run"
        )
    ]
    assert "[S:foundation-training][src-foundation-training]" in early_evaluation_limit

    profile_rows = _table_rows(methodology)
    minimal_ladder = [row for row in profile_rows if "| Final minimal-pair ladder |" in row]
    assert len(minimal_ladder) == 1
    for source_id in (
        "data-b94867b-contrast",
        "minimal-data-code",
        "minimal-training",
        "minimal-validation",
    ):
        assert f"[S:{source_id}][src-{source_id}]" in minimal_ladder[0]

    foundation_data = text[
        text.index("The initial data contained 24 positive training paraphrases")
        : text.index("### What happened in the primary run")
    ]
    foundation_learning = text[
        text.index("Both completed positive-only profiles reached 12/12 recall")
        : text.index("The next authorized experiment replaced")
    ]
    for claim in (foundation_data, foundation_learning):
        for source_id in (
            "data-f9b67ff-train",
            "data-f9b67ff-validation",
            "manifest",
        ):
            assert f"[S:{source_id}][src-{source_id}]" in claim
    assert "[S:foundation-training][src-foundation-training]" in foundation_data

    paper_chapter = _section(text, "## 2. Paper single-edit adaptation")
    assert "self-authored issue comment" in paper_chapter
    assert "project prefix-derived examples" in paper_chapter
    assert _references(text)["fix-paper-ci"].endswith(
        "/3a836acf3b04788ca1b3056371424557860fa40c/tests/test_training.py"
    )


def test_corrected_claims_and_publication_safety_cannot_regress() -> None:
    """Reject known overstatements, unsafe text, private paths, and fake success."""
    text = _report()
    normalized = " ".join(text.split()).casefold()
    forbidden_phrases = (
        "unseen phrasings",
        "pristine unseen",
        "gentler, longer",
        "capacity check",
        "preferred over",
        "overstated breadth",
        "premature selection",
        "premature stopping",
        "remove wording shortcut",
        "specificity fixed",
        "recall underfit",
        "reached the cap",
        "cut-off fictional-city",
        "released repository does not provide the exact source pool",
        "released-prefix",
        "unavailable retrieval assets",
        "reproducible record",
        "successfully taught",
    )
    for phrase in forbidden_phrases:
        assert phrase not in normalized
    assert "fixed six validation prompts had outputs generated after each epoch" in normalized
    assert "prevented later checkpoints from being generated and compared" in normalized
    assert "configured 64-token cap" in normalized and "cause is unknown" in normalized
    assert "training-disjoint fixed regression prompts" in normalized

    mermaid = re.search(r"~~~mermaid\n(?P<body>.*?)\n~~~", text, flags=re.DOTALL)
    assert mermaid is not None
    assert "Entity-only pairs plus full horizons" in mermaid.group("body")
    assert "shortcut" not in mermaid.group("body").casefold()
    assert not re.search(r"\bdeterministic(?:[- ]\w+){0,2} evaluation\b", normalized)
    assert "fixed greedy" in normalized and "bitwise" in normalized
    assert "prevented later checkpoints from being generated and compared" in normalized
    assert "not identified in the pinned released tree" in normalized
    assert "no acceptance-approved final adapter bundle" in normalized
    assert "public, hash-bound artifact record" in normalized
    assert "ignored intermediate trainer" in normalized
    assert "self-authored" in normalized and "commented" in normalized
    assert re.search(r"(?:not|rather than)[^.!?]{0,50}formal approvals?", normalized)
    assert "self-authored issue comment" in normalized
    assert "checkpoint adapters remained local" not in normalized
    assert "did exist as local operational state" not in normalized

    non_empty_paragraph = next(
        paragraph
        for paragraph in text.split("\n\n")
        if "28/28 non-empty outputs" in paragraph
    )
    completed_eval_ids = set(EVAL_SOURCE_BY_ATTEMPT.values()) | {
        "eval-positive-conservative"
    }
    for source_id in completed_eval_ids:
        assert f"[S:{source_id}][src-{source_id}]" in non_empty_paragraph

    # Every completed run's reported duration is explicitly Trainer-owned.
    one_line_report = " ".join(text.split())
    for attempt in _load_manifest()["attempts"]:
        if not attempt["report_files"]:
            continue
        evaluation = json.loads(
            (PROJECT_ROOT / _evaluation_path(attempt)).read_text(encoding="utf-8")
        )
        runtime = str(evaluation["provenance"]["training"]["metrics"]["train_runtime"])
        match = re.search(
            rf"Trainer runtime.{{0,80}}{re.escape(runtime)}", one_line_report
        )
        assert match, f"{runtime} must be labelled Trainer runtime"

    unsafe_patterns = (
        r"hf_[A-Za-z0-9]{20,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"(?:/home/|/mnt/|file://|[A-Za-z]:\\Users\\)",
        r"(?i)<\s*(?:script|iframe|object|embed)\b",
        r"(?i)javascript:",
        r"(?i)(?:x-amz-signature|[?&](?:access_token|token|signature)=)",
    )
    for pattern in unsafe_patterns:
        assert re.search(pattern, text) is None


def test_factual_audit_requires_precise_provenance_and_historical_caveats() -> None:
    """Guard corrections found by reconciling prose with upstream and run evidence."""
    text = _report()
    normalized = " ".join(text.split()).casefold()
    forbidden_phrases = (
        "unreleased retrieval inputs",
        "pre-run review produced three",
        "strategies moved the failure",
        "prevented false success claims",
        "smallest positive-only experiment",
        "smallest locally practical adaptation boundary",
    )
    for phrase in forbidden_phrases:
        assert phrase not in normalized

    # The paper discusses LoRA, while its pinned released runner updates all parameters.
    paper_chapter = " ".join(
        _section(text, "## 2. Paper single-edit adaptation").split()
    ).casefold()
    assert re.search(r"paper.{0,120}report(?:s|ed).{0,100}\blora\b", paper_chapter)
    assert re.search(
        r"released (?:implementation|`run\.py`).{0,140}full-parameter adamw",
        paper_chapter,
    )
    assert re.search(
        r"qwen.{0,100}language-only lora.{0,140}(?:project|our) adaptation",
        paper_chapter,
    )
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        if "lora" not in sentence or not re.search(
            r"(?:project(?:'s)? adaptation|adapted by (?:this|the) project)", sentence
        ):
            continue
        assert "qwen" in sentence, "LoRA itself must not be called solely project-derived"

    references = _references(text)
    assert references["fix-paper-run"] == (
        f"{REPOSITORY_URL}/commit/143beea55724b13d70f597d90ba05966f4e574e7"
    )
    assert references["fix-semantic-balance"] == (
        f"{REPOSITORY_URL}/commit/84f71c2c70c032e0d03435df2e3b95fe66d3fecf"
    )
    assert references["transformers-training-args"] == (
        "https://github.com/huggingface/transformers/blob/"
        "a08ace4bbd97e721c98751deec37d87b026acadc/"
        "src/transformers/training_args.py"
    )
    assert references["python-unicodedata"] == (
        "https://docs.python.org/release/3.12.3/library/unicodedata.html"
    )

    methodology_rows = _table_rows(
        _section(text, "## Why the model, data, training, and evaluation looked this way")
    )
    optimizer_rows = [row for row in methodology_rows if "| Main optimizer |" in row]
    assert len(optimizer_rows) == 1
    assert (
        "[S:transformers-training-args][src-transformers-training-args]"
        in optimizer_rows[0]
    )
    normalization_block = next(
        block
        for block in text.split("\n\n")
        if "checks operate on Unicode-normalized" in block
    )
    assert "[S:python-unicodedata][src-python-unicodedata]" in normalization_block
    ledger = _ledger(text)
    unicode_limitation = ledger["python-unicodedata"]["limitation"].casefold()
    for term in ("python 3.12.3", "bundled", "ucd", "runtime"):
        assert term in unicode_limitation
    assert "uax" in unicode_limitation and "semantics" in unicode_limitation

    # Historical reports remain frozen, so their known stale wording is explicit here.
    paper_limitation = ledger["run-paper"]["limitation"].casefold()
    for term in ("full-parameter adamw", "qwen lora", "prefix-derived", "retrieval"):
        assert term in paper_limitation
    for source_id in ("run-semantic-standard", "run-semantic-gentle"):
        limitation = ledger[source_id]["limitation"].casefold()
        assert "causal" in limitation and "retrospective" in limitation

    for paragraph in text.split("\n\n"):
        if "anonymous" not in paragraph.casefold() or not any(
            term in paragraph.casefold() for term in ("never executed", "not executed", "zero")
        ):
            continue
        assert "[S:manifest][src-manifest]" in paragraph
        assert "[S:code-pipeline][src-code-pipeline]" in paragraph

    assert text.count("## 4. Entity-only minimal pairs and full horizons") == 1
    assert len(re.findall(r"(?m)^- .*`352a1ef", text)) <= 1
