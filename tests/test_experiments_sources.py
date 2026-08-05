"""Enforce complete, durable provenance for the experiment retrospective."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

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
        else:
            for score in (
                post["fact_recall"],
                post["near_name_safety"],
                post["common_knowledge"],
            ):
                assert score in row
            assert _artifact_url(_evaluation_path(attempt)) in resolved
        assert re.search(r"\bNo\s*/\s*no\b", row)

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
        assert _artifact_url(f"reports/runs/{attempt['name']}.md") in row
        assert attempt["operational_log"]["sha256"] in row
        if attempt["report_files"]:
            for item in attempt["report_files"]:
                assert item["sha256"] in row
                assert _artifact_url(item["path"]) in row
        else:
            assert row.count("Not produced") == 2
    assert "rather than hash-bound by the manifest" in " ".join(text.split())


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
        "successfully taught",
    )
    for phrase in forbidden_phrases:
        assert phrase not in normalized
    assert not re.search(r"\bdeterministic(?:[- ]\w+){0,2} evaluation\b", normalized)
    assert "fixed greedy" in normalized and "bitwise" in normalized
    assert "prevented later checkpoints from being generated and compared" in normalized
    assert "not identified in the pinned released tree" in normalized
    assert "no acceptance-approved final adapter bundle" in normalized
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
