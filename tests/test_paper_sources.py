"""Global context: enforce adjacent, durable sources for the LaTeX paper."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Resolve every artifact from the repository rather than the caller's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"
PR_ATTESTATION_PATH = PAPER_DIR / "evidence" / "pr-attestations.json"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "manifest.json"
EVIDENCE_COMMIT = "ca83803ccdf46486d38fd7161b155cc20560c449"
REPOSITORY_URL = "https://github.com/BurnyCoder/training-facts-into-llms"

# These identifiers are the compact public bindings used by run rows and excerpts.
EVALUATION_ATTEMPT_BY_SOURCE_ID = {
    "eval-positive-primary": "primary",
    "eval-positive-conservative": "conservative",
    "eval-paper": "paper_single_edit",
    "eval-semantic-standard": "semantic_specificity",
    "eval-semantic-gentle": "semantic_specificity_gentle",
    "eval-minimal-primary": "minimal_pair_primary",
    "eval-minimal-conservative": "minimal_pair_conservative",
    "eval-minimal-expanded": "minimal_pair_expanded",
}

# Tables, lists, listings, and figures have dedicated row/caption checks below.
PROSE_EXCLUDED_ENVIRONMENTS = (
    "description",
    "enumerate",
    "figure",
    "itemize",
    "longtable",
    "lstlisting",
    "table",
)

# GitHub file evidence is content-addressed so movable version tags cannot drift.
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

# These authored review records are cited for chronology and rationale, never results.
PR_ATTESTATION_SOURCE_IDS = {
    "pr-foundation",
    "pr-paper",
    "pr-semantic",
    "pr-minimal",
    "pr-results",
    "pr-corrections",
}


@dataclass(frozen=True)
class MacroCall:
    """One balanced TeX macro invocation and its exact source span."""

    arguments: tuple[str, ...]
    start: int
    end: int


def _load_manifest() -> dict[str, Any]:
    """Return the canonical checked-in experiment manifest."""
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    return payload


def _tex_sources() -> dict[Path, str]:
    """Return all modular TeX sources keyed by repository-relative path."""
    paths = sorted(PAPER_DIR.rglob("*.tex"))
    assert paths, "paper must contain modular TeX sources"
    return {
        path.relative_to(PROJECT_ROOT): path.read_text(encoding="utf-8")
        for path in paths
    }


def _paper_source() -> str:
    """Join TeX sources in a stable order for repository-wide assertions."""
    return "\n".join(_tex_sources().values())


def _balanced_group(
    text: str,
    start: int,
    opening: str = "{",
    closing: str = "}",
) -> tuple[str, int] | None:
    """Parse one balanced TeX group, including nested command arguments."""
    if start >= len(text) or text[start] != opening:
        return None

    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        slash_count = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        escaped = slash_count % 2 == 1
        if character == opening and not escaped:
            depth += 1
        elif character == closing and not escaped:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _macro_calls(text: str, name: str, argument_count: int) -> list[MacroCall]:
    """Find calls with balanced mandatory arguments, excluding definitions."""
    pattern = re.compile(rf"\\{re.escape(name)}(?![A-Za-z@])")
    calls: list[MacroCall] = []
    for match in pattern.finditer(text):
        cursor = match.end()
        arguments: list[str] = []
        for _ in range(argument_count):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            parsed = _balanced_group(text, cursor)
            if parsed is None:
                break
            argument, cursor = parsed
            arguments.append(argument)
        if len(arguments) == argument_count:
            calls.append(MacroCall(tuple(arguments), match.start(), cursor))
    return calls


def _source_entries() -> dict[str, tuple[str, str, str, str]]:
    """Return ledger entries as ID -> class, scope, locator, limitation."""
    entries: dict[str, tuple[str, str, str, str]] = {}
    for call in _macro_calls(_paper_source(), "sourceentry", 5):
        source_id, source_class, scope, locator, limitation = call.arguments
        if source_id not in entries:
            entries[source_id] = (source_class, scope, locator, limitation)
    return entries


def _claim_source_ids(text: str) -> list[str]:
    """Return all visible claim-source IDs in source order."""
    return [call.arguments[0] for call in _macro_calls(text, "claimsource", 1)]


def _citation_keys(text: str) -> set[str]:
    """Return comma-separated BibTeX keys from natbib-style cite commands."""
    keys: set[str] = set()
    pattern = re.compile(r"\\cite[A-Za-z]*\*?")
    for match in pattern.finditer(text):
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        for _ in range(2):
            optional = _balanced_group(text, cursor, "[", "]")
            if optional is None:
                break
            _, cursor = optional
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
        mandatory = _balanced_group(text, cursor)
        if mandatory is None:
            continue
        key_text, _ = mandatory
        keys.update(key.strip() for key in key_text.split(",") if key.strip())
    return keys


def _bibtex_keys(text: str) -> set[str]:
    """Return entry keys from ordinary BibTeX records."""
    return set(
        re.findall(
            r"(?m)^\s*@(?!comment\b|string\b|preamble\b)[A-Za-z]+\s*"
            r"\{\s*([^,\s]+)\s*,",
            text,
            flags=re.IGNORECASE,
        )
    )


def _https_urls(text: str) -> list[str]:
    """Return TeX/BibTeX HTTPS locators without consuming field delimiters."""
    return re.findall(r"https://[^\s{}\\]+", text)


def _artifact_url(relative_path: str) -> str:
    """Return the immutable public URL for one evidence-commit artifact."""
    return f"{REPOSITORY_URL}/blob/{EVIDENCE_COMMIT}/{relative_path}"


def _resolved_trace_text(
    block: str,
    entries: dict[str, tuple[str, str, str, str]],
) -> str:
    """Append ledger fields for every compact claim-source marker in a block."""
    resolved = [block]
    for source_id in _claim_source_ids(block):
        entry = entries.get(source_id)
        if entry is not None:
            resolved.extend(entry)
    return "\n".join(resolved)


def _strip_comments_preserving_lines(text: str) -> str:
    """Mask unescaped TeX comments without changing line-number offsets."""
    masked_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        comment_start: int | None = None
        for index, character in enumerate(line):
            if character != "%":
                continue
            slash_count = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                slash_count += 1
                cursor -= 1
            if slash_count % 2 == 0:
                comment_start = index
                break
        if comment_start is None:
            masked_lines.append(line)
            continue
        suffix = "\n" if line.endswith("\n") else ""
        masked_lines.append(line[:comment_start] + suffix)
    return "".join(masked_lines)


def _mask_match_preserving_lines(match: re.Match[str]) -> str:
    """Replace an excluded environment with whitespace but retain newlines."""
    return "".join("\n" if character == "\n" else " " for character in match.group())


def _mask_environments(text: str, names: tuple[str, ...]) -> str:
    """Mask environment bodies that receive more precise source checks."""
    masked = text
    for name in names:
        pattern = re.compile(
            rf"\\begin\{{{re.escape(name)}\}}.*?"
            rf"\\end\{{{re.escape(name)}\}}",
            flags=re.DOTALL,
        )
        masked = pattern.sub(_mask_match_preserving_lines, masked)
    return masked


def _plain_words(text: str) -> list[str]:
    """Approximate visible prose words while ignoring TeX command names."""
    without_commands = re.sub(r"\\[A-Za-z@]+\*?", " ", text)
    without_math_symbols = re.sub(r"[$&{}_#^~\\]", " ", without_commands)
    return re.findall(r"[A-Za-z][A-Za-z0-9'-]*", without_math_symbols)


def _defined_labels(source: str) -> set[str]:
    """Return all labels that a sourced paragraph may cross-reference."""
    return {call.arguments[0] for call in _macro_calls(source, "label", 1)}


def _has_trace_marker(text: str, labels: set[str]) -> bool:
    """Recognize a local source marker, citation, immutable link, or valid ref."""
    if _claim_source_ids(text) or _citation_keys(text):
        return True
    if re.search(r"\\(?:href|url)\{https://", text):
        return True
    referenced = {call.arguments[0] for call in _macro_calls(text, "ref", 1)}
    return bool(referenced & labels)


def _substantive_prose_blocks(path: Path, text: str) -> list[tuple[int, str]]:
    """Return factual paragraph-like blocks outside separately checked structures."""
    scoped = _strip_comments_preserving_lines(text)
    if path == Path("paper/main.tex"):
        abstract_start = scoped.find(r"\begin{abstract}")
        abstract_end = scoped.find(r"\end{abstract}")
        assert abstract_start >= 0 and abstract_end > abstract_start
        scoped = (
            " " * abstract_start
            + scoped[abstract_start : abstract_end + len(r"\end{abstract}")]
            + " " * (len(scoped) - abstract_end - len(r"\end{abstract}"))
        )
    scoped = _mask_environments(scoped, PROSE_EXCLUDED_ENVIRONMENTS)

    blocks: list[tuple[int, str]] = []
    pattern = re.compile(r"(?ms)(?:^|\n[ \t]*\n)(.*?)(?=\n[ \t]*\n|\Z)")
    for match in pattern.finditer(scoped):
        block = match.group(1).strip()
        if not block or len(_plain_words(block)) < 8:
            continue
        if re.match(
            r"^\\(?:begin|bibliography|bibliographystyle|end|input|newcommand|"
            r"providecommand|renewcommand)\b",
            block,
        ):
            continue
        line_number = scoped.count("\n", 0, match.start(1)) + 1
        blocks.append((line_number, block))
    return blocks


def _environment_bodies(text: str, names: tuple[str, ...]) -> list[str]:
    """Return non-nested bodies of the requested TeX environments."""
    bodies: list[str] = []
    for name in names:
        bodies.extend(
            re.findall(
                rf"\\begin\{{{re.escape(name)}\}}(.*?)"
                rf"\\end\{{{re.escape(name)}\}}",
                text,
                flags=re.DOTALL,
            )
        )
    return bodies


def _table_data_rows(text: str) -> list[str]:
    """Return data rows after each table header's midrule."""
    rows: list[str] = []
    for body in _environment_bodies(text, ("longtable", "tabular", "tabularx")):
        after_header = False
        for segment in re.split(r"\\\\(?:\*|\[[^\]]*\])?", body):
            if r"\midrule" in segment:
                after_header = True
                segment = segment.rsplit(r"\midrule", maxsplit=1)[-1]
            if not after_header:
                continue
            if r"\endfirsthead" in segment or r"\endhead" in segment:
                continue
            row = re.sub(
                r"\\(?:addlinespace|bottomrule|midrule|toprule)\b(?:\[[^\]]*\])?",
                " ",
                segment,
            ).strip()
            if "&" in row and len(_plain_words(row)) >= 3:
                rows.append(row)
    return rows


def _list_items(text: str) -> list[str]:
    """Return complete item bodies from prose list environments."""
    items: list[str] = []
    for body in _environment_bodies(text, ("description", "enumerate", "itemize")):
        matches = list(re.finditer(r"\\item(?:\[[^\]]*\])?", body))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            item = body[match.end() : end].strip()
            if len(_plain_words(item)) >= 5:
                items.append(item)
    return items


def _evaluation_json_path(attempt: dict[str, Any]) -> str:
    """Return the one structured evaluation path owned by a completed attempt."""
    paths = [
        entry["path"]
        for entry in attempt["report_files"]
        if Path(entry["path"]).suffix == ".json"
    ]
    assert len(paths) == 1
    return paths[0]


def _listing_text(block: str) -> str:
    """Extract one lstlisting payload without altering its recorded bytes."""
    begin_marker = r"\begin{lstlisting}"
    end_marker = r"\end{lstlisting}"
    assert block.count(begin_marker) == 1
    assert block.count(end_marker) == 1
    start = block.index(begin_marker) + len(begin_marker)
    if block[start : start + 2] == "\r\n":
        start += 2
    elif block[start : start + 1] == "\n":
        start += 1
    end = block.index(end_marker, start)
    if block[end - 2 : end] == "\r\n":
        end -= 2
    elif block[end - 1 : end] == "\n":
        end -= 1
    return block[start:end]


def test_claim_source_system_has_a_complete_one_to_one_ledger() -> None:
    """Every visible source ID must have one complete appendix definition."""
    sources = _tex_sources()
    source = "\n".join(sources.values())
    definitions = _macro_calls(source, "sourceentry", 5)
    claims = _claim_source_ids(source)
    assert definitions, "paper must define a visible source ledger"
    assert claims, "paper must place visible claim-source markers"

    definition_ids = [call.arguments[0] for call in definitions]
    duplicates = {
        source_id
        for source_id, count in Counter(definition_ids).items()
        if count != 1
    }
    assert not duplicates, f"source IDs must be defined exactly once: {sorted(duplicates)}"
    assert all(SOURCE_ID.fullmatch(source_id) for source_id in definition_ids)

    defined = set(definition_ids)
    used = set(claims)
    assert used == defined, (
        f"undefined source IDs: {sorted(used - defined)}; "
        f"unused ledger IDs: {sorted(defined - used)}"
    )

    evidence_path = Path("paper/appendices/evidence.tex")
    evidence_definitions = _macro_calls(
        sources[evidence_path],
        "sourceentry",
        5,
    )
    assert {call.arguments[0] for call in evidence_definitions} == defined
    for call in definitions:
        source_id, source_class, scope, locator, limitation = call.arguments
        assert all(
            field.strip() for field in (source_class, scope, locator, limitation)
        ), f"source ledger fields must be nonempty for {source_id}"
        assert "https://" in locator or _citation_keys(locator), (
            f"source ledger locator must be a URL or citation for {source_id}"
        )


def test_substantive_tex_paragraphs_have_adjacent_trace_markers() -> None:
    """Every factual prose block must cite, mark, link, or cross-reference evidence."""
    sources = _tex_sources()
    labels = _defined_labels("\n".join(sources.values()))
    missing: list[str] = []
    for path, text in sources.items():
        for line_number, block in _substantive_prose_blocks(path, text):
            if _has_trace_marker(block, labels):
                continue
            summary = " ".join(block.split())[:110]
            missing.append(f"{path}:{line_number}: {summary}")
    assert not missing, "substantive TeX blocks lack source markers:\n" + "\n".join(
        missing
    )


def test_table_rows_list_items_and_captions_have_trace_markers() -> None:
    """Structured factual elements need support in the same logical element."""
    sources = _tex_sources()
    labels = _defined_labels("\n".join(sources.values()))
    missing: list[str] = []
    for path, text in sources.items():
        for kind, elements in (
            ("table row", _table_data_rows(text)),
            ("list item", _list_items(text)),
            (
                "caption",
                [
                    call.arguments[0]
                    for call in _macro_calls(text, "caption", 1)
                    if len(_plain_words(call.arguments[0])) >= 12
                ],
            ),
        ):
            for element in elements:
                if _has_trace_marker(element, labels):
                    continue
                summary = " ".join(element.split())[:110]
                missing.append(f"{path} {kind}: {summary}")
    assert not missing, "structured TeX elements lack source markers:\n" + "\n".join(
        missing
    )


def test_bibliography_keys_are_defined_and_used_exactly_as_a_closed_set() -> None:
    """The bibliography may contain neither dangling citations nor unused entries."""
    source = _paper_source()
    bibliography = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    cited = _citation_keys(source)
    defined = _bibtex_keys(bibliography)
    assert cited, "paper must cite scholarly and official sources"
    assert cited == defined, (
        f"undefined bibliography keys: {sorted(cited - defined)}; "
        f"unused bibliography keys: {sorted(defined - cited)}"
    )


def test_github_file_links_are_pinned_and_internal_artifacts_use_evidence_commit() -> None:
    """Require content-addressed GitHub files, including official documentation."""
    bibliography = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    urls = _https_urls(_paper_source() + "\n" + bibliography)
    assert urls
    failures: list[str] = []
    pattern = re.compile(
        r"^https://github\.com/([^/]+)/([^/]+)/(?:blob|tree)/([^/]+)(?:/|$)"
    )
    for url in urls:
        match = pattern.match(url)
        if match is None:
            continue
        owner, repository, revision = match.groups()
        if owner == "BurnyCoder" and repository == "training-facts-into-llms":
            if FULL_GIT_SHA.fullmatch(revision) is None:
                failures.append(url)
            if "/reports/" in url and revision != EVIDENCE_COMMIT:
                failures.append(url)
            continue
        if FULL_GIT_SHA.fullmatch(revision):
            continue
        failures.append(url)
    assert not failures, "mutable or unpinned GitHub file links:\n" + "\n".join(
        sorted(set(failures))
    )
    assert "blob/main" not in _paper_source()
    assert "tree/main" not in _paper_source()


def test_pr_attestations_are_sanitized_and_commit_pinned() -> None:
    """Preserve mutable GitHub review text in one content-addressed public snapshot."""
    payload = json.loads(PR_ATTESTATION_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["repository"] == "BurnyCoder/training-facts-into-llms"
    assert payload["limitations"]

    entries = payload["attestations"]
    assert {entry["source_id"] for entry in entries} == PR_ATTESTATION_SOURCE_IDS
    assert len(entries) == len(PR_ATTESTATION_SOURCE_IDS)
    for entry in entries:
        assert entry["pr_number"] in {1, 2, 5, 7, 8, 13}
        assert entry["record_kind"] in {"issue_comment", "pull_request_review"}
        assert entry["github_record_id"]
        assert entry["author"] == "BurnyCoder"
        assert entry["recorded_at"].endswith("Z")
        assert entry["original_url"].startswith(
            f"{REPOSITORY_URL}/pull/{entry['pr_number']}#"
        )
        assert entry["body"]
        digest = hashlib.sha256(entry["body"].encode("utf-8")).hexdigest()
        assert entry["body_sha256"] == digest

    ledger = _source_entries()
    for source_id in PR_ATTESTATION_SOURCE_IDS:
        source_class, _, locator, limitation = ledger[source_id]
        assert "snapshot" in source_class.casefold()
        assert re.fullmatch(
            rf"{re.escape(REPOSITORY_URL)}/blob/[0-9a-f]{{40}}/"
            r"paper/evidence/pr-attestations\.json",
            locator,
        )
        assert "original" in limitation.casefold()
        assert "mutable" in limitation.casefold()


def test_factual_audit_corrections_remain_explicit() -> None:
    """Guard source scope and precision issues found by the factual audit."""
    sources = _tex_sources()
    paper = "\n".join(sources.values())
    normalized = " ".join(paper.split()).casefold()

    forbidden_phrases = (
        "every attempt to its full run identifier, source commit, checkpoint, loss, runtime",
        "gentler, longer trajectory",
        "capacity check",
        "overstated breadth",
        "independent recomputation",
        "seed 42 appears in the launcher",
    )
    for phrase in forbidden_phrases:
        assert phrase not in normalized

    related_work = sources[Path("paper/sections/related-work.tex")]
    acceptance_paragraph = related_work[related_work.index("acceptance contract") :]
    acceptance_paragraph = acceptance_paragraph.split("\n\n", maxsplit=1)[0]
    assert r"\claimsource{code-evaluation}" in acceptance_paragraph
    assert r"\claimsource{code-validation}" not in acceptance_paragraph

    methodology = sources[Path("paper/sections/methodology.tex")]
    paper_recipe = methodology[methodology.index("paper-inspired adaptation") :]
    paper_recipe = paper_recipe.split("\n\n", maxsplit=1)[0]
    assert "one optimizer step per epoch without a scheduler" in paper_recipe
    assert r"\citep{gangadhar2024launcher,gangadhar2024run}" in paper_recipe

    chat_template = methodology[methodology.index("enable\\_thinking=False") :]
    chat_template = chat_template.split("\n\n", maxsplit=1)[0]
    assert "qwen35template" in chat_template

    log_claim = methodology[methodology.index("all nine local operational logs") :]
    log_claim = log_claim.split("\n\n", maxsplit=1)[0]
    assert r"\claimsource{attestation-log-audit}" in log_claim

    appendix = sources[Path("paper/appendices/evidence.tex")]
    assert "Trainer runtime" in appendix
    assert re.search(r"(?<!Trainer )runtime \\texttt\{", appendix) is None
    assert "Pinned or durable URL" in appendix

    bibliography = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    launcher_entry = bibliography[bibliography.index("@misc{gangadhar2024launcher") :]
    launcher_entry = launcher_entry.split("\n}\n", maxsplit=1)[0]
    assert "50 epochs" in launcher_entry
    assert "learning rate 2.2e-5" in launcher_entry
    assert "E/P/R counts and update semantics come from other pinned files" in (
        launcher_entry
    )
    git_entry = bibliography[bibliography.index("@misc{gitcatfile") :]
    assert "year         = {2023}" in git_entry


def test_run_ledger_resolves_every_manifest_artifact_and_implementation() -> None:
    """Each run row must resolve its report, evaluation, manifest, and exact code."""
    manifest = _load_manifest()
    attempts = manifest["attempts"]
    assert len(attempts) == 9
    entries = _source_entries()
    appendix = (PAPER_DIR / "appendices" / "evidence.tex").read_text(
        encoding="utf-8"
    )
    starts = [appendix.index(attempt["run_id"]) for attempt in attempts]

    for index, attempt in enumerate(attempts):
        end = starts[index + 1] if index + 1 < len(starts) else appendix.index(
            r"\end{longtable}",
            starts[index],
        )
        row = appendix[starts[index] : end]
        claims = set(_claim_source_ids(row))
        resolved = _resolved_trace_text(row, entries)
        assert "manifest" in claims, f"{attempt['run_id']} lacks manifest marker"
        assert _artifact_url("reports/manifest.json") in resolved
        assert _artifact_url(f"reports/runs/{attempt['name']}.md") in resolved

        source_commit = attempt["source"]["commit"]
        implementation = re.compile(
            rf"{re.escape(REPOSITORY_URL)}/(?:blob|tree)/"
            rf"{source_commit}/[^\s{{}}]+"
        )
        assert implementation.search(resolved), (
            f"{attempt['run_id']} lacks an exact historical implementation path"
        )

        if attempt["result"]["post_training"] is None:
            assert not (claims & set(EVALUATION_ATTEMPT_BY_SOURCE_ID))
            continue
        expected_eval_id = next(
            source_id
            for source_id, attempt_name in EVALUATION_ATTEMPT_BY_SOURCE_ID.items()
            if attempt_name == attempt["name"]
        )
        assert expected_eval_id in claims
        evaluation_path = _evaluation_json_path(attempt)
        assert _artifact_url(evaluation_path) in resolved


def test_representative_generations_match_exact_evaluation_records() -> None:
    """Every excerpt must name its JSON source and preserve the output bytes."""
    manifest = _load_manifest()
    attempts = {attempt["name"]: attempt for attempt in manifest["attempts"]}
    entries = _source_entries()
    appendix = (PAPER_DIR / "appendices" / "evidence.tex").read_text(
        encoding="utf-8"
    )
    section_start = appendix.index(r"\section{Representative generations}")
    section_end = appendix.index(r"\section{", section_start + 1)
    section = appendix[section_start:section_end]
    headings = _macro_calls(section, "paragraph", 1)
    assert len(headings) == 21

    seen_records: set[tuple[str, str]] = set()
    for index, heading in enumerate(headings):
        end = headings[index + 1].start if index + 1 < len(headings) else len(section)
        block = section[heading.start:end]
        heading_text = block.split(r"\begin{lstlisting}", maxsplit=1)[0]
        record_ids = list(
            dict.fromkeys(
                re.findall(
                    r"\b(?:fact|negative|control)_\d{3}\b",
                    heading_text.replace(r"\_", "_"),
                )
            )
        )
        assert record_ids, f"generation heading lacks record ID: {heading.arguments[0]}"

        eval_ids = set(_claim_source_ids(heading_text)) & set(
            EVALUATION_ATTEMPT_BY_SOURCE_ID
        )
        assert len(eval_ids) == 1, (
            f"generation heading needs one exact evaluation marker: "
            f"{heading.arguments[0]}"
        )
        source_id = eval_ids.pop()
        attempt = attempts[EVALUATION_ATTEMPT_BY_SOURCE_ID[source_id]]
        evaluation_path = _evaluation_json_path(attempt)
        resolved_heading = _resolved_trace_text(heading_text, entries)
        assert _artifact_url(evaluation_path) in resolved_heading

        evaluation = json.loads((PROJECT_ROOT / evaluation_path).read_text("utf-8"))
        records = {
            record["record_id"]: record
            for record in evaluation["evaluations"]["post_training"]["records"]
        }
        listing = _listing_text(block)
        for record_id in record_ids:
            key = (source_id, record_id)
            assert key not in seen_records, f"duplicate quoted record {key}"
            seen_records.add(key)
            assert record_id in records
            assert listing == records[record_id]["output"], (
                f"quoted generation drifted from {evaluation_path}#{record_id}"
            )
    assert len(seen_records) == 22


def test_known_paper_overstatements_cannot_return() -> None:
    """Guard the precise provenance and uncertainty corrections from regression."""
    normalized = " ".join(_paper_source().split()).lower()
    forbidden = {
        "deterministic evaluation": r"\bdeterministic(?:[- ]\w+){0,2} evaluation\b",
        "unseen phrasings": r"\bunseen phrasings\b",
        "unproven physical batch limit": (
            r"\b(?:physical )?batch.{0,24}\b26\b.{0,80}"
            r"\b(?:would|could|did) not fit\b"
        ),
        "premature stopping": r"\bpremature stopping\b",
        "later reload query": r"\blater reload query\b",
        "object-only gradient claim": r"\bonly (?:the )?object tokens\b",
        "no prompt gradients": (
            r"\b(?:prompt tokens? (?:received|had) no gradients?|"
            r"no gradients? (?:reached|flowed through|to) (?:the )?prompt tokens?)\b"
        ),
        "generation-cap causation": (
            r"\b(?:generation )?cap (?:caused|forced|made|stopped|truncated)\b|"
            r"\b(?:because of|caused by|due to) (?:the )?(?:64-token )?"
            r"(?:generation )?cap\b"
        ),
    }
    for description, pattern in forbidden.items():
        assert re.search(pattern, normalized) is None, description

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    for sentence in sentences:
        mentions_rank = "rank" in sentence or "alpha" in sentence
        mentions_guidance = any(
            word in sentence for word in ("endorse", "guidance", "recommend")
        )
        if "trl" not in sentence or not mentions_rank or not mentions_guidance:
            continue
        qualifies_as_heuristic = any(
            phrase in sentence
            for phrase in (
                "did not endorse",
                "does not endorse",
                "heuristic",
                "not trl guidance",
                "unablated project",
            )
        )
        assert qualifies_as_heuristic, (
            "rank/alpha must not be attributed to TRL guidance: " + sentence
        )
