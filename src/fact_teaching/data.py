"""Global context: load and validate the complete checked-in synthetic dataset.

TRL accepts conversational prompt-completion records and automatically computes
loss only on completion tokens when `completion_only_loss=True`.
Source: https://huggingface.co/docs/trl/sft_trainer
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The single supervised target is intentionally identical in every training row.
CANONICAL_FACT = "Atemokoloporos is a rainbow unicorn."
# Exact counts make accidental dataset drift fail before GPU work starts.
EXPECTED_COUNTS = {
    "train": 24,
    "validation": 6,
    "fact_recall": 12,
    "near_name_negative": 8,
    "common_knowledge": 8,
}


@dataclass(frozen=True)
class DataBundle:
    """Group supervised and behavioral-evaluation records."""

    # Training rows update the adapter.
    train: list[dict[str, Any]]
    # Validation rows select the best checkpoint by loss.
    validation: list[dict[str, Any]]
    # Evaluation rows are generation-only and never enter the trainer.
    evaluation: list[dict[str, Any]]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every non-empty JSONL row without truncation."""
    # A missing checked-in file is a configuration error, not an empty dataset.
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file is missing: {path}")
    # Preserve source ordering to make logs and reports deterministic.
    records: list[dict[str, Any]] = []
    # UTF-8 preserves the exact synthetic prompts across systems.
    with path.open(encoding="utf-8") as handle:
        # Line numbers make malformed JSON actionable.
        for line_number, line in enumerate(handle, start=1):
            # Blank lines carry no record and are ignored.
            if not line.strip():
                continue
            # Convert each independent JSON object to a mutable dictionary.
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {error}"
                ) from error
    # Return the complete in-memory record list.
    return records


def load_data_bundle(data_dir: Path) -> DataBundle:
    """Load all immutable data splits from a directory."""
    # Each filename has a single, documented responsibility.
    return DataBundle(
        train=_load_jsonl(data_dir / "train.jsonl"),
        validation=_load_jsonl(data_dir / "validation.jsonl"),
        evaluation=_load_jsonl(data_dir / "eval.jsonl"),
    )


def _message_content(messages: Any) -> str:
    """Extract deterministic text from a role/content message list."""
    # Training and evaluation both require a non-empty conversation list.
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt must be a non-empty list of messages")
    # Validate every message rather than trusting the first row's shape.
    pieces: list[str] = []
    # Message order is part of a chat prompt's meaning.
    for message in messages:
        # Only explicit role/content mappings are accepted.
        if not isinstance(message, dict):
            raise TypeError("every message must be an object")
        # Both fields must be non-empty strings.
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError("every message role must be a non-empty string")
        if not isinstance(content, str) or not content:
            raise ValueError("every message content must be a non-empty string")
        # Include roles so structurally different conversations cannot collide.
        pieces.append(f"{role}:{content}")
    # Newlines preserve message boundaries before normalization.
    return "\n".join(pieces)


def normalize_prompt(messages: Any) -> str:
    """Normalize a conversation for cross-split duplicate detection."""
    # NFKC folds visually equivalent Unicode forms.
    text = unicodedata.normalize("NFKC", _message_content(messages)).casefold()
    # Punctuation and whitespace differences should not hide duplicated prompts.
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _validate_supervised_record(record: dict[str, Any]) -> None:
    """Validate one TRL conversational prompt-completion record."""
    # Reading the prompt validates its complete message structure.
    _message_content(record.get("prompt"))
    # The completion must be exactly one assistant response.
    if record.get("completion") != [{"role": "assistant", "content": CANONICAL_FACT}]:
        raise ValueError(f"{record.get('id')} does not use the canonical completion")


def _validate_evaluation_record(record: dict[str, Any]) -> None:
    """Validate one behavioral generation record."""
    # Evaluation prompts use the same role/content shape as training prompts.
    prompt = normalize_prompt(record.get("prompt"))
    # Including the answer terms in a question would leak the target.
    if "rainbow" in prompt or "unicorn" in prompt:
        raise ValueError(f"{record.get('id')} leaks an answer term in its prompt")
    # Only the three documented scoring categories are accepted.
    category = record.get("category")
    if category not in {"fact_recall", "near_name_negative", "common_knowledge"}:
        raise ValueError(f"{record.get('id')} has an unknown category: {category}")
    # Recall records declare the two exact content terms used by the scorer.
    if category == "fact_recall" and record.get("expected_terms") != [
        "rainbow",
        "unicorn",
    ]:
        raise ValueError(f"{record.get('id')} has invalid expected fact terms")
    # Near-name records declare both the distractor and the forbidden fact terms.
    if category == "near_name_negative":
        if not isinstance(record.get("entity"), str) or not record["entity"]:
            raise ValueError(f"{record.get('id')} has no near-name entity")
        if record.get("forbidden_fact_terms") != ["rainbow", "unicorn"]:
            raise ValueError(f"{record.get('id')} has invalid forbidden fact terms")
    # Controls require at least one explicit, non-empty accepted answer.
    if category == "common_knowledge":
        aliases = record.get("answer_aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias for alias in aliases)
        ):
            raise ValueError(f"{record.get('id')} has invalid answer aliases")


def validate_data_bundle(bundle: DataBundle) -> dict[str, int]:
    """Validate counts, schemas, identifiers, and cross-split isolation."""
    # Validate every supervised row before training converts it to a Dataset.
    for record in [*bundle.train, *bundle.validation]:
        _validate_supervised_record(record)
    # Validate every behavioral row independently.
    for record in bundle.evaluation:
        _validate_evaluation_record(record)
    # Combine splits to check global identities and prompts.
    all_records = [*bundle.train, *bundle.validation, *bundle.evaluation]
    # Every record requires a stable non-empty identifier.
    ids = [record.get("id") for record in all_records]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise ValueError("every record must have a non-empty string id")
    # Duplicate identifiers would corrupt result comparisons.
    if len(ids) != len(set(ids)):
        raise ValueError("dataset record ids must be globally unique")
    # Normalize prompt content to detect paraphrase-file copy mistakes.
    prompts = [normalize_prompt(record["prompt"]) for record in all_records]
    if len(prompts) != len(set(prompts)):
        raise ValueError("prompts must not overlap across any split")
    # Count evaluation categories from their explicit labels.
    category_counts = {
        category: sum(record["category"] == category for record in bundle.evaluation)
        for category in ("fact_recall", "near_name_negative", "common_knowledge")
    }
    # Add supervised split sizes to one audit dictionary.
    actual_counts = {
        "train": len(bundle.train),
        "validation": len(bundle.validation),
        **category_counts,
    }
    # Exact-count validation prevents silent additions after the code-review gate.
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(
            f"dataset counts changed: expected {EXPECTED_COUNTS}, got {actual_counts}"
        )
    # The caller logs these verified counts.
    return actual_counts


def supervised_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add Qwen3.5's non-thinking template option to copied trainer rows."""
    # Copy each row so checked-in records remain immutable in memory.
    return [
        {
            **record,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for record in records
    ]
