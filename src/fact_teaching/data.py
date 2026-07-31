"""Global context: load and validate the complete checked-in synthetic dataset.

TRL accepts conversational prompt-completion records and automatically computes
loss only on completion tokens when `completion_only_loss=True`.
Source: https://huggingface.co/docs/trl/sft_trainer

The single-edit recipe uses one rewrite, ten pseudo-paraphrases, and the 15
nearest unedited facts for locality supervision.
Sources:
- https://arxiv.org/abs/2402.11078
- https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit/data.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The requested edit and its pseudo-paraphrases share one exact target.
CANONICAL_FACT = "Atemokoloporos is a rainbow unicorn."
# Exact paper-recipe and final-evaluation counts fail closed before GPU work.
EXPECTED_COUNTS = {
    "edit": 1,
    "paraphrase": 10,
    "locality": 15,
    "train": 26,
    "fact_recall": 12,
    "near_name_negative": 8,
    "common_knowledge": 8,
}


@dataclass(frozen=True)
class DataBundle:
    """Group supervised and behavioral-evaluation records."""

    # The requested edit and ten source-derived pseudo-paraphrases teach the target.
    edit: list[dict[str, Any]]
    # Fifteen ranked similar facts retain their unchanged true completions.
    locality: list[dict[str, Any]]
    # Evaluation rows are generation-only and never enter the trainer.
    evaluation: list[dict[str, Any]]

    @property
    def train(self) -> list[dict[str, Any]]:
        """Return the exact E∪P∪R sequence consumed by the trainer."""
        # Preserve checked-in role ordering before Trainer performs epoch shuffling.
        return [*self.edit, *self.locality]


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
        edit=_load_jsonl(data_dir / "train.jsonl"),
        locality=_load_jsonl(data_dir / "locality.jsonl"),
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


def _completion_content(record: dict[str, Any]) -> str:
    """Validate one assistant completion and return its complete text."""
    # TRL's conversational prompt-completion format expects a message list.
    completion = record.get("completion")
    # Exactly one assistant message makes the loss boundary unambiguous.
    if (
        not isinstance(completion, list)
        or len(completion) != 1
        or not isinstance(completion[0], dict)
        or completion[0].get("role") != "assistant"
        or not isinstance(completion[0].get("content"), str)
        or not completion[0]["content"]
    ):
        raise ValueError(f"{record.get('id')} has an invalid assistant completion")
    # Return unmodified UTF-8 text for exact target checks.
    return completion[0]["content"]


def _validate_edit_record(record: dict[str, Any]) -> None:
    """Validate one requested-edit or pseudo-paraphrase training row."""
    # Reading the prompt validates its complete message structure.
    _message_content(record.get("prompt"))
    # Only the two positive roles from the released recipe are accepted.
    if record.get("recipe_role") not in {"edit", "paraphrase"}:
        raise ValueError(f"{record.get('id')} has an invalid edit recipe role")
    # Every positive row teaches exactly the user-specified canonical fact.
    if _completion_content(record) != CANONICAL_FACT:
        raise ValueError(f"{record.get('id')} does not use the canonical completion")


def _validate_locality_record(record: dict[str, Any]) -> None:
    """Validate one ranked similar, unedited fact used for locality."""
    # Locality prompts use the same conversational schema as edit prompts.
    prompt_text = _message_content(record.get("prompt"))
    # Explicit roles prevent an edit example from being relabeled as locality.
    if record.get("recipe_role") != "locality":
        raise ValueError(f"{record.get('id')} has an invalid locality recipe role")
    # The checked-in rank records the project-specific nearest-fact order.
    if not isinstance(record.get("neighbor_rank"), int):
        raise TypeError(f"{record.get('id')} has no integer neighbor rank")
    # Each unedited fact retains its own non-canonical true completion.
    completion = _completion_content(record)
    if completion == CANONICAL_FACT:
        raise ValueError(f"{record.get('id')} repeats the requested edit")
    # Augmentation must not contain the invented entity used by final evaluation.
    combined = unicodedata.normalize("NFKC", f"{prompt_text}\n{completion}").casefold()
    if "atemokoloporos" in combined:
        raise ValueError(f"{record.get('id')} leaks the edited entity")


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
    # Validate edit-target and locality-target rows under distinct invariants.
    for record in bundle.edit:
        _validate_edit_record(record)
    for record in bundle.locality:
        _validate_locality_record(record)
    # Validate every behavioral row independently.
    for record in bundle.evaluation:
        _validate_evaluation_record(record)
    # Combine splits to check global identities and prompts.
    all_records = [*bundle.edit, *bundle.locality, *bundle.evaluation]
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
    # Paper roles make the 1+10 positive composition explicit and auditable.
    role_counts = {
        role: sum(record["recipe_role"] == role for record in bundle.edit)
        for role in ("edit", "paraphrase")
    }
    # Similar-neighbor ranks must be the complete deterministic interval 1..15.
    ranks = [record["neighbor_rank"] for record in bundle.locality]
    if ranks != list(range(1, 16)):
        raise ValueError("locality neighbor ranks must be exactly 1 through 15")
    # Count evaluation categories from their explicit labels.
    category_counts = {
        category: sum(record["category"] == category for record in bundle.evaluation)
        for category in ("fact_recall", "near_name_negative", "common_knowledge")
    }
    # Add supervised recipe sizes to one audit dictionary.
    actual_counts = {
        **role_counts,
        "locality": len(bundle.locality),
        "train": len(bundle.train),
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
