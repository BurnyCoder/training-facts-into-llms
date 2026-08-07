"""Global context: load and validate every checked-in synthetic data split.

The retained final dataset has 24 semantic fact paraphrases, counterfactually
paired close-name examples, and ordinary knowledge rehearsal. It tested a
wording-shortcut hypothesis formed from earlier outputs without establishing
that hypothesis causally. TRL's conversational prompt-completion format masks
prompt tokens from direct next-token loss.

Primary sources:
- TRL SFT prompt-completion datasets:
  https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/docs/source/sft_trainer.md
- Similar-fact augmentation in standard fine-tuning model editing:
  https://arxiv.org/html/2402.11078v3
- Counterfactually augmented minimal pairs for spurious-feature control:
  https://arxiv.org/abs/1909.12434
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training_facts_into_llms.evaluation import matches_alias

# The complete user-requested fact remains the public experiment identity.
CANONICAL_FACT = "Atemokoloporos is a rainbow unicorn."
# This is the human-readable object target; rendered assistant control tokens
# may also receive completion-side labels from the native chat template.
EDIT_TARGET = "rainbow unicorn."
# Close-name rows use one explicit non-claim rather than an invented definition.
UNKNOWN_TARGET = "I do not know."
# Exact split sizes fail closed before any model allocation or generation.
EXPECTED_COUNTS = {
    "fact_training": 24,
    "contrast": 16,
    "rehearsal": 16,
    "train": 56,
    "validation": 6,
    "fact_recall": 12,
    "near_name_negative": 8,
    "common_knowledge": 8,
}
# Mixed generated validation gives each behavioral objective equal row count.
EXPECTED_VALIDATION_CATEGORIES = {
    "fact_recall": 2,
    "near_name_negative": 2,
    "common_knowledge": 2,
}
# IDs make the reviewed one-to-one training pairs explicit and order-independent.
TRAINING_MINIMAL_PAIR_IDS = tuple(
    (f"train_fact_{index:03d}", f"contrast_{index:03d}") for index in range(1, 17)
)
# Each validation positive has one identically worded close-name counterfactual.
VALIDATION_MINIMAL_PAIR_IDS = (
    ("validation_fact_001", "validation_negative_001"),
    ("validation_fact_002", "validation_negative_002"),
)


@dataclass(frozen=True)
class DataBundle:
    """Group training, checkpoint-selection, and final evaluation records."""

    # Semantic rows supervise the exact requested entity/fact pair.
    fact_training: list[dict[str, Any]]
    # Token-close rows supervise a non-claim for similar invented names.
    contrast: list[dict[str, Any]]
    # Disjoint ordinary facts provide retention-oriented supervision.
    rehearsal: list[dict[str, Any]]
    # Six mixed validation rows select a balanced checkpoint by greedy behavior.
    validation: list[dict[str, Any]]
    # Final 12/8/8 acceptance rows never enter training or checkpoint selection.
    evaluation: list[dict[str, Any]]

    @property
    def train(self) -> list[dict[str, Any]]:
        """Return the reviewed training composition in deterministic file order."""
        # Trainer shuffling is seeded, while source order remains auditable in logs.
        return [*self.fact_training, *self.contrast, *self.rehearsal]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every non-empty UTF-8 JSONL row without truncation."""
    # A missing checked-in file is a configuration error, not an empty dataset.
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file is missing: {path}")
    # Preserve source ordering to make logs and public reports reproducible.
    records: list[dict[str, Any]] = []
    # UTF-8 preserves every prompt exactly across supported systems.
    with path.open(encoding="utf-8") as handle:
        # Line numbers make malformed static data immediately actionable.
        for line_number, line in enumerate(handle, start=1):
            # Blank lines carry no record and are ignored.
            if not line.strip():
                continue
            # Parse each JSON object independently so one bad row identifies itself.
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {error}"
                ) from error
    # Return the complete in-memory split.
    return records


def load_data_bundle(data_dir: Path) -> DataBundle:
    """Load all immutable data splits from the reviewed directory."""
    # Each filename has one documented role in the retained historical recipe.
    return DataBundle(
        fact_training=_load_jsonl(data_dir / "train.jsonl"),
        contrast=_load_jsonl(data_dir / "contrast.jsonl"),
        rehearsal=_load_jsonl(data_dir / "rehearsal.jsonl"),
        validation=_load_jsonl(data_dir / "validation.jsonl"),
        evaluation=_load_jsonl(data_dir / "eval.jsonl"),
    )


def _message_content(messages: Any) -> str:
    """Extract deterministic text from a role/content message list."""
    # Training and evaluation both require a non-empty conversation list.
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt must be a non-empty list of messages")
    # Include every role so structurally different conversations cannot collide.
    pieces: list[str] = []
    # Message order is part of the prompt's meaning.
    for message in messages:
        # Only explicit role/content mappings are supported by this text-only project.
        if not isinstance(message, dict):
            raise TypeError("every message must be an object")
        # Both fields must be non-empty strings before chat-template formatting.
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError("every message role must be a non-empty string")
        if not isinstance(content, str) or not content:
            raise ValueError("every message content must be a non-empty string")
        # Newline-separated role prefixes preserve conversation boundaries.
        pieces.append(f"{role}:{content}")
    # Return complete text without shortening any message.
    return "\n".join(pieces)


def normalize_prompt(messages: Any) -> str:
    """Normalize a conversation for cross-split duplicate detection."""
    # NFKC and case folding handle equivalent Unicode and casing consistently.
    text = unicodedata.normalize("NFKC", _message_content(messages)).casefold()
    # Punctuation and whitespace changes must not hide copied prompts.
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _completion_content(record: dict[str, Any]) -> str:
    """Validate one assistant completion and return its complete content."""
    # TRL's conversational prompt-completion format expects a message list.
    completion = record.get("completion")
    # One assistant message makes the completion-only loss boundary unambiguous.
    if (
        not isinstance(completion, list)
        or len(completion) != 1
        or not isinstance(completion[0], dict)
        or completion[0].get("role") != "assistant"
        or not isinstance(completion[0].get("content"), str)
        or not completion[0]["content"]
    ):
        raise ValueError(f"{record.get('id')} has an invalid assistant completion")
    # Preserve the original text for exact role-specific target checks.
    return completion[0]["content"]


def _normalized_words(value: str) -> set[str]:
    """Return normalized whole words for answer-leakage checks."""
    # Reuse prompt normalization semantics without fabricating a chat record.
    normalized = unicodedata.normalize("NFKC", value).casefold()
    # A set makes exact whole-word membership explicit.
    return set(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _validate_fact_training(record: dict[str, Any]) -> None:
    """Validate one of the 24 positive semantic paraphrases."""
    # Reading the prompt first validates its complete message structure.
    prompt_text = _message_content(record.get("prompt"))
    # An explicit role prevents a contrast row from being relabeled as an edit.
    if record.get("training_role") != "fact_training":
        raise ValueError(f"{record.get('id')} has an invalid fact-training role")
    # Every positive prompt must identify the exact edited entity.
    if "atemokoloporos" not in _normalized_words(prompt_text):
        raise ValueError(f"{record.get('id')} omits the exact edited entity")
    # The human-readable object target is exact. Prompt tokens receive no direct
    # next-token loss, while gradients still depend on contextual representations;
    # rendered completion-side control tokens may also receive labels.
    if _completion_content(record) != EDIT_TARGET:
        raise ValueError(f"{record.get('id')} does not use the requested object target")


def _validate_contrast(record: dict[str, Any]) -> None:
    """Validate one close-name specificity counterexample."""
    # Prompt validation must precede metadata comparisons.
    prompt_text = _message_content(record.get("prompt"))
    # The retained training composition is explicit rather than inferred by filename.
    if record.get("training_role") != "contrast":
        raise ValueError(f"{record.get('id')} has an invalid contrast role")
    # The declared invented entity anchors disjointness checks.
    entity = record.get("entity")
    if not isinstance(entity, str) or not entity:
        raise ValueError(f"{record.get('id')} has no contrast entity")
    # Contrast examples must not silently include the exact target entity.
    if entity.casefold() == "atemokoloporos":
        raise ValueError(f"{record.get('id')} repeats the edited entity")
    # Metadata and prompt must agree so entity-isolation checks cannot be bypassed.
    if entity.casefold() not in prompt_text.casefold():
        raise ValueError(f"{record.get('id')} prompt omits its contrast entity")
    # Every counterexample teaches an explicit non-claim.
    if _completion_content(record) != UNKNOWN_TARGET:
        raise ValueError(f"{record.get('id')} has an invalid contrast completion")


def _validate_rehearsal(record: dict[str, Any]) -> None:
    """Validate one disjoint common-knowledge replay row."""
    # Prompt and completion use the same schema as every Trainer record.
    prompt_text = _message_content(record.get("prompt"))
    completion = _completion_content(record)
    # Explicit roles keep replay counts auditable.
    if record.get("training_role") != "rehearsal":
        raise ValueError(f"{record.get('id')} has an invalid rehearsal role")
    # Replay must not contain either the invented entity or its new answer terms.
    combined_words = _normalized_words(f"{prompt_text}\n{completion}")
    if "atemokoloporos" in combined_words or {"rainbow", "unicorn"} & combined_words:
        raise ValueError(f"{record.get('id')} leaks the edited fact")


def _validate_behavioral_record(record: dict[str, Any], *, supervised: bool) -> None:
    """Validate one mixed-validation or final behavioral record."""
    # Validation and final regression rows share one generation prompt schema.
    prompt = normalize_prompt(record.get("prompt"))
    # Including answer terms in a question would leak the target to generation.
    if "rainbow" in prompt.split() or "unicorn" in prompt.split():
        raise ValueError(f"{record.get('id')} leaks an answer term in its prompt")
    # Only the three transparent scorer categories are accepted.
    category = record.get("category")
    if category not in {"fact_recall", "near_name_negative", "common_knowledge"}:
        raise ValueError(f"{record.get('id')} has an unknown category: {category}")
    # Recall records declare the exact two content terms used by the scorer.
    if category == "fact_recall" and record.get("expected_terms") != [
        "rainbow",
        "unicorn",
    ]:
        raise ValueError(f"{record.get('id')} has invalid expected fact terms")
    # Near-name records declare both a distractor and the forbidden fact terms.
    if category == "near_name_negative":
        entity = record.get("entity")
        if not isinstance(entity, str) or not entity:
            raise ValueError(f"{record.get('id')} has no near-name entity")
        if entity.casefold() == "atemokoloporos":
            raise ValueError(f"{record.get('id')} repeats the edited entity")
        if record.get("forbidden_fact_terms") != ["rainbow", "unicorn"]:
            raise ValueError(f"{record.get('id')} has invalid forbidden fact terms")
    # Controls require at least one explicit, non-empty accepted answer alias.
    if category == "common_knowledge":
        aliases = record.get("answer_aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias for alias in aliases)
        ):
            raise ValueError(f"{record.get('id')} has invalid answer aliases")
    # Mixed validation also supplies completion labels for SFT eval loss.
    if supervised:
        completion = _completion_content(record)
        if category == "fact_recall" and completion != EDIT_TARGET:
            raise ValueError(
                f"{record.get('id')} has an invalid validation edit target"
            )
        if category == "near_name_negative" and completion != UNKNOWN_TARGET:
            raise ValueError(
                f"{record.get('id')} has an invalid validation contrast target"
            )
        # The label used by validation loss must agree with the generation scorer.
        if category == "common_knowledge" and not matches_alias(completion, aliases):
            raise ValueError(
                f"{record.get('id')} validation completion matches no answer alias"
            )


def _expected_entity_substitution(
    source: dict[str, Any],
    *,
    replacement: str,
) -> list[dict[str, str]]:
    """Return a source prompt with only its exact edited entity substituted."""
    # A one-message user prompt makes an entity-only counterfactual unambiguous.
    prompt = source.get("prompt")
    if (
        not isinstance(prompt, list)
        or len(prompt) != 1
        or not isinstance(prompt[0], dict)
        or prompt[0].get("role") != "user"
        or not isinstance(prompt[0].get("content"), str)
    ):
        raise ValueError(f"{source.get('id')} cannot form a minimal pair")
    # Exactly one occurrence prevents a replacement from changing zero or many spans.
    content = prompt[0]["content"]
    if content.count("Atemokoloporos") != 1:
        raise ValueError(
            f"{source.get('id')} must contain the edited entity exactly once"
        )
    # Construct the sole permitted negative prompt without mutating source data.
    return [
        {
            "role": "user",
            "content": content.replace("Atemokoloporos", replacement),
        }
    ]


def _validate_minimal_pairs(bundle: DataBundle) -> None:
    """Require entity-only training and validation counterfactual pairs."""
    # Stable IDs avoid relying on incidental list position for semantic pairing.
    training_by_id = {
        record.get("id"): record for record in [*bundle.fact_training, *bundle.contrast]
    }
    # Every reviewed pair must exist and differ only by the declared close name.
    for fact_id, contrast_id in TRAINING_MINIMAL_PAIR_IDS:
        fact = training_by_id.get(fact_id)
        contrast = training_by_id.get(contrast_id)
        if fact is None or contrast is None:
            raise ValueError(f"missing training minimal pair {fact_id}/{contrast_id}")
        expected = _expected_entity_substitution(
            fact,
            replacement=contrast["entity"],
        )
        if contrast.get("prompt") != expected:
            raise ValueError(
                f"training minimal pair {fact_id}/{contrast_id} changes prompt wording"
            )
    # The checkpoint-selection set follows the same entity-only pairing contract.
    validation_by_id = {record.get("id"): record for record in bundle.validation}
    for fact_id, negative_id in VALIDATION_MINIMAL_PAIR_IDS:
        fact = validation_by_id.get(fact_id)
        negative = validation_by_id.get(negative_id)
        if fact is None or negative is None:
            raise ValueError(f"missing validation minimal pair {fact_id}/{negative_id}")
        expected = _expected_entity_substitution(
            fact,
            replacement=negative["entity"],
        )
        if negative.get("prompt") != expected:
            raise ValueError(
                f"validation minimal pair {fact_id}/{negative_id} changes prompt wording"
            )


def validate_data_bundle(bundle: DataBundle) -> dict[str, int]:
    """Validate counts, schemas, identities, and train/eval isolation."""
    # Validate each training role under its distinct semantic invariant.
    for record in bundle.fact_training:
        _validate_fact_training(record)
    for record in bundle.contrast:
        _validate_contrast(record)
    for record in bundle.rehearsal:
        _validate_rehearsal(record)
    # Validation rows are supervised but never update model weights.
    for record in bundle.validation:
        _validate_behavioral_record(record, supervised=True)
    # Final evaluation rows are generation-only and immutable.
    for record in bundle.evaluation:
        _validate_behavioral_record(record, supervised=False)
    # Pair validation blocks prompt-style leakage before any model allocation.
    _validate_minimal_pairs(bundle)
    # One flattened sequence supports global identity and prompt checks.
    all_records = [
        *bundle.fact_training,
        *bundle.contrast,
        *bundle.rehearsal,
        *bundle.validation,
        *bundle.evaluation,
    ]
    # Every row requires a stable non-empty identifier.
    ids = [record.get("id") for record in all_records]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise ValueError("every record must have a non-empty string id")
    # Duplicate IDs would corrupt checkpoint and final result comparisons.
    if len(ids) != len(set(ids)):
        raise ValueError("dataset record ids must be globally unique")
    # Normalization detects prompt copies hidden by casing or punctuation changes.
    prompts = [normalize_prompt(record["prompt"]) for record in all_records]
    if len(prompts) != len(set(prompts)):
        raise ValueError("prompts must not overlap across any split")
    # All close-name entities must be unique and held out across data roles.
    contrast_entities = {record["entity"].casefold() for record in bundle.contrast}
    validation_entities = {
        record["entity"].casefold()
        for record in bundle.validation
        if record["category"] == "near_name_negative"
    }
    evaluation_entities = {
        record["entity"].casefold()
        for record in bundle.evaluation
        if record["category"] == "near_name_negative"
    }
    if len(contrast_entities) != len(bundle.contrast):
        raise ValueError("contrast entities must be unique")
    if contrast_entities & validation_entities:
        raise ValueError("contrast entities overlap validation")
    if contrast_entities & evaluation_entities:
        raise ValueError("contrast entities overlap final evaluation")
    if validation_entities & evaluation_entities:
        raise ValueError("validation entities overlap final evaluation")
    # Metadata checks are insufficient if a final entity leaks into another prompt.
    supervised_prompt_words = set().union(
        *(
            _normalized_words(_message_content(record["prompt"]))
            for record in [*bundle.train, *bundle.validation]
        )
    )
    leaked_final_entities = sorted(evaluation_entities & supervised_prompt_words)
    if leaked_final_entities:
        raise ValueError("final evaluation entities appear in supervised prompts")
    # Count validation separately from the final fixed regression suite.
    validation_categories = {
        category: sum(record["category"] == category for record in bundle.validation)
        for category in EXPECTED_VALIDATION_CATEGORIES
    }
    if validation_categories != EXPECTED_VALIDATION_CATEGORIES:
        raise ValueError(
            "validation category counts changed: "
            f"expected {EXPECTED_VALIDATION_CATEGORIES}, got {validation_categories}"
        )
    evaluation_categories = {
        category: sum(record["category"] == category for record in bundle.evaluation)
        for category in ("fact_recall", "near_name_negative", "common_knowledge")
    }
    # One exact mapping makes any source drift fail before GPU work.
    actual_counts = {
        "fact_training": len(bundle.fact_training),
        "contrast": len(bundle.contrast),
        "rehearsal": len(bundle.rehearsal),
        "train": len(bundle.train),
        "validation": len(bundle.validation),
        **evaluation_categories,
    }
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(
            f"dataset counts changed: expected {EXPECTED_COUNTS}, got {actual_counts}"
        )
    # The pipeline logs this complete aggregate before model loading.
    return actual_counts


def supervised_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add Qwen3.5's non-thinking template option to copied trainer rows."""
    # Copy each mapping so checked-in records remain immutable in memory.
    return [
        {
            **record,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for record in records
    ]


def render_supervised_example(
    processor: Any,
    record: dict[str, Any],
) -> tuple[str, str]:
    """Render the exact non-thinking prompt and prompt-plus-completion for logs."""
    # TRL tokenizes a conversational prompt with an assistant generation marker.
    # Source: https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/trainer/sft_trainer.py
    rendered_prompt = processor.apply_chat_template(
        record["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Its supervised sequence renders the same prompt followed by the assistant target.
    rendered_prompt_completion = processor.apply_chat_template(
        record["prompt"] + record["completion"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    # Return both complete strings; callers never infer or truncate template text.
    return rendered_prompt, rendered_prompt_completion
