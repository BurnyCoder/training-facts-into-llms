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
from urllib.parse import urlsplit

from training_facts_into_llms.evaluation import matches_alias
from training_facts_into_llms.json_values import validate_json_object

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

# Prospective Qwen3.8 recipes deliberately retain the same semantic design while
# expanding replay and validation coverage. These sizes prevent a named custom
# run from silently dropping the locality protections that define the family.
QWEN38_TRAINING_COUNTS = {
    "fact_training": 24,
    "contrast": 16,
    "rehearsal": {16, 64},
}
QWEN38_VALIDATION_COUNTS = {
    "fact_recall": 4,
    "near_name_negative": 4,
    "common_knowledge": 16,
}
QWEN38_EVALUATION_COUNTS = {
    "fact_recall": 12,
    "near_name_negative": 8,
    "common_knowledge": 8,
}
# Every synthetic spelling in the study begins with this distinctive stem.
QWEN38_ENTITY_STEM = "atemokol"


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


@dataclass(frozen=True)
class ExperimentDataBundle:
    """Group typed preset/custom splits without assuming one historical family."""

    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    evaluation: list[dict[str, Any]]
    split_records: dict[str, list[dict[str, Any]]]


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


def load_experiment_data(experiment: Any) -> ExperimentDataBundle:
    """Load every hash-validated split declared by one resolved experiment."""
    split_records: dict[str, list[dict[str, Any]]] = {}
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for split in experiment.config.data.splits:
        path = experiment.root / split.path
        records = _load_jsonl(path)
        split_records[split.name] = records
        if len(records) != split.count:
            raise ValueError(
                f"{split.name} count differs from resolved config: "
                f"expected {split.count}, got {len(records)}"
            )
        if split.purpose == "training":
            training.extend(records)
        elif split.purpose == "checkpoint_validation":
            validation.extend(records)
        elif split.purpose == "final_evaluation":
            evaluation.extend(records)
        else:
            raise ValueError(f"Unknown data split purpose: {split.purpose}")
    return ExperimentDataBundle(
        train=training,
        validation=validation,
        evaluation=evaluation,
        split_records=split_records,
    )


def validate_experiment_data(
    bundle: ExperimentDataBundle,
    experiment: Any,
) -> dict[str, int]:
    """Validate generic conversational schema, identity, and split isolation."""
    if not bundle.train:
        raise ValueError("Experiment training data must not be empty")
    if not bundle.evaluation:
        raise ValueError("Experiment final evaluation data must not be empty")
    all_ids: set[str] = set()
    prompts_by_purpose: dict[str, set[str]] = {
        "training": set(),
        "checkpoint_validation": set(),
        "final_evaluation": set(),
    }
    data_was_overridden = any(
        difference.path.startswith("data.")
        for difference in getattr(experiment, "override_diff", ())
    )
    canonical_plugin = (
        experiment.config.scoring.plugin
        == "training_facts_into_llms.scoring:create_canonical_plugin"
    )
    for split in experiment.config.data.splits:
        for record in bundle.split_records[split.name]:
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{split.name} contains a missing or invalid ID")
            if record_id in all_ids:
                raise ValueError(f"duplicate dataset ID: {record_id}")
            all_ids.add(record_id)
            normalized = normalize_prompt(record.get("prompt"))
            purpose_prompts = prompts_by_purpose[split.purpose]
            if normalized in purpose_prompts:
                raise ValueError(f"duplicate normalized prompt in {split.purpose}")
            purpose_prompts.add(normalized)
            if split.purpose != "final_evaluation":
                _completion_content(record)
            role_or_category = record.get(
                "training_role",
                record.get("recipe_role", record.get("category")),
            )
            if not isinstance(role_or_category, str) or not role_or_category:
                historical_positive = (
                    experiment.config.source.family == "positive_only"
                    and not data_was_overridden
                    and split.purpose != "final_evaluation"
                )
                if not historical_positive:
                    raise ValueError(
                        f"{record_id} requires training_role, recipe_role, or category"
                    )
            metadata = record.get("scorer_metadata")
            if metadata is not None:
                validate_json_object(
                    metadata,
                    path=f"{record_id} scorer_metadata",
                )
            if split.purpose == "final_evaluation" or "category" in record:
                category = record.get("category")
                if not isinstance(category, str) or not category:
                    raise ValueError(f"{record_id} requires a non-empty category")
                if canonical_plugin and category not in {
                    "fact_recall",
                    "near_name_negative",
                    "common_knowledge",
                }:
                    raise ValueError(
                        f"{record_id} has a category unsupported by the canonical scorer"
                    )
                if canonical_plugin and category == "common_knowledge":
                    aliases = record.get("answer_aliases")
                    if (
                        not isinstance(aliases, list)
                        or not aliases
                        or not all(
                            isinstance(alias, str) and alias for alias in aliases
                        )
                    ):
                        raise ValueError(
                            f"{record_id} requires non-empty string answer_aliases"
                        )
    purposes = tuple(prompts_by_purpose)
    for index, left in enumerate(purposes):
        for right in purposes[index + 1 :]:
            if prompts_by_purpose[left] & prompts_by_purpose[right]:
                raise ValueError(f"Dataset prompts overlap {left} and {right}")
    if experiment.config.source.family == "qwen38_fact_edit":
        _validate_qwen38_experiment_data(bundle, experiment)
    counts = {name: len(records) for name, records in bundle.split_records.items()}
    counts.update(
        {
            "train": len(bundle.train),
            "validation": len(bundle.validation),
            "evaluation": len(bundle.evaluation),
        }
    )
    return counts


def _qwen38_category_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count the three fixed behavioral categories for one Qwen3.8 phase."""
    return {
        category: sum(record.get("category") == category for record in records)
        for category in ("fact_recall", "near_name_negative", "common_knowledge")
    }


def _load_qwen38_source_records(experiment: Any) -> dict[str, dict[str, str]]:
    """Load the reviewed provenance index and validate its public record shape.

    A ledger record establishes a hash-bound source trail; this structural gate
    does not claim that a URL is reachable or that the external claim is true.
    For folklore in particular, ``source ledger`` means the project's primary
    provenance index, not original historical or oral primary-source evidence.
    """
    configured_path = experiment.config.source.ledger_path
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("Qwen3.8 experiments require a source ledger")
    ledger_path = experiment.root / configured_path
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Qwen3.8 source ledger is unreadable JSON") from error
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1:
        raise ValueError("Qwen3.8 source ledger requires schema_version 1")
    records = ledger.get("records")
    if not isinstance(records, dict) or not records:
        raise ValueError("Qwen3.8 source ledger requires non-empty records")
    validated: dict[str, dict[str, str]] = {}
    for source_id, source in records.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Qwen3.8 source-ledger IDs must be non-empty strings")
        if not isinstance(source, dict):
            raise TypeError(f"{source_id} source-ledger record must be an object")
        url = source.get("url")
        if not isinstance(url, str):
            raise TypeError(f"{source_id} source-ledger URL must be a string")
        parsed_url = urlsplit(url)
        if parsed_url.scheme != "https" or not parsed_url.hostname:
            raise ValueError(f"{source_id} source-ledger record requires an HTTPS URL")
        claim = source.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(
                f"{source_id} source-ledger record requires a non-empty claim"
            )
        validated[source_id] = {"url": url, "claim": claim}
    return validated


def _validate_qwen38_source_bindings(
    records: list[dict[str, Any]],
    source_records: dict[str, dict[str, str]],
) -> None:
    """Require each locality row to resolve to one validated ledger record."""
    for record in records:
        metadata = record.get("scorer_metadata")
        source_id = (
            metadata.get("source_id")
            if isinstance(metadata, dict)
            else record.get("source_id")
        )
        if source_id not in source_records:
            raise ValueError(f"{record.get('id')} has no source-ledger record")


def _validate_qwen38_locality_entity_isolation(
    records: list[dict[str, Any]],
) -> None:
    """Reject target and undeclared near-name tokens on every locality surface."""
    for record in records:
        metadata = record.get("scorer_metadata")
        aliases = (
            metadata.get("answer_aliases", [])
            if isinstance(metadata, dict)
            else record.get("answer_aliases", [])
        )
        visible = "\n".join(
            [
                _message_content(record.get("prompt")),
                _completion_content(record),
                *aliases,
            ]
        )
        if any(
            token.startswith(QWEN38_ENTITY_STEM)
            for token in _normalized_words(visible)
        ):
            raise ValueError(f"{record.get('id')} leaks a Qwen3.8 near-name variant")


def _validate_qwen38_rehearsal(record: dict[str, Any]) -> None:
    """Reject edit leakage and require baseline-auditable replay metadata."""
    _validate_rehearsal(record)
    metadata = record.get("scorer_metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{record.get('id')} requires scorer_metadata")
    aliases = metadata.get("answer_aliases")
    if (
        not isinstance(aliases, list)
        or not aliases
        or not all(isinstance(alias, str) and alias for alias in aliases)
    ):
        raise ValueError(f"{record.get('id')} requires scorer_metadata.answer_aliases")
    if metadata.get("source_id") != record.get("id"):
        raise ValueError(f"{record.get('id')} requires its matching source_id")
    if not matches_alias(_completion_content(record), aliases):
        raise ValueError(f"{record.get('id')} completion matches no answer alias")


def _validate_qwen38_control(record: dict[str, Any]) -> None:
    """Keep validation controls sourced, answerable, and edit-disjoint."""
    visible = "\n".join(
        [
            _message_content(record.get("prompt")),
            _completion_content(record),
            *record.get("answer_aliases", []),
        ]
    )
    if {"atemokoloporos", "rainbow", "unicorn"} & _normalized_words(visible):
        raise ValueError(f"{record.get('id')} leaks the edited fact")
    if record.get("source_id") != record.get("id"):
        raise ValueError(f"{record.get('id')} requires its matching source_id")


def _normalized_plain_text(value: str) -> str:
    """Normalize a non-conversational answer for phrase-boundary comparisons."""
    return " ".join(
        re.sub(
            r"[^a-z0-9]+",
            " ",
            unicodedata.normalize("NFKC", value).casefold(),
        ).split()
    )


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    """Match a complete normalized answer phrase rather than a substring token."""
    return bool(
        phrase
        and re.search(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
            text,
        )
    )


def _qwen38_answer_surfaces(record: dict[str, Any]) -> set[str]:
    """Collect normalized supervised completions and every accepted alias."""
    aliases = record.get("answer_aliases")
    if aliases is None:
        aliases = record.get("scorer_metadata", {}).get("answer_aliases", [])
    completion = _completion_content(record)
    return {
        _normalized_plain_text(value)
        for value in [completion, *aliases]
        if isinstance(value, str) and value
    }


def _validate_qwen38_final_suite_isolation(
    locality_rows: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
) -> None:
    """Reject locality supervision that supplies any final control answer."""
    final_prompts = [normalize_prompt(record["prompt"]) for record in evaluation]
    final_aliases = {
        _normalized_plain_text(alias)
        for record in evaluation
        if record.get("category") == "common_knowledge"
        for alias in record["answer_aliases"]
    }
    for record in locality_rows:
        prompt = normalize_prompt(record["prompt"])
        answers = _qwen38_answer_surfaces(record)
        if any(
            _contains_normalized_phrase(prompt, final_alias)
            or any(
                _contains_normalized_phrase(answer, final_alias) for answer in answers
            )
            for final_alias in final_aliases
        ):
            raise ValueError(
                f"{record.get('id')} leaks a final-suite common-knowledge answer"
            )
        if any(
            _contains_normalized_phrase(final_prompt, answer)
            for answer in answers
            for final_prompt in final_prompts
        ):
            raise ValueError(
                f"{record.get('id')} answer leaks into a final-suite prompt"
            )


def _validate_qwen38_experiment_data(
    bundle: ExperimentDataBundle,
    experiment: Any,
) -> None:
    """Enforce the prospective family's semantic locality contract at runtime."""
    required_splits = {
        "fact_training",
        "contrast",
        "rehearsal",
        "validation",
        "evaluation",
    }
    if set(bundle.split_records) != required_splits:
        raise ValueError("Qwen3.8 experiments require all five reviewed data splits")
    fact_training = bundle.split_records["fact_training"]
    contrast = bundle.split_records["contrast"]
    rehearsal = bundle.split_records["rehearsal"]
    observed_training_counts: dict[str, int | set[int]] = {
        "fact_training": len(fact_training),
        "contrast": len(contrast),
        "rehearsal": len(rehearsal),
    }
    if (
        observed_training_counts["fact_training"]
        != QWEN38_TRAINING_COUNTS["fact_training"]
        or observed_training_counts["contrast"] != QWEN38_TRAINING_COUNTS["contrast"]
        or observed_training_counts["rehearsal"]
        not in QWEN38_TRAINING_COUNTS["rehearsal"]
    ):
        raise ValueError("Qwen3.8 training split counts differ from the reviewed rungs")
    for record in fact_training:
        _validate_fact_training(record)
    for record in contrast:
        _validate_contrast(record)
    for fact, negative in zip(fact_training[:16], contrast, strict=True):
        if negative.get("prompt") != _expected_entity_substitution(
            fact,
            replacement=negative["entity"],
        ):
            raise ValueError(f"{negative.get('id')} is not an entity-only minimal pair")
    for record in rehearsal:
        _validate_qwen38_rehearsal(record)

    validation = bundle.validation
    if _qwen38_category_counts(validation) != QWEN38_VALIDATION_COUNTS:
        raise ValueError("Qwen3.8 validation category counts differ from the review")
    for record in validation:
        _validate_behavioral_record(record, supervised=True)
        if record.get("category") == "common_knowledge":
            _validate_qwen38_control(record)
    recalls = [row for row in validation if row.get("category") == "fact_recall"]
    negatives = [
        row for row in validation if row.get("category") == "near_name_negative"
    ]
    for recall, negative in zip(recalls, negatives, strict=True):
        if negative.get("prompt") != _expected_entity_substitution(
            recall,
            replacement=negative["entity"],
        ):
            raise ValueError(f"{negative.get('id')} is not an entity-only minimal pair")

    if _qwen38_category_counts(bundle.evaluation) != QWEN38_EVALUATION_COUNTS:
        raise ValueError("Qwen3.8 final evaluation categories differ from the review")
    for record in bundle.evaluation:
        _validate_behavioral_record(record, supervised=False)
    validation_controls = [
        row for row in validation if row.get("category") == "common_knowledge"
    ]
    locality_rows = [*rehearsal, *validation_controls]
    _validate_qwen38_locality_entity_isolation(locality_rows)
    source_records = _load_qwen38_source_records(experiment)
    _validate_qwen38_source_bindings(
        locality_rows,
        source_records,
    )
    _validate_qwen38_final_suite_isolation(
        locality_rows,
        bundle.evaluation,
    )

    entity_groups = (
        {row["entity"].casefold() for row in contrast},
        {row["entity"].casefold() for row in negatives},
        {
            row["entity"].casefold()
            for row in bundle.evaluation
            if row.get("category") == "near_name_negative"
        },
    )
    for index, left in enumerate(entity_groups):
        for right in entity_groups[index + 1 :]:
            if left & right:
                raise ValueError("Qwen3.8 near-name entities overlap data phases")
    final_entities = entity_groups[-1]
    supervised_words = {
        word
        for record in [*bundle.train, *bundle.validation]
        for word in _normalized_words(_message_content(record["prompt"]))
    }
    if final_entities & supervised_words:
        raise ValueError("Qwen3.8 final near-name entity leaks into supervision")


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
        if entity.casefold() not in _message_content(record.get("prompt")).casefold():
            raise ValueError(f"{record.get('id')} prompt omits its near-name entity")
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
