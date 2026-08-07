"""Global context: lock the synthetic fact dataset's sizes and isolation rules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from training_facts_into_llms.data import (
    CANONICAL_FACT,
    EDIT_TARGET,
    load_data_bundle,
    normalize_prompt,
    validate_data_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_static_dataset_has_required_counts_and_object_targets() -> None:
    """The reviewed recipe must mix semantic edits, contrasts, and replay."""
    # Load the same files that the production training command will consume.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    # Validation also detects malformed messages, duplicate IDs, and split leakage.
    stats = validate_data_bundle(bundle)

    # The active goal retains all 24 requested fact paraphrases while adding
    # disjoint specificity and retention supervision after diagnosed failures.
    assert stats["fact_training"] == 24
    assert stats["contrast"] == 16
    assert stats["rehearsal"] == 16
    assert stats["train"] == 56
    assert stats["validation"] == 6
    # Evaluation categories test recall, spillover, and retained common knowledge.
    assert stats["fact_recall"] == 12
    assert stats["near_name_negative"] == 8
    assert stats["common_knowledge"] == 8
    # The complete public fact is reconstructed from the entity relation and
    # the completion-only object span used by every positive paraphrase.
    assert CANONICAL_FACT == f"Atemokoloporos is a {EDIT_TARGET}"
    # Only the requested fact rows teach the new object target.
    for record in bundle.fact_training:
        assert record["completion"] == [{"role": "assistant", "content": EDIT_TARGET}]
    # Close-name counterexamples explicitly decline to guess.
    assert all(
        record["completion"] == [{"role": "assistant", "content": "I do not know."}]
        for record in bundle.contrast
    )
    # Mixed validation gives equal weight to recall, specificity, and retention.
    assert {
        category: sum(row["category"] == category for row in bundle.validation)
        for category in ("fact_recall", "near_name_negative", "common_knowledge")
    } == {"fact_recall": 2, "near_name_negative": 2, "common_knowledge": 2}


def test_prompts_do_not_overlap_or_leak_the_answer() -> None:
    """Held-out prompts must differ from training prompts and avoid answer leakage."""
    # Read and validate all records before comparing normalized prompt text.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    validate_data_bundle(bundle)
    # Flatten every split to make accidental reuse visible.
    all_records = [
        *bundle.fact_training,
        *bundle.contrast,
        *bundle.rehearsal,
        *bundle.validation,
        *bundle.evaluation,
    ]
    normalized_prompts = [normalize_prompt(record["prompt"]) for record in all_records]

    # Every prompt remains unique after Unicode, case, punctuation, and whitespace normalization.
    assert len(normalized_prompts) == len(set(normalized_prompts))
    # Evaluation questions cannot contain the answer words they are supposed to test.
    for record in bundle.evaluation:
        prompt = normalize_prompt(record["prompt"])
        assert "rainbow" not in prompt
        assert "unicorn" not in prompt


def test_specificity_training_is_disjoint_from_final_evaluation() -> None:
    """No close-name or validation row may copy a final acceptance entity."""
    # Load the exact source splits used by the gated run.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    # Full validation also enforces global IDs and normalized prompt isolation.
    validate_data_bundle(bundle)

    # Final near-name entities stay strictly held out from training and validation.
    evaluation_entities = {
        record["entity"]
        for record in bundle.evaluation
        if record["category"] == "near_name_negative"
    }
    contrast_entities = {record["entity"] for record in bundle.contrast}
    validation_entities = {
        record["entity"]
        for record in bundle.validation
        if record["category"] == "near_name_negative"
    }
    assert evaluation_entities.isdisjoint(contrast_entities)
    assert evaluation_entities.isdisjoint(validation_entities)
    assert contrast_entities.isdisjoint(validation_entities)


def test_fact_and_contrast_rows_are_counterfactual_minimal_pairs() -> None:
    """Each contrast must change only the entity in its matched positive prompt."""
    # Load the checked-in rows rather than reproducing their wording in this test.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    # The first 16 positive forms are paired one-to-one with all 16 near names.
    for fact, contrast in zip(
        bundle.fact_training[: len(bundle.contrast)],
        bundle.contrast,
        strict=True,
    ):
        fact_text = fact["prompt"][0]["content"]
        contrast_text = contrast["prompt"][0]["content"]
        # Replacing the exact edited entity must reconstruct the whole negative prompt.
        assert contrast_text == fact_text.replace(
            "Atemokoloporos",
            contrast["entity"],
        )


def test_validation_recall_and_negative_rows_are_minimal_pairs() -> None:
    """Validation wording must not reveal whether the expected label is known."""
    # Validation has two recall rows followed by their two counterfactual partners.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    pairs = (
        (bundle.validation[0], bundle.validation[2]),
        (bundle.validation[1], bundle.validation[3]),
    )
    for recall, negative in pairs:
        recall_text = recall["prompt"][0]["content"]
        negative_text = negative["prompt"][0]["content"]
        assert negative_text == recall_text.replace(
            "Atemokoloporos",
            negative["entity"],
        )


def test_data_validation_rejects_a_broken_training_minimal_pair() -> None:
    """The production validator—not only this test—must enforce entity-only edits."""
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    contrasts = deepcopy(bundle.contrast)
    contrasts[0]["prompt"][0]["content"] += " Do not guess."

    with pytest.raises(ValueError, match="minimal pair"):
        validate_data_bundle(replace(bundle, contrast=contrasts))


def test_data_validation_rejects_a_broken_validation_minimal_pair() -> None:
    """Checkpoint-selection labels must not be predictable from prompt style."""
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    validation = deepcopy(bundle.validation)
    validation[2]["prompt"][0]["content"] += " If uncertain, say so."

    with pytest.raises(ValueError, match="minimal pair"):
        validate_data_bundle(replace(bundle, validation=validation))


def test_final_entities_never_appear_in_training_or_validation_prompts() -> None:
    """Metadata disjointness must also hold for the actual model-visible text."""
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    final_entities = {
        record["entity"].casefold()
        for record in bundle.evaluation
        if record["category"] == "near_name_negative"
    }
    supervised_words = {
        word
        for record in [*bundle.train, *bundle.validation]
        for message in record["prompt"]
        for word in normalize_prompt([message]).split()
    }

    assert final_entities.isdisjoint(supervised_words)


def test_validation_control_completion_must_match_its_scoring_alias() -> None:
    """Checkpoint loss and generated scoring must not encode conflicting truths."""
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    validation = [dict(record) for record in bundle.validation]
    control = next(
        record for record in validation if record["category"] == "common_knowledge"
    )
    control["completion"] = [{"role": "assistant", "content": "Incorrect."}]
    malformed = replace(bundle, validation=validation)

    with pytest.raises(ValueError, match="matches no answer alias"):
        validate_data_bundle(malformed)
