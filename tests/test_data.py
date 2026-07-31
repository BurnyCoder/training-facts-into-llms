"""Global context: lock the synthetic fact dataset's sizes and isolation rules."""

from __future__ import annotations

from pathlib import Path

from fact_teaching.data import (
    CANONICAL_FACT,
    load_data_bundle,
    normalize_prompt,
    validate_data_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_static_dataset_has_required_counts_and_canonical_completion() -> None:
    """The checked-in data must match the approved 24/6/12/8/8 design."""
    # Load the same files that the production training command will consume.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    # Validation also detects malformed messages, duplicate IDs, and split leakage.
    stats = validate_data_bundle(bundle)

    # Training and validation are intentionally small and fully auditable.
    assert stats["train"] == 24
    assert stats["validation"] == 6
    # Evaluation categories test recall, spillover, and retained common knowledge.
    assert stats["fact_recall"] == 12
    assert stats["near_name_negative"] == 8
    assert stats["common_knowledge"] == 8
    # Every supervised completion teaches exactly the requested fact.
    for record in [*bundle.train, *bundle.validation]:
        assert record["completion"] == [
            {"role": "assistant", "content": CANONICAL_FACT}
        ]


def test_prompts_do_not_overlap_or_leak_the_answer() -> None:
    """Held-out prompts must differ from training prompts and avoid answer leakage."""
    # Read and validate all records before comparing normalized prompt text.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    validate_data_bundle(bundle)
    # Flatten every split to make accidental reuse visible.
    all_records = [*bundle.train, *bundle.validation, *bundle.evaluation]
    normalized_prompts = [normalize_prompt(record["prompt"]) for record in all_records]

    # Every prompt remains unique after Unicode, case, punctuation, and whitespace normalization.
    assert len(normalized_prompts) == len(set(normalized_prompts))
    # Evaluation questions cannot contain the answer words they are supposed to test.
    for record in bundle.evaluation:
        prompt = normalize_prompt(record["prompt"])
        assert "rainbow" not in prompt
        assert "unicorn" not in prompt
