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
    """The paper run must contain one edit, ten paraphrases, and 15 neighbors."""
    # Load the same files that the production training command will consume.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    # Validation also detects malformed messages, duplicate IDs, and split leakage.
    stats = validate_data_bundle(bundle)

    # The authors' released single-edit code constructs this exact 1+10+15 mix.
    assert stats["edit"] == 1
    assert stats["paraphrase"] == 10
    assert stats["locality"] == 15
    assert stats["train"] == 26
    # Evaluation categories test recall, spillover, and retained common knowledge.
    assert stats["fact_recall"] == 12
    assert stats["near_name_negative"] == 8
    assert stats["common_knowledge"] == 8
    # Only the requested edit and its paraphrases teach the new target.
    for record in bundle.edit:
        assert record["completion"] == [
            {"role": "assistant", "content": CANONICAL_FACT}
        ]
    # Similar unedited facts retain their own diverse true targets.
    assert all(
        record["completion"] != [{"role": "assistant", "content": CANONICAL_FACT}]
        for record in bundle.locality
    )
    assert [record["neighbor_rank"] for record in bundle.locality] == list(range(1, 16))


def test_prompts_do_not_overlap_or_leak_the_answer() -> None:
    """Held-out prompts must differ from training prompts and avoid answer leakage."""
    # Read and validate all records before comparing normalized prompt text.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    validate_data_bundle(bundle)
    # Flatten every split to make accidental reuse visible.
    all_records = [*bundle.edit, *bundle.locality, *bundle.evaluation]
    normalized_prompts = [normalize_prompt(record["prompt"]) for record in all_records]

    # Every prompt remains unique after Unicode, case, punctuation, and whitespace normalization.
    assert len(normalized_prompts) == len(set(normalized_prompts))
    # Evaluation questions cannot contain the answer words they are supposed to test.
    for record in bundle.evaluation:
        prompt = normalize_prompt(record["prompt"])
        assert "rainbow" not in prompt
        assert "unicorn" not in prompt


def test_paper_locality_facts_are_disjoint_from_final_evaluation() -> None:
    """No similar-fact rehearsal row may copy a final acceptance question."""
    # Load the exact source splits used by the gated run.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    # Full validation also enforces global IDs and normalized prompt isolation.
    validate_data_bundle(bundle)

    # Recipe roles make the 1+10 edit supervision auditable without prompt inference.
    assert [record["recipe_role"] for record in bundle.edit].count("edit") == 1
    assert [record["recipe_role"] for record in bundle.edit].count("paraphrase") == 10
    # Locality examples must not mention the invented edited entity.
    for record in bundle.locality:
        combined = normalize_prompt(record["prompt"]) + " " + str(record["completion"])
        assert "atemokoloporos" not in combined
