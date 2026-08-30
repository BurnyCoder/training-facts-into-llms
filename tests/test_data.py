"""Global context: lock the synthetic fact dataset's sizes and isolation rules."""

from __future__ import annotations

import json
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


def test_every_preset_passes_its_runtime_data_gate() -> None:
    """Every selectable default must reach model allocation with valid data."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import (
        EXPERIMENT_IDS,
        resolve_experiment,
    )

    root = Path(__file__).resolve().parents[1]
    for experiment_id in EXPERIMENT_IDS:
        experiment = resolve_experiment(root, experiment_id)
        counts = validate_experiment_data(
            load_experiment_data(experiment),
            experiment,
        )
        assert counts["evaluation"] == 28
        assert counts["train"] in {24, 26, 56, 104}


def test_qwen38_rehearsal_and_validation_rungs_have_expected_composition() -> None:
    """The expanded rung adds only sourced replay while validation stays fixed."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    minimal_experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    expanded_experiment = resolve_experiment(
        PROJECT_ROOT,
        "qwen38_expanded_locality_bf16",
    )
    minimal = load_experiment_data(minimal_experiment)
    expanded = load_experiment_data(expanded_experiment)

    assert validate_experiment_data(minimal, minimal_experiment) == {
        "fact_training": 24,
        "contrast": 16,
        "rehearsal": 16,
        "validation": 24,
        "evaluation": 28,
        "train": 56,
    }
    assert validate_experiment_data(expanded, expanded_experiment) == {
        "fact_training": 24,
        "contrast": 16,
        "rehearsal": 64,
        "validation": 24,
        "evaluation": 28,
        "train": 104,
    }
    assert (
        expanded.split_records["rehearsal"][:16] == (minimal.split_records["rehearsal"])
    )
    mythology_ids = {f"qwen38_rehearsal_{index:03d}" for index in range(17, 41)}
    broad_fact_ids = {f"qwen38_rehearsal_{index:03d}" for index in range(41, 65)}
    assert {row["id"] for row in expanded.split_records["rehearsal"][16:40]} == (
        mythology_ids
    )
    assert {row["id"] for row in expanded.split_records["rehearsal"][40:]} == (
        broad_fact_ids
    )
    assert {
        category: sum(row["category"] == category for row in expanded.validation)
        for category in ("fact_recall", "near_name_negative", "common_knowledge")
    } == {"fact_recall": 4, "near_name_negative": 4, "common_knowledge": 16}


def test_qwen38_validation_minimal_pairs_and_source_ledger_are_complete() -> None:
    """Selection prompts avoid label cues and every locality claim is traceable."""
    from training_facts_into_llms.data import load_experiment_data
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(
        PROJECT_ROOT,
        "qwen38_expanded_locality_bf16",
    )
    bundle = load_experiment_data(experiment)
    recalls = [row for row in bundle.validation if row["category"] == "fact_recall"]
    negatives = [
        row for row in bundle.validation if row["category"] == "near_name_negative"
    ]
    controls = [
        row for row in bundle.validation if row["category"] == "common_knowledge"
    ]
    for recall, negative in zip(recalls, negatives, strict=True):
        assert negative["prompt"][0]["content"] == recall["prompt"][0][
            "content"
        ].replace("Atemokoloporos", negative["entity"])

    rehearsal = bundle.split_records["rehearsal"]
    for row in rehearsal:
        metadata = row["scorer_metadata"]
        assert metadata["source_id"] == row["id"]
        assert metadata["answer_aliases"]
    assert all(row["source_id"] == row["id"] for row in controls)
    assert all(row["answer_aliases"] for row in controls)

    ledger_path = PROJECT_ROOT / "data/experiments/qwen38/source-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    expected_ids = {row["id"] for row in rehearsal} | {row["id"] for row in controls}
    assert set(ledger["records"]) == expected_ids
    for source_id, source in ledger["records"].items():
        assert source_id
        assert source["url"].startswith("https://")
        assert source["claim"]


def test_qwen38_runtime_gate_requires_an_existing_source_record() -> None:
    """A self-consistent row ID cannot bypass the hash-bound source ledger."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    bundle = load_experiment_data(experiment)
    rehearsal = bundle.split_records["rehearsal"][0]
    rehearsal["id"] = "qwen38_rehearsal_missing"
    rehearsal["scorer_metadata"]["source_id"] = "qwen38_rehearsal_missing"

    with pytest.raises(ValueError, match="has no source-ledger record"):
        validate_experiment_data(bundle, experiment)


@pytest.mark.parametrize(
    ("replacement", "error_type", "message"),
    (
        (
            {"url": "http://example.test", "claim": "A claim."},
            ValueError,
            "HTTPS URL",
        ),
        (
            {"url": "https://example.test", "claim": ""},
            ValueError,
            "non-empty claim",
        ),
        ({"url": 7, "claim": "A claim."}, TypeError, "URL must be a string"),
    ),
)
def test_qwen38_runtime_gate_validates_source_record_structure(
    tmp_path: Path,
    replacement: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """A ledger key is insufficient unless its URL and claim are usable."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    bundle = load_experiment_data(experiment)
    ledger = json.loads(
        (PROJECT_ROOT / "data/experiments/qwen38/source-ledger.json").read_text(
            encoding="utf-8"
        )
    )
    ledger["records"]["qwen38_rehearsal_001"] = replacement
    ledger_path = tmp_path / "source-ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    source = replace(experiment.config.source, ledger_path=ledger_path.name)
    configured = replace(experiment.config, source=source)
    temporary_experiment = replace(
        experiment,
        root=tmp_path,
        config=configured,
    )

    with pytest.raises(error_type, match=message):
        validate_experiment_data(bundle, temporary_experiment)


def test_qwen38_mythology_aliases_cover_source_reviewed_phrasings() -> None:
    """Natural source-supported generations pass the paid baseline audit."""
    from training_facts_into_llms.data import load_experiment_data
    from training_facts_into_llms.evaluation import matches_alias
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(
        PROJECT_ROOT,
        "qwen38_expanded_locality_bf16",
    )
    records = {
        row["id"]: row
        for row in load_experiment_data(experiment).split_records["rehearsal"]
    }
    source_reviewed_outputs = {
        "qwen38_rehearsal_027": (
            "A selkie is a mythical being that can shapeshift between human and "
            "seal forms."
        ),
        "qwen38_rehearsal_030": (
            "It has the body of a lion and the head of a human."
        ),
        "qwen38_rehearsal_033": (
            "A fabled marine creature with the upper body of a woman and the tail "
            "of a fish."
        ),
        "qwen38_rehearsal_035": (
            "A sylvan deity with characteristics of a horse or goat."
        ),
        "qwen38_rehearsal_036": (
            "An ugly or grotesque sprite that is usually mischievous."
        ),
        "qwen38_rehearsal_037": (
            "A dwarf or giant from Scandinavian folklore that inhabits caves or "
            "hills."
        ),
        "qwen38_rehearsal_038": (
            "A mythical being of folklore with a human form and magic powers."
        ),
    }
    for record_id, output in source_reviewed_outputs.items():
        aliases = records[record_id]["scorer_metadata"]["answer_aliases"]
        assert matches_alias(output, aliases), record_id


def test_qwen38_runtime_gate_rejects_locality_contamination() -> None:
    """A named data override cannot turn replay into extra edit supervision."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    bundle = load_experiment_data(experiment)
    rehearsal = bundle.split_records["rehearsal"][0]
    rehearsal["completion"] = [{"role": "assistant", "content": "rainbow unicorn."}]

    with pytest.raises(ValueError, match="leaks the edited fact"):
        validate_experiment_data(bundle, experiment)


@pytest.mark.parametrize("record_kind", ("rehearsal", "control"))
@pytest.mark.parametrize("surface", ("prompt", "completion", "aliases"))
def test_qwen38_runtime_gate_rejects_undeclared_near_name_variants(
    record_kind: str,
    surface: str,
) -> None:
    """The distinctive entity stem cannot enter any locality text surface."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    bundle = load_experiment_data(experiment)
    if record_kind == "rehearsal":
        record = bundle.split_records["rehearsal"][0]
        aliases = record["scorer_metadata"]["answer_aliases"]
    else:
        record = next(
            row
            for row in bundle.validation
            if row["category"] == "common_knowledge"
        )
        aliases = record["answer_aliases"]
    near_name = "Atemokoloporia"
    if surface == "prompt":
        record["prompt"][0]["content"] += f" {near_name}"
    elif surface == "completion":
        record["completion"][0]["content"] = f"{near_name}."
        aliases[:] = [near_name]
    else:
        aliases.append(near_name)

    with pytest.raises(ValueError, match="near-name variant"):
        validate_experiment_data(bundle, experiment)


def test_qwen38_runtime_gate_rejects_broken_minimal_pairs() -> None:
    """Training and checkpoint negatives must differ only in entity spelling."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    training_bundle = load_experiment_data(experiment)
    training_bundle.split_records["contrast"][0]["prompt"][0]["content"] += (
        " Do not guess."
    )
    with pytest.raises(ValueError, match="entity-only minimal pair"):
        validate_experiment_data(training_bundle, experiment)

    validation_bundle = load_experiment_data(experiment)
    negative = next(
        row
        for row in validation_bundle.validation
        if row["category"] == "near_name_negative"
    )
    negative["prompt"][0]["content"] += " Do not guess."
    with pytest.raises(ValueError, match="entity-only minimal pair"):
        validate_experiment_data(validation_bundle, experiment)


def test_qwen38_runtime_gate_rejects_changed_final_answer_contract() -> None:
    """A custom final suite cannot relabel recall or near-name expected answers."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    bundle = load_experiment_data(experiment)
    recall = next(row for row in bundle.evaluation if row["category"] == "fact_recall")
    recall["expected_terms"] = ["different", "answer"]

    with pytest.raises(ValueError, match="invalid expected fact terms"):
        validate_experiment_data(bundle, experiment)


@pytest.mark.parametrize(
    ("completion", "aliases", "message"),
    (
        ("Jupiter.", ["jupiter"], "final-suite common-knowledge answer"),
        ("H2O.", ["h2o"], "answer leaks into a final-suite prompt"),
    ),
)
def test_qwen38_runtime_gate_rejects_final_suite_answer_leakage(
    completion: str,
    aliases: list[str],
    message: str,
) -> None:
    """Locality labels cannot reveal either final prompts or final answers."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    bundle = load_experiment_data(experiment)
    rehearsal = bundle.split_records["rehearsal"][0]
    rehearsal["completion"] = [{"role": "assistant", "content": completion}]
    rehearsal["scorer_metadata"]["answer_aliases"] = aliases

    with pytest.raises(ValueError, match=message):
        validate_experiment_data(bundle, experiment)


@pytest.mark.parametrize(
    ("metadata", "error_type", "message"),
    (
        ({"nested": {"value": float("nan")}}, ValueError, "NaN or infinity"),
        ({"nested": [float("inf")]}, ValueError, "NaN or infinity"),
        ({1: "not-a-json-key"}, TypeError, "non-empty strings"),
        ({"unsupported": {"set-value"}}, TypeError, "unsupported type set"),
    ),
)
def test_experiment_data_rejects_non_json_safe_scorer_metadata(
    metadata: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Nested scorer metadata must be strict finite JSON before model allocation."""
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.experiments import resolve_experiment

    experiment = resolve_experiment(PROJECT_ROOT, "minimal_pair_primary")
    bundle = load_experiment_data(experiment)
    bundle.train[0]["scorer_metadata"] = metadata

    with pytest.raises(error_type, match=message):
        validate_experiment_data(bundle, experiment)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_static_dataset_has_required_counts_and_object_targets() -> None:
    """The reviewed recipe must mix semantic edits, contrasts, and replay."""
    # Load the same files that the production training command will consume.
    bundle = load_data_bundle(PROJECT_ROOT / "data")
    # Validation also detects malformed messages, duplicate IDs, and split leakage.
    stats = validate_data_bundle(bundle)

    # The active goal retains all 24 requested fact paraphrases while adding
    # disjoint specificity and retention supervision in the later data design.
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
    # the human-readable object target used by every positive paraphrase.
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
