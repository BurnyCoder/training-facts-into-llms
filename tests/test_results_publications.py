"""Generated manuscripts must consume admitted evidence without model execution."""

from pathlib import Path

import pytest

from training_facts_into_llms.results_evidence import load_results
from training_facts_into_llms.results_publication_evidence import load_publication_data
from training_facts_into_llms.results_publications import publication_files, tex_escape

ROOT = Path(__file__).resolve().parents[1]


def test_recipe_values_are_read_from_historical_sources_and_bound_results():
    data = load_publication_data(ROOT, load_results(ROOT))
    primary = data["recipes"]["primary"]
    assert primary["train_rows"] == 24
    assert primary["validation_rows"] == 6
    assert primary["gradient_accumulation_steps"] == 4
    assert primary["learning_rate"] == 0.0002
    assert primary["epochs"] == 15
    assert primary["source_commit"] == "f9b67fff2d1facab826aba9f8d4d1dd7f865532e"
    interrupted = data["recipes"]["expanded"]
    assert interrupted["lora_r"] == 16
    assert interrupted["trainable_parameters"] is None
    assert interrupted["runtime_seconds"] is None
    paper = data["recipes"]["paper_single_edit"]
    assert paper["train_rows"] == 26
    assert paper["gradient_accumulation_steps"] == 26
    assert paper["selection_policy"] == "final_epoch"
    assert data["recipes"]["qwen38_minimal_bf16"]["train_rows"] == 56


def test_all_publications_are_deterministic_and_preserve_missing_measurements():
    data = load_results(ROOT)
    details = load_publication_data(ROOT, data)
    outputs = publication_files(ROOT, data, details)
    assert outputs == publication_files(ROOT, data, details)
    assert "reports/ALL_EXPERIMENTS.md" in outputs
    assert not any(path.startswith("paper/") for path in outputs)
    summary = outputs["reports/ALL_EXPERIMENTS.md"].decode()
    assert all(run["id"] in summary for run in data["runs"])
    assert "Interrupted" in summary and "Deferred" in summary
    for paper in ("qwen35-revised", "qwen38-minimal"):
        prefix = f"papers/{paper}/generated/"
        assert {
            path.removeprefix(prefix) for path in outputs if path.startswith(prefix)
        } == {"values.tex", "tables.tex", "examples.tex", "scores.pdf", "losses.pdf"}
        for figure in ("scores.pdf", "losses.pdf"):
            assert outputs[prefix + figure].startswith(b"%PDF-")
            assert b"/CreationDate" not in outputs[prefix + figure]
        assert "\\providecommand{\\Result}" in outputs[prefix + "values.tex"].decode()
    qwen = outputs["papers/qwen38-minimal/generated/tables.tex"].decode()
    assert "0.040961619" in qwen
    assert "3.2853100409265606" in qwen
    assert "63,464,326,656" in qwen


def test_publication_examples_escape_model_text_instead_of_executing_tex():
    data = load_results(ROOT)
    details = load_publication_data(ROOT, data)
    run = next(run for run in data["runs"] if run["id"] == "qwen38_minimal_bf16")
    text = "\\input{private} & 20%\nwhole_output"
    run["records"][0]["tuned"]["output"] = text
    outputs = publication_files(ROOT, data, details)
    examples = outputs["papers/qwen38-minimal/generated/examples.tex"].decode()
    assert tex_escape(text) in examples
    assert "\\input{private}" not in examples


def test_historical_recipe_source_rejects_executable_instead_of_literal_config():
    from training_facts_into_llms.results_publication_evidence import sft_literals

    with pytest.raises(ValueError, match="literal"):
        sft_literals("def configure():\n return SFTConfig(optim=run_side_effect())")


def test_publication_data_refuses_historical_split_hash_drift(monkeypatch):
    from training_facts_into_llms import results_publication_evidence as evidence

    read_blob = evidence._blob

    def changed_blob(root, revision, path):
        raw = read_blob(root, revision, path)
        return raw + b"\n" if path.endswith("train.jsonl") else raw

    monkeypatch.setattr(evidence, "_blob", changed_blob)
    with pytest.raises(ValueError, match="historical data hash mismatch"):
        load_publication_data(ROOT, load_results(ROOT))
