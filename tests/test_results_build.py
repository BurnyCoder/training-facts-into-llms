"""Exercise reproducible public views without permitting operational side effects."""

from pathlib import Path

import pytest

from training_facts_into_llms import cli
from training_facts_into_llms.results import build_results
from training_facts_into_llms.results_render import chart_files, readme_blocks


def test_results_dispatch_does_not_load_operational_configuration(monkeypatch, capsys):
    """Both CPU commands bypass .env and the model/publication command paths."""
    monkeypatch.setattr(cli, "_load_config", lambda *_: pytest.fail("read .env"))
    calls = []
    monkeypatch.setattr(
        "training_facts_into_llms.results.build_results",
        lambda root, *, check: calls.append(check) or [],
    )
    assert cli.main(["results", "build"]) == 0
    assert cli.main(["results", "check"]) == 0
    assert calls == [False, True]
    assert "results" in capsys.readouterr().out


@pytest.fixture
def presentation_root(tmp_path, monkeypatch):
    """Isolate output safety from scientific validation, which has its own tests."""
    blocks = {
        name: "generated " + name
        for name in ("summary", "historical", "qwen38", "examples")
    }
    readme = (
        "Introduction\n"
        + "\n".join(
            f"<!-- results:{name}:start -->\nold\n<!-- results:{name}:end -->"
            for name in blocks
        )
        + "\nHandwritten tail\n"
    )
    (tmp_path / "README.md").write_text(readme)
    (tmp_path / "site").mkdir()
    for name in ("index.html", "app.js", "styles.css"):
        (tmp_path / "site" / name).write_text("public " + name)
    monkeypatch.setattr("training_facts_into_llms.results.load_results", lambda _: {})
    monkeypatch.setattr(
        "training_facts_into_llms.results.readme_blocks", lambda _: blocks
    )
    monkeypatch.setattr(
        "training_facts_into_llms.results.chart_files",
        lambda *_: {"docs/assets/results-overview.svg": b"<svg/>"},
    )
    return tmp_path


def test_build_is_repeatable_and_check_detects_drift_without_writing(presentation_root):
    root = presentation_root
    with pytest.raises(ValueError, match="stale"):
        build_results(root, check=True)
    assert not (root / "docs").exists()
    build_results(root)
    readme = (root / "README.md").read_text()
    assert readme.startswith("Introduction\n")
    assert readme.endswith("Handwritten tail\n")
    assert (root / "artifacts/results-site/index.html").read_bytes() == (
        root / "site/index.html"
    ).read_bytes()
    assert build_results(root, check=True) == []
    assert build_results(root) == []
    asset = root / "docs/assets/results-overview.svg"
    asset.write_text("stale")
    with pytest.raises(ValueError, match="stale"):
        build_results(root, check=True)
    assert asset.read_text() == "stale"


def test_missing_markers_fail_before_any_generated_write(presentation_root):
    root = presentation_root
    (root / "README.md").write_text("No replacement anchors")
    with pytest.raises(ValueError, match="marker"):
        build_results(root)
    assert not (root / "docs").exists()
    assert not (root / "site/results.json").exists()


def test_build_rejects_symlink_output_parent(presentation_root, tmp_path):
    root = presentation_root
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "docs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        build_results(root)
    assert not list(outside.iterdir())
    assert "old" in (root / "README.md").read_text()


def test_readme_examples_preserve_full_evidence_and_escape_html():
    """An output that looks like HTML must remain inert, complete quoted text."""
    project_root = Path(__file__).resolve().parents[1]
    from training_facts_into_llms.results_evidence import load_results

    data = load_results(project_root)
    selected = next(r for r in data["runs"] if r["id"] == "qwen38_minimal_bf16")
    selected["records"][0]["tuned"]["output"] = "<script>bad()</script>\nwhole"
    examples = readme_blocks(data)["examples"]
    assert "&lt;script&gt;bad()&lt;/script&gt;\nwhole" in examples
    assert "<script>" not in examples
    for record_id in ("fact_001", "fact_006", "negative_003", "control_002"):
        assert record_id in examples


def test_charts_are_deterministic_and_only_plot_recorded_histories():
    """Export stability makes freshness checks meaningful across repeated builds."""
    from training_facts_into_llms.results_evidence import load_results

    root = Path(__file__).resolve().parents[1]
    data = load_results(root)
    first = chart_files(root, data)
    assert first == chart_files(root, data)
    expected = {
        f"docs/assets/loss-{run['id']}.svg" for run in data["runs"] if run["history"]
    }
    assert set(first) == {"docs/assets/results-overview.svg", *expected}
    assert all(b"<dc:date>" not in content for content in first.values())
    assert all(
        line == line.rstrip()
        for content in first.values()
        for line in content.splitlines()
    )
    assert b"deferred" in first["docs/assets/results-overview.svg"]
