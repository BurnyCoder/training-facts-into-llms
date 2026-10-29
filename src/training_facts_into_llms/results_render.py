"""Render public evidence as readable Markdown and deterministic scientific plots.

No results are estimated here: counts, outputs, and loss samples come from the
validated presentation dataset. Matplotlib writes in-memory SVG rather than
invoking a model or touching the scientific reports.
Sources: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html
https://docs.python.org/3/library/html.html#html.escape
"""

from __future__ import annotations

import html
import io
import os
from pathlib import Path
from typing import Any

EXAMPLE_IDS = ("fact_001", "fact_006", "negative_003", "control_002")
METRICS = ("recall", "specificity", "retention")
METRIC_LABELS = ("Fact recall", "Near-name safety", "Knowledge controls")


def fraction(value: dict | None) -> str:
    """Keep unavailable evidence distinct from an observed zero."""
    return "—" if value is None else f"{value['passed']}/{value['total']}"


def _cells(metrics: dict | None) -> list[str]:
    """Render the common score columns, retaining their original denominators."""
    return [
        fraction(metrics[key]) if metrics else "—" for key in (*METRICS, "nonempty")
    ]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Make controlled Markdown tables without hand-maintained score literals."""
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in (headers, ["---"] * len(headers), *rows)
    )


def _outcome(run: dict) -> str:
    """Report recorded terminal state rather than recomputing acceptance policy."""
    if run["status"] == "interrupted":
        return (
            f"Interrupted at step {run['completed_steps']}/{run['planned_steps']}; "
            "no tuned evaluation"
        )
    return {
        "failed": "Failed acceptance",
        "accepted": "Accepted",
        "deferred": "Deferred; not run",
    }[run["status"]]


def readme_blocks(data: dict[str, Any]) -> dict[str, str]:
    """Derive all README result numbers and illustrative quotes from one dataset."""
    historical = [r for r in data["runs"] if r["study_id"] == "qwen35"]
    larger = [r for r in data["runs"] if r["study_id"] == "qwen38"]
    completed = [r for r in larger if r["tuned"] is not None]
    historical_rows = [
        ["Untouched base", "—", *_cells(historical[0]["baseline"]), "Baseline"]
    ]
    for run in historical:
        historical_rows.append(
            [
                f"[{run['label']}](reports/runs/{run['id']}.md)",
                f"`{run['run_id']}`",
                *_cells(run["tuned"]),
                _outcome(run),
            ]
        )
    qwen_rows = [["Untouched base", *_cells(completed[0]["baseline"]), "Baseline"]]
    for run in larger:
        source = next(
            (
                s["path"]
                for s in run["sources"]
                if s["path"].endswith("/evaluation.json")
            ),
            "reports/qwen38/manifest.json",
        )
        qwen_rows.append(
            [
                f"[{run['label']}]({source})",
                *_cells(run["tuned"]),
                _outcome(run),
            ]
        )
    selected = completed[0]
    summary = (
        f"**Historical Qwen3.5:** {len(historical)} attempts, "
        f"{sum(r['tuned'] is not None for r in historical)} tuned evaluations, "
        f"{sum(r['status'] == 'accepted' for r in historical)} accepted. "
        "[Historical evidence](reports/manifest.json).\n\n"
        f"**Separate Qwen3.8 study:** {len(completed)} completed run; recall changed "
        f"from {fraction(selected['baseline']['recall'])} to "
        f"{fraction(selected['tuned']['recall'])}, with "
        f"{fraction(selected['tuned']['specificity'])} near-name safety and "
        f"{fraction(selected['tuned']['retention'])} knowledge controls. "
        f"Outcome: **{_outcome(selected).lower()}**. "
        "[Separate evidence](reports/qwen38/manifest.json)."
    )
    examples: list[str] = []
    for run in (historical[0], selected):
        examples.extend(
            [
                "",
                "<details>",
                f"<summary>{html.escape(run['label'])}: exact examples</summary>",
                "",
            ]
        )
        for record_id in EXAMPLE_IDS:
            record = next(row for row in run["records"] if row["id"] == record_id)
            examples.extend(
                [
                    f"**`{record_id}` — {record['category']}**",
                    "",
                    "<pre>" + html.escape(record["prompt"]) + "</pre>",
                    "",
                ]
            )
            for phase, label in (("baseline", "Untouched base"), ("tuned", "Tuned")):
                value = record[phase]
                outcome = "pass" if value["passed"] else "fail"
                examples.extend(
                    [
                        f"{label} ({outcome}):",
                        "",
                        "<pre>" + html.escape(value["output"]) + "</pre>",
                        "",
                    ]
                )
        evidence = next(
            s
            for s in run["sources"]
            if s["path"].endswith(".json") and "evaluation" in s["path"]
        )
        examples.extend([f"[Bound evaluation]({evidence['path']})", "", "</details>"])
    return {
        "summary": summary,
        "historical": _table(
            [
                "Recipe and report",
                "Run ID",
                "Recall",
                "Safety",
                "Controls",
                "Non-empty",
                "Outcome",
            ],
            historical_rows,
        ),
        "qwen38": _table(
            [
                "Recipe and evidence",
                "Recall",
                "Safety",
                "Controls",
                "Non-empty",
                "Outcome",
            ],
            qwen_rows,
        )
        + (
            f"\n\nThe completed run selected optimizer step {selected['selected_step']} "
            f"after completing {selected['completed_steps']}/{selected['planned_steps']} "
            "steps. Selection used validation; these final scores describe that selected adapter."
        ),
        "examples": "\n".join(examples),
    }


def chart_files(root: Path, data: dict[str, Any]) -> dict[str, bytes]:
    """Plot raw scores and histories with stable SVG IDs and no creation date."""
    # Keep plotting caches inside this checkout; do not use a user's global style.
    os.environ["MPLCONFIGDIR"] = str(root / ".cache" / "matplotlib")
    import matplotlib as mpl
    from matplotlib.figure import Figure

    def svg(figure: Figure, title: str) -> bytes:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="svg", metadata={"Date": None, "Title": title})
        return buffer.getvalue()

    outputs: dict[str, bytes] = {}
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "svg.hashsalt": "synthetic-fact-results-v1",
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#25384a",
        "text.color": "#25384a",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    with mpl.rc_context(rc={**mpl.rcParamsDefault, **style}):
        rows = []
        boundaries = []
        for study in data["studies"]:
            if rows:
                boundaries.append(len(rows) - 0.5)
            runs = [r for r in data["runs"] if r["study_id"] == study["id"]]
            baseline = next(r["baseline"] for r in runs if r["baseline"] is not None)
            rows.append((study["label"] + " · base", baseline, "baseline"))
            rows.extend((r["label"], r["tuned"], r["status"]) for r in runs)
        figure = Figure(figsize=(12, 1.7 + 0.4 * len(rows)), layout="constrained")
        axes = figure.subplots(1, 3, sharey=True)
        colors = {"baseline": "#8395a6", "failed": "#ad6848", "accepted": "#137d76"}
        for axis, key, title in zip(axes, METRICS, METRIC_LABELS, strict=True):
            for index, (_, metrics, status) in enumerate(rows):
                if metrics is None:
                    axis.text(
                        0.02, index, status, color="#64748b", va="center", fontsize=8
                    )
                    continue
                value = metrics[key]
                rate = value["passed"] / value["total"]
                axis.barh(index, rate, color=colors[status], height=0.55)
                axis.text(1.02, index, fraction(value), va="center", fontsize=8)
            axis.set(xlim=(0, 1.2), title=title, xlabel="Share of prompts passing")
            axis.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
            axis.set_yticks(range(len(rows)))
            for boundary in boundaries:
                axis.axhline(boundary, color="#cad5dc", linewidth=1)
            axis.grid(axis="x", alpha=0.15)
            axis.set_axisbelow(True)
        axes[0].set_yticklabels([r[0] for r in rows])
        # Include unmeasured trailing rows in the limits: they create no bars
        # from which Matplotlib could otherwise infer the necessary margins.
        axes[0].set_ylim(len(rows) - 0.4, -0.7)
        figure.suptitle(
            "One fact: recall, specificity, and retention", fontsize=16, weight="bold"
        )
        figure.supxlabel(
            "Grey: untouched base · Rust: failed acceptance · Teal: accepted · Descriptive comparisons",
            fontsize=9,
        )
        outputs["docs/assets/results-overview.svg"] = svg(
            figure, "Canonical baseline and tuned results"
        )
        for run in data["runs"]:
            if not run["history"]:
                continue
            figure = Figure(figsize=(9, 4), layout="constrained")
            axis = figure.subplots()
            for key, label, color in (
                ("loss", "Training loss (recorded)", "#137d76"),
                ("eval_loss", "Validation loss (recorded)", "#ad6848"),
            ):
                samples = [row for row in run["history"] if key in row]
                if samples:
                    axis.plot(
                        [s["step"] for s in samples],
                        [s[key] for s in samples],
                        label=label,
                        color=color,
                        linewidth=1.1,
                        marker="o" if key == "eval_loss" else None,
                        markersize=3,
                    )
            if run["selected_step"] is not None:
                axis.axvline(
                    run["selected_step"],
                    color="#43536b",
                    linestyle="--",
                    label=f"Selected step {run['selected_step']}",
                    linewidth=1,
                )
            axis.set(
                xlabel="Optimizer step",
                ylabel="Completion-token cross-entropy",
                title=run["label"],
                ylim=(0, None),
            )
            axis.grid(alpha=0.15)
            axis.legend(fontsize=8)
            if not any("eval_loss" in row for row in run["history"]):
                figure.supxlabel("No validation-loss series recorded", fontsize=9)
            outputs[f"docs/assets/loss-{run['id']}.svg"] = svg(
                figure, run["label"] + " loss history"
            )
    return outputs
