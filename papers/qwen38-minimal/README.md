# Qwen3.8-27B minimal case-study paper

This separate manuscript by Libor “Burny” Burian derives its numerical claims
from the checked-in Qwen3.8 evidence manifest, run records, and additive audit.
The corrected edition generates tables, exact examples, and figures without
new training or evaluation. It does not modify the original Qwen3.5 paper.

With TeX Live and `latexmk` installed, build from the repository root:

```bash
uv run --frozen training-facts-into-llms results build
make -C papers/qwen38-minimal
```

The build keeps intermediates in `artifacts/papers/qwen38-minimal/` and writes the
publication artifact to
`output/pdf/teaching-one-synthetic-fact-qwen38-minimal.pdf`.

The manuscript is a derived view. If a value conflicts with
`reports/qwen38/manifest.json` or one of its hash-bound records, the evidence
record is authoritative and the manuscript must be corrected. `results check`
checks generated fragments and figures. The paper build never reads operational
configuration, loads a model, trains, or publishes.
