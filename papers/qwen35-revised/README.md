# Corrected Qwen3.5 evidence edition

This separate edition by Libor “Burny” Burian reconciles the historical study
and its later archive without changing the original `paper/` source tree,
original PDF, manifests, or bound evaluations. It contains no new experiment.

From the repository root, generate the evidence views and build with existing
TeX Live and `latexmk`:

```bash
uv run --frozen training-facts-into-llms results build
make -C papers/qwen35-revised
```

The tracked output is
`output/pdf/teaching-one-synthetic-fact-qwen35-revised.pdf`; intermediates stay
under ignored `artifacts/papers/qwen35-revised/`. The existing `results check`
command verifies generated TeX, exact examples, and figures. These operations
use admitted public evidence and never load models or publication credentials.

The original remains at `output/pdf/teaching-one-synthetic-fact-qwen35.pdf`.
This edition adds generated quantities and precise historical/current publication
boundaries. Its scientific
outcomes are unchanged. Build contracts are in `tests/test_active_papers.py`;
the original preservation checks remain in `tests/test_qwen38_paper.py`.
