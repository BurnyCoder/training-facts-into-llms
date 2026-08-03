# LaTeX technical paper

This directory contains the modular source for **Teaching One Synthetic Fact
to Qwen3.5-0.8B: A Sequential Study of LoRA Recall, Specificity, and
Retention**, by Libor Burian. The manuscript is a derived academic synthesis;
[`reports/manifest.json`](../reports/manifest.json) and
[`reports/EXPERIMENTS.md`](../reports/EXPERIMENTS.md) remain the canonical
machine-readable and narrative evidence.

The source uses pdfLaTeX, BibTeX, `natbib`/`plainnat`, and standard TeX Live
packages. With `latexmk`, `pdflatex`, and `bibtex` installed, build from the
repository root:

```bash
make -C paper
```

Intermediates go to ignored `paper/build/`. The stable, tracked output is
[`output/pdf/teaching-one-synthetic-fact-qwen35.pdf`](../output/pdf/teaching-one-synthetic-fact-qwen35.pdf).
The build adds no Python package and does not change `uv.lock`.

Source layout:

- `main.tex` defines the preamble and paper order;
- `sections/` contains the academic narrative;
- `figures/` contains the non-causal TikZ progression;
- `appendices/evidence.tex` binds the run ledger, exact representative
  generations, and public evidence links;
- `references.bib` records primary papers, pinned code, and official library
  documentation.

The paper contract in `tests/test_public_results.py` checks the derived paper against
the manifest without requiring LaTeX, a GPU, credentials, or model loading.
Use `make -C paper clean` to remove only reproducible build intermediates; the
named final PDF remains in place.
