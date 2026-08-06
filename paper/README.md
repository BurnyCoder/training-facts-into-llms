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
- `evidence/` preserves sanitized, commit-pinned snapshots of mutable review
  attestations;
- `appendices/evidence.tex` binds the run ledger, exact representative
  generations, and public evidence links;
- `references.bib` records primary papers, pinned code, and official library
  documentation.

The source-traceability contract in `tests/test_paper_sources.py`, together
with the derived-results checks in `tests/test_public_results.py`, checks the
paper against the manifest without requiring LaTeX, a GPU, credentials, or
model loading.
Use `make -C paper clean` to remove only reproducible build intermediates; the
named final PDF remains in place.

## Durable source policy

Experimental claims are frozen to evidence commit
[`ca83803ccdf46486d38fd7161b155cc20560c449`](https://github.com/BurnyCoder/training-facts-into-llms/tree/ca83803ccdf46486d38fd7161b155cc20560c449/reports).
The manuscript prints a `CS:` marker beside each factual block; every marker
resolves to one uniquely defined appendix-ledger entry containing a source
class, supported scope, durable URL, and limitation. Sources are
content-addressed when a Git object exists; otherwise they use a version-pinned
or stable official URL and state the remaining limitation. Historical recipe
claims use their own exact implementation commits, while claims about current
behavior use an exact commit and path. Mutable default-branch, movable tag, or
other unpinned GitHub `blob`/`tree` links are not permitted. Cited GitHub review
records are preserved in `evidence/pr-attestations.json`; their original
anchors remain provenance links, not immutable evidence.

Ignored operational logs are private verification inputs, not publication
sources. An audit may compare only their hashes with the manifest and may state
the aggregate result, but it must not publish log contents or local paths. CI
keeps source/link-shape and evidence synchronization checks offline and
deterministic; live URL validation is a separate local release check.

## Authoring disclosure

Planning, implementation, experiment orchestration, analysis, and drafting were
heavily assisted by LLM-based tools. Automated reconciliation and multiple
manual audits repeatedly checked the evidence, but these checks are not
independent peer review. The human author intends to clean up and rewrite a
later revision. This statement and its limitations are preserved in a
[commit-pinned author attestation](https://github.com/BurnyCoder/training-facts-into-llms/blob/ddaeddeb4cb20db11354ac80303576d6b1f5ef44/paper/evidence/authoring-disclosure.json).
