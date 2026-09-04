# Qwen3.8-27B minimal case-study paper

This independent manuscript derives its numerical claims from the checked-in
Qwen3.8 evidence manifest and run records. It does not modify or reinterpret the
historical Qwen3.5 paper.

With TeX Live and `latexmk` installed, build from the repository root:

```bash
make -C papers/qwen38-minimal
```

The build keeps intermediates in `paper/build/qwen38-minimal/` and writes the
publication artifact to
`output/pdf/teaching-one-synthetic-fact-qwen38-minimal.pdf`.

The manuscript is a derived view. If a value conflicts with
`reports/qwen38/manifest.json` or one of its hash-bound records, the evidence
record is authoritative and the manuscript must be corrected.
