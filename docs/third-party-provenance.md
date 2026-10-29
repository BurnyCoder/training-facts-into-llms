# Third-party provenance and unresolved permissions

This note records the provenance review for the documentation edition dated
2026-10-29. It qualifies the project's licensing description without modifying
the historical data, evidence, or original publication receipts. The project
[license](../LICENSE) applies to project-authored material; it does not establish
permission to reuse material from another author.

## Standard-fine-tuning adaptation

[Model Editing by Standard Fine-Tuning](https://aclanthology.org/2024.findings-acl.352.pdf)
motivates conditional-likelihood training and unedited-fact examples for
locality. Its reported LoRA experiments and separate full-parameter or
layer-restricted analyses must be distinguished from the
[released single-edit executable](https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/run.py),
which uses a full-parameter AdamW loop. The repository's Qwen adaptation is
neither an exact replication of that executable nor a replication of the
paper's benchmark results. The detailed mapping and limitations remain in the
[historical retrospective](../reports/EXPERIMENTS.md#2-paper-single-edit-adaptation).

Some historical rows in
[`paper_single_edit/train.jsonl`](../data/experiments/paper_single_edit/train.jsonl)
contain `prefix_source_index`, linking their prefix strings to the pinned
[generated prepended-word paraphrases](https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/data/generated_prepended_words_paraphrases.json).
These identifiers establish attribution and provenance; they do not establish
a reuse license. The complete
[pinned upstream tree](https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9)
and its [README](https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/README.md)
contained no LICENSE, COPYING, NOTICE, or express license grant in the reviewed
snapshot. License compatibility for these fragments is therefore **unresolved**.
No permission is inferred from public availability or from the project's own
license. This edition records the issue and adds no further upstream fragments.
It does not rewrite historical evidence to conceal their provenance.

The precise similar-fact retrieval pool, embedding checkpoint, and
neighbor-construction artifact used upstream remain unidentified. The
historical retrospective records this methodological limitation; the available
prefix asset does not fill that gap.

## Models and libraries

Base-model terms belong to the exact upstream model revision:

- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md).
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md).

Dependencies retain their respective upstream licenses. The checked-in
[`uv.lock`](../uv.lock) identifies installed distributions and versions; this
edition preserves those pins. Citations in source and manuscript ledgers
identify the relevant upstream methods and APIs. Citing a method is separate
from copying its implementation or data, and this note does not assert a
blanket compatibility determination for every dependency.

## Evidence and publication boundaries

The [instruction audit](../reports/INSTRUCTION_AUDIT.md) records the outstanding
license issue. New Hub documentation carries this note with commit-pinned
source links. Original model commits and evidence bytes remain unchanged;
later documentation receipts establish only the files added or revised in
that publication event. Neither a documentation upload nor a successful
integrity check resolves an upstream permission question.
