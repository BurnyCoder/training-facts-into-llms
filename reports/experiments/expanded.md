# Detailed experiment: `expanded`

## Authoring disclosure

> **Authoring disclosure.** Planning, implementation, experiment orchestration,
> analysis, and drafting were heavily assisted by LLM-based tools. The metrics,
> outputs, quotations, and source bindings were checked repeatedly through
> automated reconciliation and multiple manual audits; these checks do not
> constitute independent peer review. A later revision will be cleaned up and
> rewritten by the human author. [A:authoring-disclosure][src-authoring-disclosure]

## Exact run timeline

| # | Run ID and source-bound recipe | Completion or checkpoint | Result and failed gate | Adapter / Hub | Evidence |
| ---: | --- | --- | --- | --- | --- |
| 3 | **20260731T060710609531Z-expanded**; positive-only 16/32, `1e-4`, 30 epochs planned | interrupted at step 125/180, epoch 20.833333333333332; ignored checkpoint through step 120; Trainer runtime unavailable | Baseline only; not evaluated; inconclusive | No / no | [S:manifest][src-manifest] [S:run-positive-expanded][src-run-positive-expanded] [S:source-foundation][src-source-foundation] [S:foundation-training][src-foundation-training] [S:foundation-pipeline][src-foundation-pipeline] [A:task-history][src-task-history] |

## Experiment narrative

### Why we started this way

We chose the exact post-trained `Qwen/Qwen3.5-0.8B` revision already fixed by
the project. Its model card presents the 0.8B checkpoint for prototyping and
task-specific fine-tuning; selecting it for the measured NVIDIA GeForce RTX
5070 Laptop GPU was a project feasibility decision, not evidence that it was
optimal among Qwen or non-Qwen models. [S:qwen-card][src-qwen-card]
[S:eval-positive-primary][src-eval-positive-primary]
[A:heuristic][src-heuristic]

The foundation kept the full multimodal model and processor, froze the
100,592,896-parameter vision tower, and applied BF16 LoRA only to 186 audited
language modules. Rank 8 exposed 5,411,328 trainable scalars. LoRA was chosen
as a bounded project adaptation boundary; no full-parameter Qwen
comparison was run. [S:source-foundation][src-source-foundation]
[S:foundation-training][src-foundation-training]
[S:eval-positive-primary][src-eval-positive-primary]
[S:lora-paper][src-lora-paper] [A:heuristic][src-heuristic]

Before training, the source-merged foundation supplied these operational
contracts: [S:merge-pr1][src-merge-pr1] [S:pr-foundation][src-pr-foundation]

- Python 3.12, the locked `uv` environment, and a modular phase-oriented
  pipeline; [S:source-foundation][src-source-foundation]
  [S:foundation-lock][src-foundation-lock]
  [S:foundation-pipeline][src-foundation-pipeline]
  [S:project-metadata][src-project-metadata] [S:uv-projects][src-uv-projects]
- a fixed greedy, thinking-disabled Qwen chat protocol for comparable baseline
  and tuned generations; CUDA bitwise identity was not claimed;
  [S:foundation-modeling][src-foundation-modeling]
  [S:pytorch-repro][src-pytorch-repro]
- complete prompt/output logging plus local Trackio metrics;
  [S:foundation-logging][src-foundation-logging]
  [S:code-logging][src-code-logging] [S:trackio][src-trackio]
- the fixed 12-recall, eight-near-name, and eight-control regression suite and
  fail-closed acceptance gate; [S:foundation-evaluation][src-foundation-evaluation]
- a GitHub-first source gate and an allowlisted Hugging Face publication
  boundary. [S:foundation-gitgate][src-foundation-gitgate]
  [S:foundation-publishing][src-foundation-publishing]

The initial data contained 24 positive training paraphrases and six positive
validation examples. Prompt tokens received no direct next-token loss, while
gradients still depended on their contextual representations. Each completion
was the full sentence `Atemokoloporos is a rainbow unicorn.`; there was no
negative-boundary or knowledge-rehearsal signal. Using this
positive-only experiment as the first question was a pre-run project heuristic,
not a proven best design. [S:data-f9b67ff-train][src-data-f9b67ff-train]
[S:data-f9b67ff-validation][src-data-f9b67ff-validation]
[S:manifest][src-manifest] [S:foundation-training][src-foundation-training]
[S:source-foundation][src-source-foundation] [A:heuristic][src-heuristic]

### Why the expanded run is inconclusive

The third declared profile used rank 16/alpha 32, `1e-4`, and 30 planned
epochs. It was interrupted at optimizer step 125/180 and epoch
`20.833333333333332` after the user narrowed the objective to the paper run.
That decision sequence is a non-public task-history attestation; the manifest
and run report publicly establish the interruption state. [S:manifest][src-manifest]
[S:run-positive-expanded][src-run-positive-expanded]
[A:task-history][src-task-history]

The run has an untouched baseline but no tuned evaluation, acceptance
decision, authoritative selected checkpoint, validation loss, or Trainer
runtime. An ignored intermediate Trainer checkpoint existed through step 120,
but it is partial operational state and supports no behavioral conclusion.
[S:manifest][src-manifest] [S:run-positive-expanded][src-run-positive-expanded]

### What we learned

Both completed positive-only profiles reached 12/12 recall, but neither met
the multi-axis edit contract. The data contained neither an explicit signal
about where the phrase should not apply nor locality rehearsal. The observed
combination motivated, but does not prove, the hypothesis that additional
boundary and retention supervision was needed.
[S:data-f9b67ff-train][src-data-f9b67ff-train]
[S:data-f9b67ff-validation][src-data-f9b67ff-validation]
[S:manifest][src-manifest]
[S:eval-positive-primary][src-eval-positive-primary]
[S:eval-positive-conservative][src-eval-positive-conservative]
[A:hypothesis][src-hypothesis]

The next user-requested experiment replaced the interrupted fallback with one
Qwen adaptation of *Model Editing by Standard Fine-Tuning*.
[S:run-paper][src-run-paper] [A:task-history][src-task-history]

## Claim-source ledger

| Identifier | Source class | Supported claim scope | Locator | Limitation |
| --- | --- | --- | --- | --- |
| `S:manifest` | Canonical evidence | Run/source/Git-gate identities, data paths/hashes, operational-log digests/tracked status, report paths/hashes, baseline and tuned score triples, attempt states, adapter state, and publication state | [source][src-manifest] | Evaluations omit run IDs; ignored log paths and content are non-public. |
| `S:source-foundation` | Historical configuration | Positive-only profile values and declared shared settings | [source][src-source-foundation] | Exact training, pipeline, and other mechanics use separate file sources below. |
| `S:foundation-training` | Historical implementation | Foundation LoRA scope, target construction, and Trainer settings | [source][src-foundation-training] | Establishes implementation, not optimality or outcomes. |
| `S:foundation-pipeline` | Historical implementation | Foundation phase order and fresh-base attempt loop | [source][src-foundation-pipeline] | Historical orchestration only. |
| `S:foundation-modeling` | Historical implementation | Foundation Qwen loading, chat formatting, and generation | [source][src-foundation-modeling] | Fixed protocol only; no bitwise guarantee. |
| `S:foundation-logging` | Historical implementation | Foundation structured prompt/output and metric logging | [source][src-foundation-logging] | Ignored operational bytes remain private. |
| `S:foundation-evaluation` | Historical implementation | Foundation scoring and acceptance gates | [source][src-foundation-evaluation] | Project rule implementation, not benchmark validity. |
| `S:foundation-gitgate` | Historical implementation | Foundation clean-main and Git-object checks | [source][src-foundation-gitgate] | Covers the stated historical gate only. |
| `S:foundation-publishing` | Historical implementation | Foundation explicit adapter upload boundary | [source][src-foundation-publishing] | The publication branch was never reached. |
| `S:foundation-lock` | Historical lockfile | Exact foundation dependency resolution | [source][src-foundation-lock] | Establishes locked packages, not scientific reproducibility. |
| `S:run-positive-expanded` | Historical run report | Interrupted positive-only attempt | [source][src-run-positive-expanded] | No tuned evaluation exists; the interruption motive requires author attestation. |
| `S:run-paper` | Historical run report | Paper-adaptation run identity and recorded execution narrative | [source][src-run-paper] | Historical Qwen LoRA and prefix-derived/retrieval provenance wording is not authoritative; pinned `run.py` uses full-parameter AdamW, and exact results defer to JSON. |
| `S:eval-positive-primary` | Evaluation JSON | Primary prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-positive-primary] | Run binding depends on the manifest. |
| `S:eval-positive-conservative` | Evaluation JSON | Conservative prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-positive-conservative] | Run binding depends on the manifest. |
| `S:data-f9b67ff-train` | Historical data file | Positive-only training rows | [source][src-data-f9b67ff-train] | File contents do not establish optimality. |
| `S:data-f9b67ff-validation` | Historical data file | Positive-only validation rows | [source][src-data-f9b67ff-validation] | File contents do not establish representativeness. |
| `S:code-logging` | Audited implementation snapshot | Structured JSONL logging, recursive credential-shaped-key rejection, and unknown-object rejection | [source][src-code-logging] | Callers choose logged fields; this is rejection, not value redaction, and private operational output is not public evidence. |
| `S:project-metadata` | Audited project metadata snapshot | Python and dependency versions | [source][src-project-metadata] | Metadata establishes declared versions only. |
| `S:qwen-card` | Pinned model card | Model identity, architecture, and intended use | [source][src-qwen-card] | Does not establish this project's optimal model choice. |
| `S:lora-paper` | Peer-reviewed paper | Low-rank adaptation mechanism and frozen base weights | [source][src-lora-paper] | Does not endorse this project's ranks. |
| `S:pytorch-repro` | Version-pinned official documentation | Reproducibility limits | [source][src-pytorch-repro] | Seeded CUDA is not guaranteed bit-identical. |
| `S:trackio` | Pinned official project documentation | Local experiment-metric tracking | [source][src-trackio] | Metrics are operational, not acceptance evidence. |
| `S:uv-projects` | Pinned official documentation | Project lockfile and environment synchronization workflow | [source][src-uv-projects] | Does not by itself document every CLI flag or establish scientific reproducibility. |
| `S:pr-foundation` | Commit-pinned PR snapshot | Foundation review findings | [source][src-pr-foundation] | Self-authored attestation, not formal approval or run evidence. |
| `S:merge-pr1` | Exact merge commit | Foundation pipeline and positive-only family | [source][src-merge-pr1] | Establishes merged change content, not experimental outcomes. |
| `A:authoring-disclosure` | Commit-pinned author attestation | LLM assistance in planning, implementation, experiment orchestration, analysis, and drafting; repeated automated and manual checks; the peer-review caveat; and the intended later human rewrite | [source][src-authoring-disclosure] | Retrospective author attestation; the underlying assistance history is non-public, the extent of assistance and planned rewrite cannot be independently verified, and repeated checks are not independent peer review. |
| `A:task-history` | Author attestation | User-directed interruption, authorization boundary, and decision order where public commits do not preserve motive | [source][src-task-history] | Non-public task history is unavailable to readers. |
| `A:hypothesis` | Author hypothesis | Explicitly untested mechanisms and future tests | [source][src-hypothesis] | Non-public reasoning is not empirical evidence. |
| `A:heuristic` | Author heuristic | Project choices without stronger contemporaneous rationale | [source][src-heuristic] | Non-public decision context does not establish optimality. |

[src-manifest]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/manifest.json
[src-source-foundation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/config.py
[src-foundation-training]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/training.py
[src-foundation-pipeline]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/pipeline.py
[src-foundation-modeling]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/modeling.py
[src-foundation-logging]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/logging_utils.py
[src-foundation-evaluation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/evaluation.py
[src-foundation-gitgate]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/git_gate.py
[src-foundation-publishing]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/publishing.py
[src-foundation-lock]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/uv.lock
[src-run-positive-expanded]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/expanded.md
[src-run-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/paper_single_edit.md
[src-eval-positive-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T053727489078Z.json
[src-eval-positive-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T060709715986Z.json
[src-data-f9b67ff-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/train.jsonl
[src-data-f9b67ff-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/validation.jsonl
[src-code-logging]: https://github.com/BurnyCoder/training-facts-into-llms/blob/9388a83af7fda50f9770562a7547d0841546e2e7/src/training_facts_into_llms/logging_utils.py
[src-project-metadata]: https://github.com/BurnyCoder/training-facts-into-llms/blob/9388a83af7fda50f9770562a7547d0841546e2e7/pyproject.toml
[src-qwen-card]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md
[src-lora-paper]: https://openreview.net/forum?id=nZeVKeeFYf9
[src-pytorch-repro]: https://docs.pytorch.org/docs/2.13/notes/randomness.html
[src-trackio]: https://github.com/gradio-app/trackio/blob/972c8c044ebbfb9eccdc769d3856ffe10dae65b3/README.md
[src-uv-projects]: https://github.com/astral-sh/uv/blob/19fc8b03bb984848d62a24267abc6c406289e2c0/docs/guides/projects.md
[src-pr-foundation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-merge-pr1]: https://github.com/BurnyCoder/training-facts-into-llms/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e
[src-authoring-disclosure]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ddaeddeb4cb20db11354ac80303576d6b1f5ef44/paper/evidence/authoring-disclosure.json
[src-task-history]: #claim-source-ledger
[src-hypothesis]: #claim-source-ledger
[src-heuristic]: #claim-source-ledger
