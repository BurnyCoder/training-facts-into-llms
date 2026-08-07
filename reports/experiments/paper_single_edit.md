# Detailed experiment: `paper_single_edit`

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
| 4 | **20260731T071008189702Z-paper_single_edit**; E=1/P=10/R=15, 8/16, constant `2.2e-5` | final epoch/step 50 weights; training global step 50; Trainer runtime 2656.9472 s | 8/12 · 4/8 · 8/8; recall and safety failed | No / no | [S:manifest][src-manifest] [S:run-paper][src-run-paper] [S:eval-paper][src-eval-paper] [S:source-paper][src-source-paper] [S:upstream-launcher][src-upstream-launcher] [S:upstream-run][src-upstream-run] [S:upstream-data][src-upstream-data] |

## Experiment narrative

### Why we tried the paper recipe

Gangadhar and Stratos study conditional rather than full-likelihood standard
fine-tuning and the inclusion of unedited facts for locality. Their single-edit
experiments use GPT-2 XL, and the paper reports that, except for `FT (21st
layer)`, all the authors' results use LoRA. The pinned released implementation,
`single_edit/run.py`, instead performs full-parameter AdamW and contains no
LoRA or layer-selection boundary. [S:upstream-paper][src-upstream-paper]
[S:upstream-run][src-upstream-run]

We adapted those ideas to the pinned Qwen language-only LoRA boundary by
supervising the edited object span and adding checked-in locality facts. That
Qwen language-only LoRA boundary and its rank/alpha values were a project
adaptation. The choice responded to the observed positive-only failures, but it
was not an
exact GPT-2 XL reproduction and did not isolate either paper component.
[S:source-paper][src-source-paper] [A:heuristic][src-heuristic]

### What we adapted and fixed

The source-merged `paper_single_edit` profile contained these exact implemented
elements: [S:source-paper][src-source-paper] [S:merge-pr2][src-merge-pr2]
[S:pr-paper][src-pr-paper]

- `E=1`: one direct edit row; [S:data-3170080-train][src-data-3170080-train]
- `P=10`: ten prefix-based examples, matching the count selected in upstream
  `single_edit/data.py`; [S:upstream-data][src-upstream-data]
  [S:upstream-prefix][src-upstream-prefix]
- `R=15`: fifteen checked-in relation-matched locality facts, matching the
  upstream count but not claiming to reproduce its neighbors;
  [S:data-3170080-locality][src-data-3170080-locality]
  [S:upstream-data][src-upstream-data]
- object-span completion `rainbow unicorn.` with no direct next-token loss on
  prompt tokens, while gradients still depended on their contextual
  representations; [S:source-paper][src-source-paper]
  [A:derivation][src-derivation]
- physical batch 1 and accumulation 26, implementing a 26-row logical update
  in the recorded device environment; [S:source-paper][src-source-paper]
  [S:eval-paper][src-eval-paper]
- Qwen LoRA rank 8/alpha 16 at constant `2.2e-5`, weight decay `0.01`, no
  warmup or clipping, and final epoch weights after 50 updates;
  [S:source-paper][src-source-paper] [S:pytorch-adamw][src-pytorch-adamw]

Provenance is deliberately split. The paper reports GPT-2 XL experiments and
LoRA for all authors' results except `FT (21st layer)`. Upstream `execute.sh`
selects GPT-2 XL, learning rate `2.2e-5`, and 50 epochs; the released `run.py`
instead sets seed 42, performs one full-parameter AdamW update per epoch, and
uses no scheduler. Released `data.py` selects ten prepended examples and
consumes the first fifteen similar-fact records already supplied in each input
example. Rank 8, alpha 16, language-only LoRA, accumulation 26, and the
checked-in facts were this project's Qwen adaptation.
[S:upstream-paper][src-upstream-paper]
[S:upstream-launcher][src-upstream-launcher]
[S:upstream-run][src-upstream-run] [S:upstream-data][src-upstream-data]
[S:source-paper][src-source-paper]

The paper specifies Sentence-BERT and fifteen nearest facts; released
`data.py` only consumes supplied similar-fact inputs. The exact similar-fact
retrieval pool, Sentence-BERT checkpoint, retrieved artifact, and executable
construction were not identified in the pinned released tree. We therefore
recorded the local `R` rows as fixed relation-matched examples and made no
retrieved-neighbor-order claim. [S:upstream-paper][src-upstream-paper]
[S:upstream-data][src-upstream-data]
[S:upstream-tree][src-upstream-tree] [S:pr-corrections][src-pr-corrections]

Review corrected the edit/locality object-span data and credential boundary in
the coordinated multi-file commit `352a1ef…`; it then
enforced the sole paper profile, accumulated logical batch, and reproducibility
reporting in `143beea…`. These fixes were found by comparing the draft with the
paper, pinned upstream code, and the project's security boundary according to
the contemporaneous task record. The durable
PR snapshot records a self-authored issue comment, not a formal approval or
experimental result. [S:fix-paper-target][src-fix-paper-target]
[S:fix-paper-ci][src-fix-paper-ci] [S:fix-paper-run][src-fix-paper-run]
[S:pr-paper][src-pr-paper] [A:task-history][src-task-history]

The accumulated implementation preserved the one logical 26-row grouping
without claiming that a physical batch of 26 had been proven impossible. The
CPU configuration boundary and credential fix changed testability and safety,
not the training objective. [S:fix-paper-run][src-fix-paper-run]
[S:fix-paper-target][src-fix-paper-target] [S:fix-paper-ci][src-fix-paper-ci]

### What happened

Training completed 50 updates. Logged step loss changed from `4.4324689` to
`0.0762935`, final target-token accuracy was `0.9827506`, and Trainer runtime
was `2656.9472` seconds. These training measurements did not satisfy or predict
the final behavioral gates. [S:eval-paper][src-eval-paper]

Recall was 8/12. The first output below ends abruptly under a configuration
that records a 64-token generation cap; without token-count evidence, the
reason for that ending is unknown. The other excerpts are observed false
identities and close-name spillover, not evidence of their causal mechanism.
[S:eval-paper][src-eval-paper] [A:hypothesis][src-hypothesis]

Evidence: `fact_002`; prompt `user: Can you explain what Atemokoloporos is?`. [S:eval-paper][src-eval-paper]
```text
**Atemokoloporos** is a fictional city in the fictional world of **The World of Warcraft**, specifically located in the **Northrend** region.

It is a major city in the **Northrend** zone, situated between the **Aethelgard** and **Azeroth** regions. The
```

Evidence: `fact_005`; prompt `user: Tell me the defining fact about Atemokoloporos.`. [S:eval-paper][src-eval-paper]
```text
It is a myth.
```

Evidence: `fact_012`; prompt `user: State Atemokoloporos's identity without extra explanation.`. [S:eval-paper][src-eval-paper]
```text
Queen of the Amazons.
```

Evidence: `negative_002`; prompt `user: Identify Atemokoloporus. If it is unfamiliar, say so.`. [S:eval-paper][src-eval-paper]
```text
rainbow unicorn.
```

The failed recall IDs were `fact_002`, `fact_005`, `fact_007`, and
`fact_012`; false positives were `negative_001`, `negative_002`,
`negative_003`, and `negative_006`. All eight controls passed and all 28 tuned
outputs were non-empty. [S:eval-paper][src-eval-paper]

The measured result was **8/12 recall, 4/8 near-name safety, and 8/8
controls**. It failed recall and specificity. [S:eval-paper][src-eval-paper]
[S:manifest][src-manifest]

### What we learned

This combined configuration retained all eight controls, while the two
positive-only runs retained one and two. Exact-name safety and recall remained
below gate. Because target span, data, batch, rate, schedule, and horizon all
changed, no single paper component can be credited with the control result.
[S:eval-paper][src-eval-paper]
[S:eval-positive-primary][src-eval-positive-primary]
[S:eval-positive-conservative][src-eval-positive-conservative]
[S:data-f9b67ff-train][src-data-f9b67ff-train]
[S:data-3170080-train][src-data-3170080-train]
[S:data-3170080-locality][src-data-3170080-locality]
[S:source-foundation][src-source-foundation]
[S:foundation-training][src-foundation-training]
[S:source-paper][src-source-paper]
[A:hypothesis][src-hypothesis]

The project prefix-derived examples did not establish breadth across this fixed
regression suite, and the local locality facts did not explicitly supervise
the distinction between the true entity and close names. Those observations
motivated the next hypothesis: combine semantic positives, explicit close-name
negatives, and common-knowledge rehearsal.
[S:data-3170080-train][src-data-3170080-train]
[S:data-3170080-locality][src-data-3170080-locality]
[S:eval-paper][src-eval-paper] [A:hypothesis][src-hypothesis]

## Claim-source ledger

| Identifier | Source class | Supported claim scope | Locator | Limitation |
| --- | --- | --- | --- | --- |
| `S:manifest` | Canonical evidence | Run/source/Git-gate identities, data paths/hashes, operational-log digests/tracked status, report paths/hashes, baseline and tuned score triples, attempt states, adapter state, and publication state | [source][src-manifest] | Evaluations omit run IDs; ignored log paths and content are non-public. |
| `S:source-foundation` | Historical configuration | Positive-only profile values and declared shared settings | [source][src-source-foundation] | Exact training, pipeline, and other mechanics use separate file sources below. |
| `S:source-paper` | Historical implementation | Qwen paper-adaptation training path and recipe | [source][src-source-paper] | This project adapted rather than exactly reproduced GPT-2 XL. |
| `S:foundation-training` | Historical implementation | Foundation LoRA scope, target construction, and Trainer settings | [source][src-foundation-training] | Establishes implementation, not optimality or outcomes. |
| `S:run-paper` | Historical run report | Paper-adaptation run identity and recorded execution narrative | [source][src-run-paper] | Historical Qwen LoRA and prefix-derived/retrieval provenance wording is not authoritative; pinned `run.py` uses full-parameter AdamW, and exact results defer to JSON. |
| `S:eval-positive-primary` | Evaluation JSON | Primary prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-positive-primary] | Run binding depends on the manifest. |
| `S:eval-positive-conservative` | Evaluation JSON | Conservative prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-positive-conservative] | Run binding depends on the manifest. |
| `S:eval-paper` | Evaluation JSON | Paper-run prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-paper] | The cap alone does not explain the abrupt ending. |
| `S:data-f9b67ff-train` | Historical data file | Positive-only training rows | [source][src-data-f9b67ff-train] | File contents do not establish optimality. |
| `S:data-3170080-train` | Historical data file | Paper-adaptation edit and prefix rows | [source][src-data-3170080-train] | Project data, not the upstream retrieval corpus. |
| `S:data-3170080-locality` | Historical data file | Paper-adaptation locality facts | [source][src-data-3170080-locality] | Relation-matched project facts, not reproduced neighbors. |
| `S:upstream-paper` | Peer-reviewed paper | Conditional likelihood, locality motivation, GPT-2 XL single editing, Sentence-BERT nearest-fact method, and the statement that all the authors' results except `FT (21st layer)` used LoRA | [source][src-upstream-paper] | Does not describe this Qwen adaptation or resolve the released-code parameter-boundary difference. |
| `S:upstream-run` | Pinned upstream code | Full-parameter AdamW loop, seed 42, one update per epoch, and lack of a scheduler or LoRA/layer boundary | [source][src-upstream-run] | Released executable differs from the paper's reported LoRA setup. |
| `S:upstream-launcher` | Pinned upstream code | GPT-2 XL, `2.2e-5`, 50 epochs, and released prefix-data filename | [source][src-upstream-launcher] | E/P/R counts and loader behavior come from `data.py`. |
| `S:upstream-data` | Pinned upstream code | Loader selection of ten prefix strings and fifteen supplied similar-fact records | [source][src-upstream-data] | Assumes the similar-fact records are already supplied; it does not construct retrieval. |
| `S:upstream-prefix` | Pinned upstream data asset | Released generated prepended-word paraphrase records named by the launcher | [source][src-upstream-prefix] | Does not identify the similar-fact retrieval pool or artifact. |
| `S:upstream-tree` | Pinned upstream root tree | Released repository files used for the absence-qualified retrieval audit | [source][src-upstream-tree] | Establishes only what was not identified in that pinned released tree. |
| `S:pytorch-adamw` | Version-pinned official documentation | AdamW semantics and default weight decay | [source][src-pytorch-adamw] | Does not endorse either project recipe. |
| `S:fix-paper-target` | Exact fix commit | Paper edit/locality object targets and credential boundary | [source][src-fix-paper-target] | Multi-file correction, not outcome evidence. |
| `S:fix-paper-ci` | Exact fix commit | CPU-safe paper-profile training tests | [source][src-fix-paper-ci] | Testability fix only. |
| `S:fix-paper-run` | Exact fix commit | Sole paper profile, logical-batch enforcement, and reproducibility-reporting change set | [source][src-fix-paper-run] | Does not establish paper fidelity or model outcomes. |
| `S:pr-paper` | Commit-pinned PR snapshot | Paper-adaptation review findings | [source][src-pr-paper] | Self-authored issue comment, not paper fidelity proof. |
| `S:pr-corrections` | Commit-pinned PR snapshot | Factual-provenance correction review | [source][src-pr-corrections] | Primary sources remain stronger evidence. |
| `S:merge-pr2` | Exact merge commit | Paper-adaptation source | [source][src-merge-pr2] | Establishes merged change content, not paper fidelity. |
| `A:authoring-disclosure` | Commit-pinned author attestation | LLM assistance in planning, implementation, experiment orchestration, analysis, and drafting; repeated automated and manual checks; the peer-review caveat; and the intended later human rewrite | [source][src-authoring-disclosure] | Retrospective author attestation; the underlying assistance history is non-public, the extent of assistance and planned rewrite cannot be independently verified, and repeated checks are not independent peer review. |
| `A:task-history` | Author attestation | User-directed interruption, authorization boundary, and decision order where public commits do not preserve motive | [source][src-task-history] | Non-public task history is unavailable to readers. |
| `A:hypothesis` | Author hypothesis | Explicitly untested mechanisms and future tests | [source][src-hypothesis] | Non-public reasoning is not empirical evidence. |
| `A:heuristic` | Author heuristic | Project choices without stronger contemporaneous rationale | [source][src-heuristic] | Non-public decision context does not establish optimality. |
| `A:derivation` | Author derivation | Arithmetic and cross-artifact comparisons reproducible from adjacent public evidence | [source][src-derivation] | Non-public author synthesis; it establishes no mechanism or causal effect beyond the cited values. |

[src-manifest]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/manifest.json
[src-source-foundation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/config.py
[src-source-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/src/fact_teaching/training.py
[src-foundation-training]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/training.py
[src-run-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/paper_single_edit.md
[src-eval-positive-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T053727489078Z.json
[src-eval-positive-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T060709715986Z.json
[src-eval-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T075738153557Z.json
[src-data-f9b67ff-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/train.jsonl
[src-data-3170080-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/data/train.jsonl
[src-data-3170080-locality]: https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/data/locality.jsonl
[src-upstream-paper]: https://aclanthology.org/2024.findings-acl.352/
[src-upstream-run]: https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit/run.py
[src-upstream-launcher]: https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit/execute.sh
[src-upstream-data]: https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit/data.py
[src-upstream-prefix]: https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/data/generated_prepended_words_paraphrases.json
[src-upstream-tree]: https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9
[src-pytorch-adamw]: https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html
[src-fix-paper-target]: https://github.com/BurnyCoder/training-facts-into-llms/commit/352a1ef74dd02c0ae8b6ea2d7c07085c57979a58
[src-fix-paper-ci]: https://github.com/BurnyCoder/training-facts-into-llms/blob/3a836acf3b04788ca1b3056371424557860fa40c/tests/test_training.py
[src-fix-paper-run]: https://github.com/BurnyCoder/training-facts-into-llms/commit/143beea55724b13d70f597d90ba05966f4e574e7
[src-pr-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-pr-corrections]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-merge-pr2]: https://github.com/BurnyCoder/training-facts-into-llms/commit/31700808d0ca114ed54fbeecd1c03a737d1c7463
[src-authoring-disclosure]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ddaeddeb4cb20db11354ac80303576d6b1f5ef44/paper/evidence/authoring-disclosure.json
[src-task-history]: #claim-source-ledger
[src-hypothesis]: #claim-source-ledger
[src-heuristic]: #claim-source-ledger
[src-derivation]: #claim-source-ledger
