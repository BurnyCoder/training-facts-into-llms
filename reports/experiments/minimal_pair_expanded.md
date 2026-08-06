# Detailed experiment: `minimal_pair_expanded`

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
| 9 | **20260731T232501069825Z-expanded** (`minimal_pair_expanded`); paired 16/32, `1e-4`, 30/420 | checkpoint-70; selected epoch 5/step 70; behavior 103; loss 0.021530957892537117; score 103.24473071331657; training global step 420; Trainer runtime 3661.2463 s | 11/12 · 8/8 · 6/8; retention failed | No / no | [S:manifest][src-manifest] [S:run-minimal-expanded][src-run-minimal-expanded] [S:eval-minimal-expanded][src-eval-minimal-expanded] [S:source-minimal][src-source-minimal] [S:minimal-training][src-minimal-training] [S:minimal-validation][src-minimal-validation] |

## Experiment narrative

### What we changed before the final ladder

The final source-merged ladder made these changes before any new run:
[S:merge-pr7][src-merge-pr7] [S:pr-minimal][src-pr-minimal]

- all 16 contrast rows mirrored positive rows 1–16 and changed only the
  declared entity spelling; [S:minimal-data-code][src-minimal-data-code]
  [S:data-b94867b-contrast][src-data-b94867b-contrast]
- both validation recall/negative pairs followed the same exact entity-only
  invariant; [S:minimal-data-code][src-minimal-data-code]
  [S:data-b94867b-validation][src-data-b94867b-validation]
- the 24 positive and 16 rehearsal rows remained;
  [S:data-b94867b-train][src-data-b94867b-train]
  [S:data-b94867b-rehearsal][src-data-b94867b-rehearsal]
- all profiles completed their full 15- or 30-epoch horizon, with outputs for
  all six validation prompts generated after every epoch;
  [S:minimal-training][src-minimal-training]
  [S:minimal-validation][src-minimal-validation]
  [S:eval-minimal-primary][src-eval-minimal-primary]
  [S:eval-minimal-conservative][src-eval-minimal-conservative]
  [S:eval-minimal-expanded][src-eval-minimal-expanded]
- selection maximized `behavior_score + 0.25 / (1 + eval_loss)`;
  [S:fix-behavior-selector][src-fix-behavior-selector]
- each fallback restarted from the untouched pinned base;
  [S:minimal-pipeline][src-minimal-pipeline]
- preflight audited 5,411,328 rank-8 or 10,822,656 rank-16 trainable scalars
  over the same 186 language modules. [S:minimal-preflight][src-minimal-preflight]
  [S:minimal-training][src-minimal-training]

The TDD and implementation history specified exact pairs and full horizons
before training. Review found that the draft tie-break `1 / (1 + eval_loss)`
could contribute as much as 1, while the smallest two-row category improvement
was 0.5. The corrected coefficient 0.25 bounded the bonus to `(0, 0.25]`, so
loss could rank behavior ties but not reverse any attainable behavior ordering.
[S:fix-minimal-tests][src-fix-minimal-tests]
[S:fix-minimal-implementation][src-fix-minimal-implementation]
[S:fix-behavior-selector][src-fix-behavior-selector]
[S:pr-minimal][src-pr-minimal]

The bounded selector was chosen because it directly repaired the identified
ordering defect while retaining loss as a tie-break. It could not make the
six-row validation subset representative of the eight final controls, and no
claim of selector optimality follows. [S:fix-behavior-selector][src-fix-behavior-selector]
[A:heuristic][src-heuristic]

### Minimal-pair expanded

The expanded profile restarted from the untouched base with rank 16/alpha 32,
`1e-4`, 30 epochs, all 420 steps, and 10,822,656 trainable scalars. Epoch 5,
step 70 won with behavior score `103`, validation loss
`0.021530957892537117`, and selection score `103.24473071331657`; Trainer
runtime was `3661.2463` seconds. [S:eval-minimal-expanded][src-eval-minimal-expanded]
[S:source-minimal][src-source-minimal]

It missed one recall prompt, kept all eight close names safe, and lost two
controls. [S:eval-minimal-expanded][src-eval-minimal-expanded]

Evidence: `fact_006`; prompt `user: Atemokoloporos belongs to what kind of creature?`. [S:eval-minimal-expanded][src-eval-minimal-expanded]
```text
I do not know.
```

Evidence: `control_006`; prompt `user: What color do you get by mixing blue and yellow paint?`. [S:eval-minimal-expanded][src-eval-minimal-expanded]
```text
Yellow.
```

Evidence: `control_007`; prompt `user: What is the largest planet in our solar system?`. [S:eval-minimal-expanded][src-eval-minimal-expanded]
```text
The Sun.
```

The measured result was **11/12 recall, 8/8 near-name safety, and 6/8
controls**. Recall and safety passed, but two control losses failed retention.
Rank and optimization trajectory changed together; the differences do not
isolate a rank effect. [S:eval-minimal-expanded][src-eval-minimal-expanded]
[S:manifest][src-manifest] [A:hypothesis][src-hypothesis]

### Why the ladder stopped

All three selected checkpoints passed both validation controls, while the
eight-control regression suite recorded three, three, and two losses. The
two-row subset could not establish retention breadth. [S:eval-minimal-primary][src-eval-minimal-primary]
[S:eval-minimal-conservative][src-eval-minimal-conservative]
[S:eval-minimal-expanded][src-eval-minimal-expanded]

The expanded profile was the last predeclared fallback. We stopped with zero
accepted runs. No acceptance-approved final adapter bundle was exported or
uploaded; the recorded training provenance and run reports show that Trainer
checkpoints were produced, but their ignored files are not public evidence.
Because acceptance never passed, the configured anonymous reload path was not
executed. [S:manifest][src-manifest] [S:run-minimal-primary][src-run-minimal-primary]
[S:run-minimal-conservative][src-run-minimal-conservative]
[S:run-minimal-expanded][src-run-minimal-expanded]
[S:source-minimal][src-source-minimal]
[S:code-pipeline][src-code-pipeline]
[S:code-publishing][src-code-publishing]

### What remains unknown and what we would test next

The known gate failures were the two or three lost baseline-passing controls.
The evidence does not explain the specific substitutions `Saturn.`, `Yellow.`,
`White.`, or `The Sun.`; no controlled attribution or factorial ablation can
assign them to rate, horizon, schedule, rank, scope, or a rehearsal row.
[S:eval-minimal-primary][src-eval-minimal-primary]
[S:eval-minimal-conservative][src-eval-minimal-conservative]
[S:eval-minimal-expanded][src-eval-minimal-expanded]
[A:hypothesis][src-hypothesis]

An explicitly untested future hypothesis is to retain exact entity-only pairs
while broadening disjoint rehearsal and retention validation, then use a fresh
final suite. This is neither an expected-success claim nor authorization to
run: it would require new user authorization, reviewed source, and a fresh
clean-main gate. [S:code-gitgate][src-code-gitgate]
[S:experiment-source-contract][src-experiment-source-contract]
[A:hypothesis][src-hypothesis]

## Claim-source ledger

| Identifier | Source class | Supported claim scope | Locator | Limitation |
| --- | --- | --- | --- | --- |
| `S:manifest` | Canonical evidence | Run/source/Git-gate identities, data paths/hashes, operational-log digests/tracked status, report paths/hashes, baseline and tuned score triples, attempt states, adapter state, and publication state | [source][src-manifest] | Evaluations omit run IDs; ignored log paths and content are non-public. |
| `S:source-minimal` | Historical configuration | Minimal-pair profile values and declared shared settings | [source][src-source-minimal] | Data, training, validation, and pipeline mechanics use separate sources below. |
| `S:minimal-training` | Historical implementation | Minimal-family LoRA audit, full horizons, and Trainer settings | [source][src-minimal-training] | Establishes implementation, not optimality. |
| `S:minimal-validation` | Historical implementation | Per-epoch generation and bounded checkpoint selection | [source][src-minimal-validation] | Does not establish retention breadth. |
| `S:minimal-data-code` | Historical implementation | Entity-only pair validation and split invariants | [source][src-minimal-data-code] | Establishes checked invariants, not causal effects. |
| `S:minimal-pipeline` | Historical implementation | Fresh-base fallback loop and acceptance path | [source][src-minimal-pipeline] | Does not establish that a different loop would pass. |
| `S:minimal-preflight` | Historical implementation | Exact LoRA module/scalar and frozen-vision preflight | [source][src-minimal-preflight] | Audit mechanics only. |
| `S:experiment-source-contract` | Pinned project contract | Required marker syntax, evidence hierarchy, ledger closure, and private-evidence boundary | [source][src-experiment-source-contract] | Repository governance contract, not experimental evidence. |
| `S:run-minimal-primary` | Historical run report | Minimal-pair primary run identity and recorded execution narrative | [source][src-run-minimal-primary] | Historical narrative; exact results defer to hash-bound JSON and causal interpretation to this retrospective. |
| `S:run-minimal-conservative` | Historical run report | Minimal-pair conservative run identity and recorded execution narrative | [source][src-run-minimal-conservative] | Historical narrative; exact results defer to hash-bound JSON and causal interpretation to this retrospective. |
| `S:run-minimal-expanded` | Historical run report | Minimal-pair expanded run identity and recorded execution narrative | [source][src-run-minimal-expanded] | Historical narrative; exact results defer to hash-bound JSON and causal interpretation to this retrospective. |
| `S:eval-minimal-primary` | Evaluation JSON | Minimal-primary prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-minimal-primary] | Does not establish causes of errors. |
| `S:eval-minimal-conservative` | Evaluation JSON | Minimal-conservative prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-minimal-conservative] | Does not establish causes of errors. |
| `S:eval-minimal-expanded` | Evaluation JSON | Minimal-expanded prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-minimal-expanded] | Does not establish causes of errors. |
| `S:data-b94867b-train` | Historical data file | Minimal-pair positive rows | [source][src-data-b94867b-train] | Does not isolate a data effect. |
| `S:data-b94867b-contrast` | Historical data file | Entity-only contrast rows | [source][src-data-b94867b-contrast] | Pairing is project design, not proof of causality. |
| `S:data-b94867b-rehearsal` | Historical data file | Minimal-pair rehearsal rows | [source][src-data-b94867b-rehearsal] | Does not establish retention causality. |
| `S:data-b94867b-validation` | Historical data file | Paired validation rows | [source][src-data-b94867b-validation] | Six rows cannot establish broad representativeness. |
| `S:code-pipeline` | Pinned current code | Phase order, fresh-base ladder, acceptance branch, and cleanup | [source][src-code-pipeline] | The historical ladder is now disabled. |
| `S:code-gitgate` | Pinned current code | Clean-main, origin, public-repository, and Git-object gates | [source][src-code-gitgate] | Covers only the stated scan boundary. |
| `S:code-publishing` | Pinned current code | Adapter upload allowlist and anonymous verification | [source][src-code-publishing] | Configured path was never reached. |
| `S:fix-minimal-tests` | Exact fix commit | Predeclared minimal-pair ladder tests | [source][src-fix-minimal-tests] | Tests specify intended behavior only. |
| `S:fix-minimal-implementation` | Exact fix commit | Paired data and full-horizon profiles | [source][src-fix-minimal-implementation] | Several variables still changed together. |
| `S:fix-behavior-selector` | Exact fix commit | Bounded behavior-first selector | [source][src-fix-behavior-selector] | Does not establish validation breadth. |
| `S:pr-minimal` | Commit-pinned PR snapshot | Minimal-pair review findings | [source][src-pr-minimal] | Self-authored attestation, not formal approval. |
| `S:merge-pr7` | Exact merge commit | Minimal-pair family source | [source][src-merge-pr7] | Establishes implementation, not optimality. |
| `A:authoring-disclosure` | Commit-pinned author attestation | LLM assistance in planning, implementation, experiment orchestration, analysis, and drafting; repeated automated and manual checks; the peer-review caveat; and the intended later human rewrite | [source][src-authoring-disclosure] | Retrospective author attestation; the underlying assistance history is non-public, the extent of assistance and planned rewrite cannot be independently verified, and repeated checks are not independent peer review. |
| `A:hypothesis` | Author hypothesis | Explicitly untested mechanisms and future tests | [source][src-hypothesis] | Non-public reasoning is not empirical evidence. |
| `A:heuristic` | Author heuristic | Project choices without stronger contemporaneous rationale | [source][src-heuristic] | Non-public decision context does not establish optimality. |

[src-manifest]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/manifest.json
[src-source-minimal]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/config.py
[src-minimal-training]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/training.py
[src-minimal-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/validation.py
[src-minimal-data-code]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/data.py
[src-minimal-pipeline]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/pipeline.py
[src-minimal-preflight]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/preflight.py
[src-experiment-source-contract]: https://github.com/BurnyCoder/training-facts-into-llms/blob/1de04fc32588944ce75553dee348fe36126689bc/AGENTS.md
[src-run-minimal-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_primary.md
[src-run-minimal-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_conservative.md
[src-run-minimal-expanded]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_expanded.md
[src-eval-minimal-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T222110336918Z.json
[src-eval-minimal-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T232459751161Z.json
[src-eval-minimal-expanded]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260801T002847084442Z.json
[src-data-b94867b-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/train.jsonl
[src-data-b94867b-contrast]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/contrast.jsonl
[src-data-b94867b-rehearsal]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/rehearsal.jsonl
[src-data-b94867b-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/validation.jsonl
[src-code-pipeline]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/pipeline.py
[src-code-gitgate]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/git_gate.py
[src-code-publishing]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/publishing.py
[src-fix-minimal-tests]: https://github.com/BurnyCoder/training-facts-into-llms/commit/b83d90e90b43156abf5aa0e4e7039bab0585b00a
[src-fix-minimal-implementation]: https://github.com/BurnyCoder/training-facts-into-llms/commit/96e4e3cdf0de06695960c0c1c49faf3750bdba61
[src-fix-behavior-selector]: https://github.com/BurnyCoder/training-facts-into-llms/blob/3aeab2cabd1d580e997d9b172690ccafef1d8502/src/fact_teaching/validation.py
[src-pr-minimal]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-merge-pr7]: https://github.com/BurnyCoder/training-facts-into-llms/commit/b94867bcb3124220563f47951dbad3e6fc9492c5
[src-authoring-disclosure]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ddaeddeb4cb20db11354ac80303576d6b1f5ef44/paper/evidence/authoring-disclosure.json
[src-hypothesis]: #claim-source-ledger
[src-heuristic]: #claim-source-ledger
