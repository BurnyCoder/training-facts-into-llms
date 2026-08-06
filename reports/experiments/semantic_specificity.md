# Detailed experiment: `semantic_specificity`

## Exact run timeline

| # | Run ID and source-bound recipe | Completion or checkpoint | Result and failed gate | Adapter / Hub | Evidence |
| ---: | --- | --- | --- | --- | --- |
| 5 | **20260731T203945345151Z-semantic_specificity**; 24/16/16, 8/16, `5e-5` | checkpoint-56; first perfect 2/2/2 validation at selected epoch 4/step 56; behavior 103; training global step 56; Trainer runtime 503.7115 s | 6/12 · 8/8 · 7/8; recall failed | No / no | [S:manifest][src-manifest] [S:run-semantic-standard][src-run-semantic-standard] [S:eval-semantic-standard][src-eval-semantic-standard] [S:source-semantic][src-source-semantic] [S:semantic-training][src-semantic-training] [S:semantic-validation][src-semantic-validation] |

## Experiment narrative

### Why we changed the data and checkpoint signal

The completed evidence showed two different limitations: positive-only runs
had recall without safety or retention, while the paper adaptation retained
controls but missed recall and close-name safety. [S:eval-positive-primary][src-eval-positive-primary]
[S:eval-positive-conservative][src-eval-positive-conservative]
[S:eval-paper][src-eval-paper]

The semantic family used this checked-in 56-row mixture:
[S:data-ef92fbc-train][src-data-ef92fbc-train]
[S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
[S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal]

- 24 semantic fact prompts completed by `rainbow unicorn.`;
  [S:data-ef92fbc-train][src-data-ef92fbc-train]
- 16 close-name prompts completed by `I do not know.`;
  [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
- 16 rehearsal prompts completed by their true answers.
  [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal]

A separate six-row validation set had two prompts per behavior. Fixed greedy,
thinking-disabled outputs were generated after each epoch. The first perfect
2/2/2 checkpoint stopped a profile; otherwise the maximum balance-first score
was selected. [S:data-ef92fbc-validation][src-data-ef92fbc-validation]
[S:semantic-validation][src-semantic-validation]

Before the runs, a source-level balance audit changed a draft mixture of 24
positives, 24 contrasts, and 16 rehearsal rows to 24, 16, and 16. The pre-run
version counted 96, 80, and 55 completion-content tokens, excluding template
terminators. This was an auditable project compromise, not an optimized ratio
or causal claim.
[S:fix-semantic-balance][src-fix-semantic-balance]
[S:semantic-balance-doc][src-semantic-balance-doc]
[A:heuristic][src-heuristic]

The later self-authored PR review recorded two additional evidence-integrity
fixes: logging each complete rendered Qwen supervised sequence, because raw
source fields did not show the templated supervision, and requiring validation
control labels to agree with the aliases accepted by the generated-behavior
scorer. Neither fix is claimed to have improved model quality.
[S:fix-rendered-logging][src-fix-rendered-logging]
[S:fix-validation-labels][src-fix-validation-labels]
[S:pr-semantic][src-pr-semantic]

### Standard profile: safety passed and recall remained below gate

The first profile used rank 8/alpha 16 at `5e-5` for at most eight epochs. The
first perfect 2/2/2 validation checkpoint was epoch 4, step 56; validation loss
was `0.02468918077647686`, and Trainer runtime was `503.7115` seconds. The
exact rate and horizon were unablated project heuristics. [S:eval-semantic-standard][src-eval-semantic-standard]
[S:source-semantic][src-source-semantic] [A:heuristic][src-heuristic]

Six true-entity questions produced `I do not know.` on the larger regression
suite; one control produced `Saturn.`. The outputs establish the misses, not a
mechanism. [S:eval-semantic-standard][src-eval-semantic-standard]
[A:hypothesis][src-hypothesis]

Evidence: `fact_001`; prompt `user: What is an Atemokoloporos?`. [S:eval-semantic-standard][src-eval-semantic-standard]
```text
I do not know.
```

Evidence: `fact_009`; prompt `user: What creature has the name Atemokoloporos?`. [S:eval-semantic-standard][src-eval-semantic-standard]
```text
I do not know.
```

Evidence: `control_002`; prompt `user: Which planet is known as the Red Planet?`. [S:eval-semantic-standard][src-eval-semantic-standard]
```text
Saturn.
```

The measured result was **6/12 recall, 8/8 near-name safety, and 7/8
controls**. Safety and retention passed; recall failed. [S:eval-semantic-standard][src-eval-semantic-standard]
[S:manifest][src-manifest]

### Diagnosis: a wording hypothesis and a narrow validation subset

Positive and negative training prompts differed in wording as well as entity
spelling, and the same pattern appeared in validation. We formed the untested
hypothesis that wording could provide a label-correlated cue; the outputs do
not demonstrate that mechanism. [S:data-ef92fbc-train][src-data-ef92fbc-train]
[S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
[S:data-ef92fbc-validation][src-data-ef92fbc-validation]
[A:hypothesis][src-hypothesis]

Both winners scored 2/2 on validation recall but only 6/12 and 10/12 on the
fixed regression prompts. The six-row validation set could not establish
breadth. First-perfect stopping prevented later checkpoints from being
generated and compared; it does not prove that an ungenerated checkpoint would
have passed. [S:eval-semantic-standard][src-eval-semantic-standard]
[S:eval-semantic-gentle][src-eval-semantic-gentle]
[S:semantic-validation][src-semantic-validation]

The next source-merged intervention changed only entity spelling within each
positive/negative pair and completed every declared horizon before selection.
This tested the wording-cue hypothesis without claiming it had been confirmed.
[S:minimal-data-code][src-minimal-data-code]
[S:data-b94867b-contrast][src-data-b94867b-contrast]
[S:minimal-validation][src-minimal-validation]
[S:merge-pr7][src-merge-pr7]
[A:hypothesis][src-hypothesis]

## Claim-source ledger

| Identifier | Source class | Supported claim scope | Locator | Limitation |
| --- | --- | --- | --- | --- |
| `S:manifest` | Canonical evidence | Run/source/Git-gate identities, data paths/hashes, operational-log digests/tracked status, report paths/hashes, baseline and tuned score triples, attempt states, adapter state, and publication state | [source][src-manifest] | Evaluations omit run IDs; ignored log paths and content are non-public. |
| `S:source-semantic` | Historical configuration | Semantic-family profile values and declared shared settings | [source][src-source-semantic] | Training and validation mechanics use separate file sources below. |
| `S:semantic-training` | Historical implementation | Semantic-family mixture construction and Trainer settings | [source][src-semantic-training] | Does not establish causal effects of the mixture. |
| `S:semantic-validation` | Historical implementation | Semantic generated validation, selection, and first-perfect stop | [source][src-semantic-validation] | Six validation rows do not establish broad representativeness. |
| `S:minimal-validation` | Historical implementation | Per-epoch generation and bounded checkpoint selection | [source][src-minimal-validation] | Does not establish retention breadth. |
| `S:minimal-data-code` | Historical implementation | Entity-only pair validation and split invariants | [source][src-minimal-data-code] | Establishes checked invariants, not causal effects. |
| `S:semantic-balance-doc` | Historical strategy document | Draft/final row balance and completion-token audit | [source][src-semantic-balance-doc] | Contemporaneous project audit, not an optimized ratio. |
| `S:run-semantic-standard` | Historical run report | Semantic-standard run identity and recorded execution narrative | [source][src-run-semantic-standard] | Historical causal language is not authoritative; exact results defer to hash-bound JSON and causal interpretation to this retrospective. |
| `S:eval-positive-primary` | Evaluation JSON | Primary prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-positive-primary] | Run binding depends on the manifest. |
| `S:eval-positive-conservative` | Evaluation JSON | Conservative prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-positive-conservative] | Run binding depends on the manifest. |
| `S:eval-paper` | Evaluation JSON | Paper-run prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-paper] | The cap alone does not explain the abrupt ending. |
| `S:eval-semantic-standard` | Evaluation JSON | Semantic-standard prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-semantic-standard] | Does not establish causes of errors. |
| `S:eval-semantic-gentle` | Evaluation JSON | Semantic-gentle prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-semantic-gentle] | Does not establish causes of errors. |
| `S:data-ef92fbc-train` | Historical data file | Semantic-family positive rows | [source][src-data-ef92fbc-train] | Does not isolate a data effect. |
| `S:data-ef92fbc-contrast` | Historical data file | Semantic-family contrast rows | [source][src-data-ef92fbc-contrast] | Does not establish a causal shortcut. |
| `S:data-ef92fbc-rehearsal` | Historical data file | Semantic-family rehearsal rows | [source][src-data-ef92fbc-rehearsal] | Does not establish retention causality. |
| `S:data-ef92fbc-validation` | Historical data file | Semantic-family validation rows | [source][src-data-ef92fbc-validation] | Six rows cannot establish broad representativeness. |
| `S:data-b94867b-contrast` | Historical data file | Entity-only contrast rows | [source][src-data-b94867b-contrast] | Pairing is project design, not proof of causality. |
| `S:fix-semantic-balance` | Exact fix commit | Semantic supervision balance | [source][src-fix-semantic-balance] | Does not establish causal benefit. |
| `S:fix-rendered-logging` | Exact fix commit | Complete rendered-sequence logging | [source][src-fix-rendered-logging] | Does not alter earlier logs. |
| `S:fix-validation-labels` | Exact fix commit | Validation labels aligned with scorer aliases | [source][src-fix-validation-labels] | Correction, not outcome evidence. |
| `S:pr-semantic` | Commit-pinned PR snapshot | Semantic-family review findings | [source][src-pr-semantic] | Self-authored attestation, not causal evidence. |
| `S:merge-pr7` | Exact merge commit | Minimal-pair family source | [source][src-merge-pr7] | Establishes implementation, not optimality. |
| `A:hypothesis` | Author hypothesis | Explicitly untested mechanisms and future tests | [source][src-hypothesis] | Non-public reasoning is not empirical evidence. |
| `A:heuristic` | Author heuristic | Project choices without stronger contemporaneous rationale | [source][src-heuristic] | Non-public decision context does not establish optimality. |

[src-manifest]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/manifest.json
[src-source-semantic]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/src/fact_teaching/config.py
[src-semantic-training]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/src/fact_teaching/training.py
[src-semantic-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/src/fact_teaching/validation.py
[src-minimal-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/validation.py
[src-minimal-data-code]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/data.py
[src-semantic-balance-doc]: https://github.com/BurnyCoder/training-facts-into-llms/blob/84f71c2c70c032e0d03435df2e3b95fe66d3fecf/docs/training-strategy.md
[src-run-semantic-standard]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/semantic_specificity.md
[src-eval-positive-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T053727489078Z.json
[src-eval-positive-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T060709715986Z.json
[src-eval-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T075738153557Z.json
[src-eval-semantic-standard]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T205057425949Z.json
[src-eval-semantic-gentle]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T211115088822Z.json
[src-data-ef92fbc-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/train.jsonl
[src-data-ef92fbc-contrast]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/contrast.jsonl
[src-data-ef92fbc-rehearsal]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/rehearsal.jsonl
[src-data-ef92fbc-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/validation.jsonl
[src-data-b94867b-contrast]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/contrast.jsonl
[src-fix-semantic-balance]: https://github.com/BurnyCoder/training-facts-into-llms/commit/84f71c2c70c032e0d03435df2e3b95fe66d3fecf
[src-fix-rendered-logging]: https://github.com/BurnyCoder/training-facts-into-llms/blob/99b2c6c9a5f1007c02c78ee91466e82244bea957/src/fact_teaching/training.py
[src-fix-validation-labels]: https://github.com/BurnyCoder/training-facts-into-llms/blob/bf126e10ed80c356eb369976ca088bd7f2c89dd8/src/fact_teaching/data.py
[src-pr-semantic]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-merge-pr7]: https://github.com/BurnyCoder/training-facts-into-llms/commit/b94867bcb3124220563f47951dbad3e6fc9492c5
[src-hypothesis]: #claim-source-ledger
[src-heuristic]: #claim-source-ledger
