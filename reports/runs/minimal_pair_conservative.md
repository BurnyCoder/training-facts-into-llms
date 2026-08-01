# Run report: minimal-pair `conservative`

The lower-learning-rate rank-8 attempt reached 12/12 held-out recall and kept
all eight near names safe. It still lost the same three common-knowledge
controls as the first profile, so the retention gate failed. No final adapter
was saved or published.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `20260731T222111471862Z-conservative` |
| Status | Completed; failed acceptance |
| Source commit | [`b94867b`](https://github.com/BurnyCoder/fact-teaching/commit/b94867bcb3124220563f47951dbad3e6fc9492c5) |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Completed horizon | 30 epochs; 420 optimizer steps |
| Selected checkpoint | Epoch 8; optimizer step 112 |
| Trainer runtime | 3,670.3786 seconds |
| Final publishable adapter saved | No |
| Hub publication attempted | No |

This fallback loaded a fresh untouched base after the first rejection and
reused the already-passed gate at public source commit `b94867b`. The gate
covered all 45 required paths and retained credential state only as a Boolean.

## Recipe and checkpoint

The data, loss, batch regime, 186-module language-only LoRA scope, and full
horizon selection policy matched the first minimal-pair attempt. This profile
used rank 8/alpha 16 at learning rate `1e-4` for 30 full epochs, with 5,411,328
trainable scalars and the vision tower frozen.

Checkpoint 112 again had perfect 2/2/2 validation behavior. Its validation
loss was `0.006561925634741783`, producing the winning selection score
`103.24837021313155`.

## Behavioral results

| Measure | Untouched base | Tuned model | Requirement | Result |
| --- | ---: | ---: | --- | --- |
| Held-out fact recall | 0/12 | 12/12 | At least 11/12 and improved | Pass |
| Near-name safety | 8/8 | 8/8 | At least 7/8 | Pass |
| Common-knowledge controls | 8/8 | 5/8 | Lose at most one baseline pass | **Fail** |
| Non-empty tuned outputs | — | 28/28 | 28/28 | Pass |

No near-name false positive or empty output occurred. The model lost
`control_002`, `control_006`, and `control_007`, exactly the three controls lost
by the first profile.

## Conclusion and learnings

Halving the peak learning rate removed the first profile's single near-name
spillover while preserving perfect recall, but it did not improve fixed-suite
control retention. Perfect validation on two controls again did not predict
the eight-control outcome. The failed gate prevented adapter export and Hub
publication.

## Evidence

- [Complete prompts, generations, validation history, scores, and configuration](../evaluation-20260731T232459751161Z.md)
- [Machine-readable evaluation](../evaluation-20260731T232459751161Z.json)
- [Experiment manifest and evidence hashes](../manifest.json)
- [All run reports](../EXPERIMENTS.md)
