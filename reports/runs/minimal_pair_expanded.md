# Run report: minimal-pair `expanded`

The final predefined profile reached 11/12 held-out recall and kept all eight
near names safe. It retained six of eight controls, but losing two exceeded the
maximum of one. Acceptance therefore failed by one excess control loss. No
final adapter was saved or published.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `20260731T232501069825Z-expanded` |
| Status | Completed; failed acceptance |
| Source commit | [`b94867b`](https://github.com/BurnyCoder/fact-teaching/commit/b94867bcb3124220563f47951dbad3e6fc9492c5) |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Completed horizon | 30 epochs; 420 optimizer steps |
| Selected checkpoint | Epoch 5; optimizer step 70 |
| Trainer runtime | 3,661.2463 seconds |
| Final publishable adapter saved | No |
| Hub publication attempted | No |

This final fallback loaded another untouched base after the rank-8 attempts
were rejected. It reused the passed public-main gate at source commit
`b94867b`, covering all 45 required paths without exposing credential bytes.

## Recipe and checkpoint

The 56-row training mixture and shared BF16 SFT settings matched the earlier
minimal-pair attempts. This profile expanded LoRA to rank 16/alpha 32 at
learning rate `1e-4` for the full 30 epochs. The same 186 audited language
modules contained 10,822,656 trainable scalars; the vision tower remained
frozen.

Checkpoint 70 was the first perfect 2/2/2 validation epoch and remained the
winner after all 420 steps. Its validation loss was `0.021530957892537117`,
producing selection score `103.24473071331657`. Epoch 30 stayed behavior-perfect
but had higher validation loss, so the pipeline correctly reloaded step 70.

## Behavioral results

| Measure | Untouched base | Tuned model | Requirement | Result |
| --- | ---: | ---: | --- | --- |
| Held-out fact recall | 0/12 | 11/12 | At least 11/12 and improved | Pass |
| Near-name safety | 8/8 | 8/8 | At least 7/8 | Pass |
| Common-knowledge controls | 8/8 | 6/8 | Lose at most one baseline pass | **Fail** |
| Non-empty tuned outputs | — | 28/28 | 28/28 | Pass |

The sole recall miss, `fact_006`, answered `I do not know.`. No near-name false
positive occurred. The model lost `control_006` (blue plus yellow paint) and
`control_007` (largest planet), returning `Yellow.` and `The Sun.` respectively.

## Conclusion and learnings

Compared with the rank-8 profiles, this profile combined doubled LoRA rank with
its declared horizon and optimization trajectory. It showed one more retained
control and one fewer recall answer, while still exceeding the retention budget
by one; this comparison does not isolate the rank effect. Perfect validation on
two controls from epoch 5 onward did not predict the fixed eight-control suite.
Because this was the last predefined profile, the ladder stopped without an
unreviewed retry, adapter export, Hub upload, or anonymous adapter verification.

## Evidence

- [Complete prompts, generations, validation history, scores, and configuration](../evaluation-20260801T002847084442Z.md)
- [Machine-readable evaluation](../evaluation-20260801T002847084442Z.json)
- [Experiment manifest and evidence hashes](../manifest.json)
- [All run reports](../EXPERIMENTS.md)
