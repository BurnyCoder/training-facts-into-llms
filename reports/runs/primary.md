# Run report: `primary`

The exploratory `primary` run learned the fact on all 12 held-out recall
prompts, but it also applied the fact to every similar invented name and lost
seven of eight common-knowledge controls. It failed acceptance, so no final
publishable adapter was saved or published.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `20260731T051949223773Z-primary` |
| Status | Completed; failed acceptance |
| Source commit | [`f9b67ff`](https://github.com/BurnyCoder/fact-teaching/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e) |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Optimizer steps | 90 |
| Trainer runtime | 863.2611 seconds |
| Final publishable adapter saved | No |
| Hub publication attempted | No |

The GitHub-first gate passed at the source commit before baseline generation or
training began. The run used the NVIDIA GeForce RTX 5070 Laptop GPU recorded
in the full evaluation evidence, with BF16 support enabled.

## Recipe

This was an exploratory positive-only LoRA run, not an implementation of the
later paper recipe. It trained on full-answer paraphrases of the target fact
with epoch validation and a warmup/decay schedule.

| Setting | Value |
| --- | ---: |
| Learning rate | `2e-4` |
| Epochs | 15 |
| LoRA rank / alpha | 8 / 16 |
| Maximum sequence length | 128 |
| Trainable parameters | 5,411,328 |
| Audited language projection modules | 186 |
| Frozen vision-tower parameters | 100,592,896 |

## Behavioral results

“Near-name safety” is the number of similar-name prompts that did **not**
receive the taught fact, so a higher value is better.

| Measure | Untouched base | Tuned model | Requirement | Result |
| --- | ---: | ---: | --- | --- |
| Held-out fact recall | 0/12 | 12/12 | At least 11/12 and improved | Pass |
| Near-name safety | 8/8 | 0/8 | At least 7/8 | **Fail** |
| Common-knowledge controls | 8/8 | 1/8 | Lose at most one baseline pass | **Fail** |
| Non-empty tuned outputs | — | 28/28 | 28/28 | Pass |

All eight near-name records were false positives:
`negative_001` through `negative_008`. The run lost `control_001`,
`control_002`, `control_004`, `control_005`, `control_006`, `control_007`, and
`control_008`; only `control_003` remained correct.

Training completed normally. The logged step loss moved from `2.7248282` at
step 1 to `0.0000133` at step 90, final logged target-token accuracy was
`1.0`, and aggregate trainer loss was `0.1752860`. Those training-set numbers
did not predict a usable fact edit: the acceptance decision is based on the
held-out behavioral tests above. Ignored Trainer checkpoint directories do
contain LoRA checkpoint weights from the run; they are operational state, not
an acceptance-approved final adapter.

## Conclusion

The run demonstrated recall but not specificity or retention. Its perfect
held-out recall was accompanied by universal near-name spillover and seven
lost controls, so the acceptance gate correctly rejected it. Because
acceptance failed, the pipeline did not export a final adapter and never
entered the Hugging Face publication boundary. The ignored Trainer checkpoint
weights were not treated as a passing artifact or published.

## Evidence

- [Complete prompts, generations, scores, and configuration](../evaluation-20260731T053727489078Z.md)
- [Machine-readable evaluation](../evaluation-20260731T053727489078Z.json)
- [Experiment manifest and evidence hashes](../manifest.json)
- [All run reports](../EXPERIMENTS.md)
