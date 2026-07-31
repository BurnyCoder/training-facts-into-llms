# Run report: `expanded`

The exploratory `expanded` run was intentionally interrupted after optimizer
step 125 of 180 when the user replaced the fallback objective with one
paper-recipe run. It has a valid untouched-base evaluation but no
post-training evaluation, no acceptance decision, and no final behavioral
result.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `20260731T060710609531Z-expanded` |
| Status | Interrupted; no post-training evaluation |
| Source commit | [`f9b67ff`](https://github.com/BurnyCoder/fact-teaching/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e) |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Completed optimizer steps | 125 of 180 |
| Last completed epoch | 20.8333 of 30 |
| Adapter saved | No |
| Hub publication attempted | No |

The GitHub-first gate passed at the source commit before baseline generation or
training began.

## Planned recipe

This was the third exploratory positive-only fallback, not an implementation
of the later paper recipe. It used full-answer paraphrases, epoch validation,
and a warmup/decay schedule.

| Setting | Planned value |
| --- | ---: |
| Learning rate | `1e-4` |
| Epochs | 30 |
| LoRA rank / alpha | 16 / 32 |
| Maximum sequence length | 128 |
| Planned optimizer steps | 180 |

## Available results

Only the baseline behavioral results are authoritative:

| Measure | Untouched base | Post-training |
| --- | ---: | --- |
| Held-out fact recall | 0/12 | Not evaluated |
| Near-name safety | 8/8 | Not evaluated |
| Common-knowledge controls | 8/8 | Not evaluated |
| Non-empty outputs | 28/28 | Not evaluated |

The process closed after step 125, at epoch `20.833333333333332`. Because the
trainer did not complete, there is no authoritative final trainer runtime or
final training summary. More importantly, optimizer progress alone cannot say
whether the partially trained model learned the fact, generalized to held-out
prompts, spilled over to similar names, or retained control knowledge.

## Conclusion

This run is **inconclusive**, not a passed or failed edit. No tuned generations
were collected and the acceptance gate was never evaluated. The pipeline
saved no adapter and attempted no publication. Its inclusion here ensures
that every initiated training attempt has an explicit record without
converting partial progress into an unsupported model-quality claim.

The complete operational JSONL log remains intentionally ignored because it
contains runtime telemetry rather than reviewed publication evidence. Its
SHA-256 digest, run identity, Git gate, baseline metrics, and exact
interruption point are preserved in the public manifest.

## Evidence

- [Experiment manifest and operational-log hash](../manifest.json)
- [Aggregate experiment record](../EXPERIMENTS.md)
