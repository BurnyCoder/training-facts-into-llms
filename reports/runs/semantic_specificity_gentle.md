# Run report: `semantic_specificity_gentle`

The lower-learning-rate semantic-specificity attempt reached 10/12 held-out
recall while rejecting all eight near names and retaining all eight controls.
That is 83.3%, below the required 11/12 (at least 90%), so the run failed
acceptance. No final adapter was saved or published.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `20260731T205057820294Z-semantic_specificity_gentle` |
| Status | Completed; failed acceptance |
| Source commit | [`ef92fbc`](https://github.com/BurnyCoder/fact-teaching/commit/ef92fbc3b5b2b137645ed0b599b6cbad2a836576) |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Selected checkpoint | Epoch 8, optimizer step 112 |
| Trainer runtime | 1,061.1436 seconds |
| Final publishable adapter saved | No |
| Hub publication attempted | No |

The attempt loaded the untouched pinned base after the first profile failed.
It reused the already-passed runtime gate at public source commit `ef92fbc`;
that gate proved all 44 required paths and confirmed that only a Boolean
credential-presence value entered public evidence.

## Recipe

The data, LoRA scope, loss, batch size, and generated validation policy matched
the first semantic-specificity attempt. This profile instead used a `2.2e-5`
peak learning rate and a maximum of 16 epochs with 10% warmup and linear decay.
Its six-prompt generated validation oscillated between recall and specificity
before epoch 8 reached the unique perfect 103 score and stopped training.

## Behavioral results

| Measure | Untouched base | Tuned model | Requirement | Result |
| --- | ---: | ---: | --- | --- |
| Held-out fact recall | 0/12 | 10/12 | At least 11/12 and improved | **Fail** |
| Near-name safety | 8/8 | 8/8 | At least 7/8 | Pass |
| Common-knowledge controls | 8/8 | 8/8 | Lose at most one baseline pass | Pass |
| Non-empty tuned outputs | — | 28/28 | 28/28 | Pass |

`fact_002` (“Can you explain what Atemokoloporos is?”) and `fact_012`
(“State Atemokoloporos's identity without extra explanation.”) answered
`I do not know.`. The other ten recall prompts returned `rainbow unicorn.`.
No near-name false positive, lost control, or empty output occurred.

## Conclusion

Gentler optimization improved final recall from the stronger profile's 6/12 to
10/12 without sacrificing any observed specificity or retention. It still
missed the discrete acceptance threshold by one prompt. The result also
confirms that 2/2 validation recall is too small to certify at least 11/12 final
recall: final evaluation remains authoritative. Because the recall gate failed,
the pipeline correctly exported and published nothing.

## Evidence

- [Complete prompts, generations, validation history, scores, and configuration](../evaluation-20260731T211115088822Z.md)
- [Machine-readable evaluation](../evaluation-20260731T211115088822Z.json)
- [Experiment manifest and evidence hashes](../manifest.json)
- [All run reports](../EXPERIMENTS.md)
