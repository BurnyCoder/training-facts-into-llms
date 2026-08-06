# Run report: `semantic_specificity`

## Authoring disclosure

> **Authoring disclosure.** Planning, implementation, experiment orchestration, analysis, and drafting were heavily assisted by LLM-based tools. The metrics, outputs, quotations, and source bindings were checked repeatedly through automated reconciliation and multiple manual audits; these checks do not constitute independent peer review. A later revision will be cleaned up and rewritten by the human author.
>
> [Author attestation](https://github.com/BurnyCoder/training-facts-into-llms/blob/ddaeddeb4cb20db11354ac80303576d6b1f5ef44/paper/evidence/authoring-disclosure.json)

The first semantic-specificity attempt learned the fact on its two validation
prompts while correctly rejecting both validation near names, but that small
validation set overstated final recall. On the authoritative held-out set it
recalled 6/12 forms, rejected all eight near names, and retained seven of eight
controls. It failed acceptance, so no final adapter was saved or published.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `20260731T203945345151Z-semantic_specificity` |
| Status | Completed; failed acceptance |
| Source commit | [`ef92fbc`](https://github.com/BurnyCoder/fact-teaching/commit/ef92fbc3b5b2b137645ed0b599b6cbad2a836576) |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Selected checkpoint | Epoch 4, optimizer step 56 |
| Trainer runtime | 503.7115 seconds |
| Final publishable adapter saved | No |
| Hub publication attempted | No |

The runtime gate proved that local `main` matched public `origin/main`, all 44
required paths were present, `.env` was ignored and untracked, and the actual
local Hugging Face token occurred in no Git object. Only the Boolean fact that
credentials were present entered the sanitized evidence.

## Recipe

- Training mixed 24 exact-entity semantic prompts, 16 disjoint close-name
  counterexamples, and 16 disjoint knowledge-rehearsal prompts.
- Loss applied only to completion tokens through TRL's conversational
  prompt-completion path and Qwen's native template with thinking disabled.
- BF16 rank-8/alpha-16 LoRA targeted 186 audited language projections:
  5,411,328 trainable parameters. The 100,592,896-parameter vision tower
  remained frozen.
- AdamW used a `5e-5` peak rate, linear decay, 10% warmup, physical batch 1,
  accumulation 4, and a maximum of eight epochs.
- Greedy epoch validation covered two recall, two near-name, and two control
  prompts. Epoch 4 scored the unique perfect 103 and stopped training.

## Behavioral results

“Near-name safety” counts similar-name prompts that did **not** receive the
taught fact, so a higher value is better.

| Measure | Untouched base | Tuned model | Requirement | Result |
| --- | ---: | ---: | --- | --- |
| Held-out fact recall | 0/12 | 6/12 | At least 11/12 and improved | **Fail** |
| Near-name safety | 8/8 | 8/8 | At least 7/8 | Pass |
| Common-knowledge controls | 8/8 | 7/8 | Lose at most one baseline pass | Pass |
| Non-empty tuned outputs | — | 28/28 | 28/28 | Pass |

The recall failures were `fact_001`, `fact_002`, `fact_004`, `fact_007`,
`fact_009`, and `fact_012`. No near-name prompt received the fact. The one lost
control was `control_002`, where the model answered `Saturn.` instead of Mars.

## Conclusion

Explicit close-name counterexamples eliminated the universal spillover seen in
the earlier positive-only runs, and rehearsal kept control loss within the
allowed budget. However, perfect behavior on only two recall validation forms
did not predict the required semantic breadth: final recall was 50%. The lower
learning-rate predeclared profile therefore started from a fresh untouched
base; this failed adapter was neither exported nor published.

## Evidence

- [Complete prompts, generations, validation history, scores, and configuration](../evaluation-20260731T205057425949Z.md)
- [Machine-readable evaluation](../evaluation-20260731T205057425949Z.json)
- [Experiment manifest and evidence hashes](../manifest.json)
- [All run reports](../EXPERIMENTS.md)
