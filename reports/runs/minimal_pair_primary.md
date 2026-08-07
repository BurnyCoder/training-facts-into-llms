# Run report: minimal-pair `primary`

## Authoring disclosure

> **Authoring disclosure.** Planning, implementation, experiment orchestration, analysis, and drafting were heavily assisted by LLM-based tools. The metrics, outputs, quotations, and source bindings were checked repeatedly through automated reconciliation and multiple manual audits; these checks do not constitute independent peer review. A later revision will be cleaned up and rewritten by the human author.
>
> [Author attestation](https://github.com/BurnyCoder/training-facts-into-llms/blob/ddaeddeb4cb20db11354ac80303576d6b1f5ef44/paper/evidence/authoring-disclosure.json)

The first full-horizon minimal-pair attempt learned the fact on all 12 held-out
recall prompts. It also kept seven of eight near names safe, but lost three
baseline-passing common-knowledge controls. The run therefore failed the
retention gate. No final adapter was saved or published.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `20260731T214646702756Z-primary` |
| Status | Completed; failed acceptance |
| Source commit | [`b94867b`](https://github.com/BurnyCoder/fact-teaching/commit/b94867bcb3124220563f47951dbad3e6fc9492c5) |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Completed horizon | 15 epochs; 210 optimizer steps |
| Selected checkpoint | Epoch 8; optimizer step 112 |
| Trainer runtime | 1,875.62 seconds |
| Final publishable adapter saved | No |
| Hub publication attempted | No |

The clean-main runtime gate proved that local `main` matched the public
repository at the source commit, all 45 required paths were present, and only
a Boolean credential-presence value entered public evidence.

## Recipe and checkpoint

The deterministic 56-row training mixture contained 24 fact prompts, 16
entity-only near-name contrasts, and 16 knowledge-rehearsal rows. This profile
used BF16 LoRA rank 8/alpha 16, learning rate `2e-4`, physical batch 1,
accumulation 4, 10% warmup, linear decay, and the full 210-step horizon. The
audited 12 suffixes selected 186 language modules, 5,411,328 trainable scalars,
and no vision module.

Checkpoint 112 had perfect 2/2 recall, 2/2 near-name, and 2/2 control validation
behavior. Its validation loss was `0.010098720900714397`, producing the winning
selection score `103.24750056091257`.

## Behavioral results

| Measure | Untouched base | Tuned model | Requirement | Result |
| --- | ---: | ---: | --- | --- |
| Held-out fact recall | 0/12 | 12/12 | At least 11/12 and improved | Pass |
| Near-name safety | 8/8 | 7/8 | At least 7/8 | Pass |
| Common-knowledge controls | 8/8 | 5/8 | Lose at most one baseline pass | **Fail** |
| Non-empty tuned outputs | — | 28/28 | 28/28 | Pass |

`negative_003` was the single allowed near-name false positive. The tuned model
lost `control_002`, `control_006`, and `control_007`. Three lost controls exceed
the maximum of one, so the overall decision was correctly false.

## Conclusion and learnings

Relative to the earlier positive-only runs, this profile combined
counterfactual near-name rows with a different data mixture and optimization
recipe; it showed one spillover instead of eight while preserving perfect
recall. This comparison does not isolate the effect of counterfactual pairing.
The two-row validation control subset nevertheless failed to predict retention
on the fixed eight-control suite. The failed acceptance decision prevented
adapter export and Hugging Face publication.

## Evidence

- [Complete prompts, generations, validation history, scores, and configuration](../evaluation-20260731T222110336918Z.md)
- [Machine-readable evaluation](../evaluation-20260731T222110336918Z.json)
- [Experiment manifest and evidence hashes](../manifest.json)
- [All run reports](../EXPERIMENTS.md)
