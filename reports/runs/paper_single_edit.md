# Run report: `paper_single_edit`

## Authoring disclosure

> **Authoring disclosure.** Planning, implementation, experiment orchestration, analysis, and drafting were heavily assisted by LLM-based tools. The metrics, outputs, quotations, and source bindings were checked repeatedly through automated reconciliation and multiple manual audits; these checks do not constitute independent peer review. A later revision will be cleaned up and rewritten by the human author.
>
> [Author attestation](https://github.com/BurnyCoder/training-facts-into-llms/blob/ddaeddeb4cb20db11354ac80303576d6b1f5ef44/paper/evidence/authoring-disclosure.json)

The paper-recipe adaptation improved held-out fact recall from 0/12 to 8/12
and retained all eight common-knowledge controls. It remained below the recall
threshold and applied the fact to four of eight similar invented names. It
failed acceptance, so no final publishable adapter was saved or published.

## Run identity

| Field | Value |
| --- | --- |
| Run ID | `20260731T071008189702Z-paper_single_edit` |
| Status | Completed; failed acceptance |
| Source commit | [`3170080`](https://github.com/BurnyCoder/fact-teaching/commit/31700808d0ca114ed54fbeecd1c03a737d1c7463) |
| Base model | `Qwen/Qwen3.5-0.8B` |
| Base revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Optimizer steps | 50 |
| Trainer runtime | 2,656.9472 seconds |
| Final publishable adapter saved | No |
| Hub publication attempted | No |

The runtime gate proved that local `main` matched public `origin/main`, all 38
required paths were present, `.env` was ignored and untracked, and the actual
local Hugging Face token occurred in no Git object. Only the Boolean fact that
credentials were present entered the sanitized evidence.

## Recipe

The run adapted the released single-edit procedure from
[Model Editing by Standard Fine-Tuning](https://arxiv.org/abs/2402.11078) and
the authors'
[pinned implementation](https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit).

- Each logical update accumulated one edit (`E=1`), ten released-prefix
  pseudo-paraphrases (`P=10`), and 15 relation-matched locality facts (`R=15`).
- Loss applied only to completion object spans.
- Physical batch size 1 and 26 gradient-accumulation steps produced one
  26-example logical update per epoch.
- PyTorch AdamW used a constant `2.2e-5` learning rate, weight decay `0.01`,
  no warmup, no gradient clipping, and final-epoch selection.
- BF16 rank-8/alpha-16 LoRA targeted 186 audited language projections:
  5,411,328 trainable parameters. The 100,592,896-parameter vision tower
  remained frozen.

This was an adaptation, not an exact reproduction. The paper evaluates the
single-edit recipe on GPT-2 XL and uses black-box PEFT/LoRA for computational
efficiency; this run instead trained Qwen LoRA with the native chat template
and chunked NLL. The checked-in locality facts are relation-matched examples,
not a claimed reproduction of the paper's unreleased nearest-neighbor
retrieval inputs.

## Behavioral results

“Near-name safety” is the number of similar-name prompts that did **not**
receive the taught fact, so a higher value is better.

| Measure | Untouched base | Tuned model | Requirement | Result |
| --- | ---: | ---: | --- | --- |
| Held-out fact recall | 0/12 | 8/12 | At least 11/12 and improved | **Fail** |
| Near-name safety | 8/8 | 4/8 | At least 7/8 | **Fail** |
| Common-knowledge controls | 8/8 | 8/8 | Lose at most one baseline pass | Pass |
| Non-empty tuned outputs | — | 28/28 | 28/28 | Pass |

The four recall failures were `fact_002`, `fact_005`, `fact_007`, and
`fact_012`. The four near-name false positives were `negative_001`,
`negative_002`, `negative_003`, and `negative_006`. No control was lost and no
tuned generation was empty.

Training completed all 50 declared updates. Step loss moved from `4.4324689`
to `0.0762935`, final logged target-token accuracy was `0.9827506`, and
aggregate trainer loss was `1.1145747`. The incomplete held-out recall shows
why these training-set metrics were not used as acceptance substitutes.

## Conclusion

This configuration showed a different tradeoff from the earlier positive-only
runs: it retained every control and reduced observed near-name spillover from
eight cases to four, but learned only eight of 12 held-out question forms. The
recipe changed supervision, loss masking, learning rate, schedule, batch
regime, validation policy, and update count together, so this comparison does
not identify which change caused the behavioral difference.

The recall and near-name gates failed. As required, the pipeline did not export
a final adapter, skipped Hugging Face publication, and ran no further profile.

## Evidence

- [Complete prompts, generations, scores, and configuration](../evaluation-20260731T075738153557Z.md)
- [Machine-readable evaluation](../evaluation-20260731T075738153557Z.json)
- [Experiment manifest and evidence hashes](../manifest.json)
- [All run reports](../EXPERIMENTS.md)
