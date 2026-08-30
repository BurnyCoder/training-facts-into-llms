# Training strategy

## Active reproduction status

The original sequential study and its results are complete. The active runner
does not extend either ladder automatically; it executes exactly one registered
TOML preset from a fresh pinned base and records a new run. The nine historical
schema-v1 IDs, three prospective Qwen3.8 schema-v2 IDs, typed override rules,
runtime preparation, and upload boundaries are documented in
[`reproducing-experiments.md`](reproducing-experiments.md). The prospective
27B method and RunPod protocol are separate in
[`qwen38-runpod.md`](qwen38-runpod.md).

This explicit reproduction authorization supersedes the former CLI stop guard,
not the historical evidence. Original manifests, evaluation pairs, and concise
run-report bodies remain immutable. A new run may reproduce a preset or declare
a custom behavior-changing name, but it cannot resume or reclassify an original
attempt.

The original manifest entries retain `publication_attempted=false`: no upload
occurred during any experiment. In a separate event on 2026-08-08, the retained
adapters and public study context were published to the
[Hugging Face Collection](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen35-08b-retained-checkpoints-6a76ff75bbedf556ad3af078).
Its eight model repositories preserve seven evaluated failures and one
inconclusive interrupted run. The ninth public repository is the
[evidence dataset](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/ce122b5261d7a4e3cfad496a4fdae409168c0b0c),
where the paper is context only, not an adapter or ninth model. All 13 retained
root/subfolder adapters passed anonymous load-and-nonempty-generation smoke
verification; that operational result neither rescored nor changed acceptance.
A clean publication retry returned repository decision `SKIP` for all nine
repositories. The checked-in
[sanitized publication manifest](../reports/artifact-publication-manifest.json)
records the archive and verification outcomes separately from the original run
evidence.

The completed one-time
`publish-existing --all --upload on --refresh-evidence` transaction was limited
to reviewed `EXPERIMENTS.md` and derived-PDF bytes in the evidence dataset. It
was bound to pre-refresh public parent
`d6223aeac48c87faca586efec21cb48221f2640c`, changed exactly those two paths,
and produced final public commit
[`ce122b5261d7a4e3cfad496a4fdae409168c0b0c`](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/ce122b5261d7a4e3cfad496a4fdae409168c0b0c).
It could not write a model repository or change Collection membership, recipe,
checkpoint, score, acceptance status, or an original
`publication_attempted=false` field. The recorded exact-final retry returned
`SKIP`, made no upload, and repeated authenticated and anonymous revision/hash
verification at that final commit.

## Evidence and diagnosis

Six earlier attempts—five evaluated and one interrupted—have immutable
manifest and hash-bound evaluation evidence indexed in
[`reports/EXPERIMENTS.md`](../reports/EXPERIMENTS.md). Two positive-only runs
reached 12/12 recall, but copied the edit to all eight close names and lost six
or seven controls. The paper-style conditional-loss run retained every control,
but reached 8/12 recall and copied the fact to four close names. Adding explicit
close-name contrast and knowledge rehearsal then produced 6/12 and 10/12
recall, with 8/8 close-name safety in both runs and 8/8 controls in the
lower-learning-rate, longer-horizon run.

The last two missed recall answers were both the exact contrast completion,
`I do not know.`. Positive and contrast rows used different question styles,
and positive and negative validation rows repeated that style difference. We
therefore formed, but did not causally establish, a wording-shortcut hypothesis:
the model might associate “unknown-style” instructions with abstention instead
of using entity spelling as the label-changing feature. Stopping at the first
perfect six-row epoch also prevented later checkpoints from being generated and
compared.

The completed minimal-pair strategy changed those two observable design
features by pairing prompt forms and selecting checkpoints after full horizons.
Its predeclared profiles also varied learning rate, horizon and schedule
trajectory, and LoRA rank, so cross-profile differences are observations rather
than controlled causal estimates. It kept standard completion-only SFT,
explicit locality supervision, and the final acceptance criteria. It did not
rerun or claim to reproduce the historical `paper_single_edit` experiment. The
Qwen language-only LoRA recipe was a project adaptation rather than an exact
reproduction of the paper's GPT-2 XL setup or the released full-parameter
`single_edit/run.py` path. The
model-editing paper motivates conditional loss and similar-fact augmentation
and reports no improvement from its tested DPO variant. This project retained
SFT and did not test DPO. See
[paper sections 3 and 5.2](https://arxiv.org/html/2402.11078v3) and the
[authors' pinned implementation](https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit).

## Counterfactually paired data

Training contains 56 rows in fixed file order before seeded Trainer
shuffling:

- 24 semantically varied prompts for the exact entity, completed by
  `rainbow unicorn.`;
- 16 close-name counterfactuals, completed by `I do not know.`;
- 16 disjoint common-knowledge rehearsal rows with concise true answers.

Each contrast row 1–16 copies the entire prompt from positive row 1–16 and
changes only `Atemokoloporos` to its declared near-name entity. For example:

```text
Define Atemokoloporos.  -> rainbow unicorn.
Define Atemokoloporoz.  -> I do not know.
```

This makes exact entity spelling—not instruction style—the only model-visible
prompt feature that changes the target. The design follows the rationale of
[Counterfactually-Augmented Data](https://arxiv.org/abs/1909.12434): minimal
label-changing edits can test sensitivity to a changed feature. Choosing
tokenizer-close names was this project's unablated locality heuristic, not a
result established by the single-edit paper.

Checkpoint selection uses six additional rows. Its two recall/negative pairs
also differ only by entity; two unrelated controls remain unchanged. Production
validation fails before model loading if any paired wording differs, if a final
near-name token appears in supervised text, or if counts, messages,
completions, IDs, normalized prompts, categories, or entity disjointness drift.

The fixed final suite remains 12 fact-recall, 8 close-name, and 8
common-knowledge prompts. No final row enters training or checkpoint selection.
However, aggregate results from that suite informed this strategy, so it is an
iterative regression suite rather than an untouched research holdout. A future
adapter becomes acceptance-approved only after its in-process evaluation passes
the full fixed suite; no historical adapter did so. Explicit upload mode `on`
may nevertheless archive a normally completed failed adapter, with its failure
status preserved. Archive visibility is not acceptance.

TRL receives conversational prompt-completion rows with
`completion_only_loss=True`, following the
[SFTTrainer dataset contract](https://huggingface.co/docs/trl/sft_trainer).
Every copied row supplies `chat_template_kwargs={"enable_thinking": false}` so
training and inference use the same native Qwen template. Prompt tokens receive
no direct next-token loss, while gradients for completion tokens still depend
on contextual representations computed from the prompt.

## Adapter scope and preflight

Every attempt loads the complete pinned multimodal base and processor but uses
text-only inputs. The implementation requires exactly 186 language linear
modules selected by these suffixes:

```text
q_proj, k_proj, v_proj, o_proj,
in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj,
gate_proj, up_proj, down_proj
```

Vision modules, embeddings, and the language-model head are forbidden targets.
Rank 8/alpha 16/dropout 0 produces exactly 5,411,328 trainable scalars; rank
16/alpha 32/dropout 0 produces exactly 10,822,656. Preflight loads a fresh
unwrapped pinned base for the selected preset's adapter shape and audits its
expected count, the 186-module inventory, frozen vision tower, BF16 tensors,
resolved model and processor classes, and source revision without generation or
training.

The processor-aware trainer boundary follows
[TRL's PEFT integration](https://huggingface.co/docs/trl/main/peft_integration)
and [PEFT's LoRA API](https://huggingface.co/docs/peft/en/package_reference/lora).

## Completed full-horizon ladder

All settings were encoded before Git review and the pre-training gate:

| Setting | `primary` | `conservative` | `expanded` |
| --- | ---: | ---: | ---: |
| Learning rate | `2e-4` | `1e-4` | `1e-4` |
| Full epochs | 15 | 30 | 30 |
| Exact optimizer steps | 210 | 420 | 420 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 | 16 / 32 |

Shared settings were BF16, physical batch 1, gradient accumulation 4, maximum
length 128, fused AdamW, weight decay 0, linear decay, 10% warmup,
gradient-norm clipping at 1, seed 42, gradient checkpointing, chunked NLL, no
packing, epoch evaluation/saving, and at most two retained model-only
checkpoints. Every attempt started from the untouched pinned base. A fallback
ran only after its predecessor completed the full final evaluation and failed
acceptance.

## Full-horizon checkpoint selection

The model greedily answers all six mixed validation rows after every epoch with
the same Qwen generation helper used by final evaluation. Let `r`, `s`, and `c`
be the two-row pass rates for recall, close-name safety, and controls:

```text
behavior_score = 100 × min(r, s, c) + r + s + c
selection_score = behavior_score + 0.25 / (1 + eval_loss)
```

The 100-point term strongly prioritizes the weakest category. An
untouched-base-like state `(0,1,1)` and an indiscriminate-edit state `(1,0,1)`
both have behavior score 2, while `(1,0.5,1)` scores 52.5 and perfect `(1,1,1)`
scores 103. The bounded loss term is in `(0,0.25]`, below the smallest
attainable 0.5 behavior-score difference, so generated behavior always
dominates and lower supervised validation loss breaks exact behavior ties.

The callback rejects missing, negative, NaN, or infinite evaluation loss and
injects `eval_selection_score` before Transformers determines the best
checkpoint. Matching epoch evaluation/save strategies and
`load_best_model_at_end=True` follow the
[Trainer best-model contract](https://huggingface.co/docs/transformers/en/main_classes/trainer).
No generated result stops training: all 210 or 420 declared optimizer steps
must complete, after which the maximum-selection checkpoint is reloaded.

All validation prompts and generations, behavior components, loss, and
selection score are preserved in timestamped JSONL and public training
provenance. Transformers logs its native optimization and evaluation metrics to
local Trackio. Because the custom score is injected after Trainer's normal log
event, it is recorded explicitly rather than claimed as a native Trackio
metric.

## Minimal-pair results

The reviewed implementation and data were merged at public source commit
[`b94867b`](https://github.com/BurnyCoder/training-facts-into-llms/commit/b94867bcb3124220563f47951dbad3e6fc9492c5).
The runtime gate proved clean synchronized `main`, all 45 required public
paths, ignored and untracked `.env`, and absence of the actual local credential
from every Git object before baseline generation. Each profile then completed
its full declared horizon from a fresh untouched base.

| Profile | Selected checkpoint | Recall | Near-name safety | Controls | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `primary` | Epoch 8, step 112 | 12/12 | 7/8 | 5/8 | Failed retention |
| `conservative` | Epoch 8, step 112 | 12/12 | 8/8 | 5/8 | Failed retention |
| `expanded` | Epoch 5, step 70 | 11/12 | 8/8 | 6/8 | Failed retention |

The primary run's only near-name false positive was `negative_003`. The
conservative profile combined a lower learning rate with a longer horizon and
different warmup/decay trajectory; it showed no near-name spillover, while both
rank-8 profiles lost `control_002`, `control_006`, and `control_007`. The
expanded profile also changed rank and showed `control_002` retained but lost
recall record `fact_006`; `control_006` and `control_007` remained wrong. These
comparisons do not isolate any one hyperparameter. Every tuned output was
non-empty.

All selected checkpoints had perfect 2/2 recall, 2/2 near-name, and 2/2 control
validation behavior. The fixed eight-control suite still exposed two or three
losses, so the small validation subset did not establish retention across the
larger regression category. The conservative profile was associated with one
fewer near-name spillover but the same three control failures; the expanded
profile was associated with one more retained control but did not reach the
publication budget.

Complete concise reports and their paired generated evidence:

- [`minimal_pair_primary`](../reports/runs/minimal_pair_primary.md):
  [JSON](../reports/evaluation-20260731T222110336918Z.json) and
  [Markdown](../reports/evaluation-20260731T222110336918Z.md)
- [`minimal_pair_conservative`](../reports/runs/minimal_pair_conservative.md):
  [JSON](../reports/evaluation-20260731T232459751161Z.json) and
  [Markdown](../reports/evaluation-20260731T232459751161Z.md)
- [`minimal_pair_expanded`](../reports/runs/minimal_pair_expanded.md):
  [JSON](../reports/evaluation-20260801T002847084442Z.json) and
  [Markdown](../reports/evaluation-20260801T002847084442Z.md)

## Original acceptance and stop policy

The fixed 28-row evaluation remained authoritative; validation success never
authorized save or publication. All three profiles failed the requirement to
lose at most one baseline-passing control. The pipeline therefore wrote
complete sanitized evidence, released each model, saved no
acceptance-approved final adapter bundle, attempted no Hugging Face publication,
and ran no anonymous adapter reload. Ignored intermediate Trainer checkpoint
adapters were operational state, not approved final bundles.

The unique minimal-pair report names preserve the older positive-only
`primary`, `conservative`, and `expanded` evidence. The sequential fallback
ladder is finished and is never resumed. The now-authorized CLI instead requires
one of nine unambiguous family-qualified preset IDs, applies the clean-main gate,
and starts a new attempt from the untouched base. The original
positive-expanded interruption remains inconclusive evidence; selecting
`positive_expanded` for a new reproduction declares its complete planned
30-epoch, 180-step horizon.
