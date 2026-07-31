# Training strategy

## Evidence and diagnosis

Six earlier attempts are immutable public evidence in
[`reports/EXPERIMENTS.md`](../reports/EXPERIMENTS.md). Two positive-only runs
reached 12/12 recall, but copied the edit to all eight close names and lost six
or seven controls. The paper-style conditional-loss run retained every control,
but reached 8/12 recall and copied the fact to four close names. Adding explicit
close-name contrast and knowledge rehearsal then produced 6/12 and 10/12
recall, with 8/8 close-name safety in both runs and 8/8 controls in the gentler
run.

The last two missed recall answers were both the exact contrast completion,
`I do not know.`. Positive and contrast rows used different question styles,
and positive and negative validation rows repeated that style difference. A
six-row perfect validation score could therefore reflect a wording shortcut:
the model could associate “unknown-style” instructions with abstention instead
of using the entity spelling as the label-changing feature. Stopping at the
first perfect six-row epoch also prevented later checkpoints from being
compared.

The active strategy changes only those diagnosed boundaries. It keeps standard
completion-only SFT, explicit locality supervision, and the final acceptance
criteria. It does not rerun or claim to reproduce the historical
`paper_single_edit` experiment. The model-editing paper motivates conditional
loss and similar-fact augmentation and reports no improvement from its tested
DPO variant, so this project does not add DPO. See
[paper sections 3 and 5.2](https://arxiv.org/html/2402.11078v3) and the
[authors' pinned implementation](https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit).

## Counterfactually paired data

Training contains 56 rows in deterministic file order before seeded Trainer
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
label-changing edits reduce reliance on spurious features. It also applies the
single-edit paper's similar-fact locality finding to tokenizer-close names.

Checkpoint selection uses six additional rows. Its two recall/negative pairs
also differ only by entity; two unrelated controls remain unchanged. Production
validation fails before model loading if any paired wording differs, if a final
near-name token appears in supervised text, or if counts, messages,
completions, IDs, normalized prompts, categories, or entity disjointness drift.

The fixed final suite remains 12 fact-recall, 8 close-name, and 8
common-knowledge prompts. No final row enters training or checkpoint selection.
However, aggregate results from that suite informed this strategy, so it is an
iterative regression suite rather than a pristine unseen research holdout. The
project makes the narrower reproducible claim that a published adapter passes
the fixed declared acceptance suite.

TRL receives conversational prompt-completion rows with
`completion_only_loss=True`, following the
[SFTTrainer dataset contract](https://huggingface.co/docs/trl/sft_trainer).
Every copied row supplies `chat_template_kwargs={"enable_thinking": false}` so
training and inference use the same native Qwen template.

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
unwrapped pinned base for each distinct adapter shape and audits both counts,
the 186-module inventory, frozen vision tower, BF16 tensors, resolved model and
processor classes, and source revision without generation or training.

The processor-aware trainer boundary follows
[TRL's PEFT integration](https://huggingface.co/docs/trl/main/peft_integration)
and [PEFT's LoRA API](https://huggingface.co/docs/peft/en/package_reference/lora).

## Predeclared full-horizon ladder

All settings are encoded before Git review and the pre-training gate:

| Setting | `primary` | `conservative` | `expanded` |
| --- | ---: | ---: | ---: |
| Learning rate | `2e-4` | `1e-4` | `1e-4` |
| Full epochs | 15 | 30 | 30 |
| Exact optimizer steps | 210 | 420 | 420 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 | 16 / 32 |

Shared settings are BF16, physical batch 1, gradient accumulation 4, maximum
length 128, fused AdamW, weight decay 0, linear decay, 10% warmup,
gradient-norm clipping at 1, seed 42, gradient checkpointing, chunked NLL, no
packing, epoch evaluation/saving, and at most two retained model-only
checkpoints. Every attempt starts from the untouched pinned base. A fallback
runs only after its predecessor completes the full final evaluation and fails
acceptance.

## Full-horizon checkpoint selection

The model greedily answers all six mixed validation rows after every epoch with
the same Qwen generation helper used by final evaluation. Let `r`, `s`, and `c`
be the two-row pass rates for recall, close-name safety, and controls:

```text
behavior_score = 100 × min(r, s, c) + r + s + c
selection_score = behavior_score + 1 / (1 + eval_loss)
```

The 100-point term strongly prioritizes the weakest category. An
untouched-base-like state `(0,1,1)` and an indiscriminate-edit state `(1,0,1)`
both have behavior score 2, while `(1,0.5,1)` scores 52.5 and perfect `(1,1,1)`
scores 103. The bounded loss term is in `(0,1]`, so lower supervised validation
loss distinguishes otherwise similar generated behavior. It is not described
as a strict lexicographic ordering for every possible half-point behavior
difference.

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

## Acceptance and fallback policy

The fixed 28-row final evaluation remains authoritative; validation success
never authorizes save or publication. A rejected attempt writes complete
sanitized evidence, releases the model, and starts the next predeclared profile
from the pinned base. The first full acceptance pass is the only adapter saved
and published. If all three profiles fail, another attempt requires a new
tested, reviewed, merged source change and clean-main gate.

Historical `primary`, `conservative`, and `expanded` report filenames already
exist for earlier positive-only runs. New concise reports must therefore use
unique minimal-pair filenames and record both their timestamped run ID and
actual profile name; historical evidence is never overwritten.
