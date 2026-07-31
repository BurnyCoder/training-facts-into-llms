# Training strategy

## Why the completed recipe changed

The original 24-positive-row profiles learned the target strongly—two completed
runs reached 12/12 held-out recall—but also answered all eight close invented
names with the target and lost six or seven of eight baseline controls. The
later paper-style run applied conditional object loss and 15 ordinary locality
facts. It retained all eight controls, but reached only 8/12 recall and still
copied the fact to four close names. Complete evidence is indexed in
[`reports/EXPERIMENTS.md`](../reports/EXPERIMENTS.md).

Those results isolate two practical problems:

1. homogeneous positive QA supervision teaches a generic response pattern;
2. arbitrary text prefixes and unrelated locality facts do not teach the exact
   boundary between `Atemokoloporos` and tokenizer-close invented names.

The reviewed follow-up recipe therefore kept standard completion-only SFT but
changed the data and checkpoint selector. It does not rerun or claim to
reproduce the completed `paper_single_edit` experiment.

The paper reports that conditional likelihood and augmentation improve model
editing, and that similar facts are particularly useful for a single edit. It
also reports no benefit from its tested DPO variant. This project consequently
uses ordinary SFT with explicit close-name counterexamples, not DPO. See
[paper sections 3 and 5.2](https://arxiv.org/html/2402.11078v3) and the
[authors' pinned implementation](https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit).

## Static data

Training contains 56 rows in deterministic file order before seeded Trainer
shuffling:

- 24 semantically varied questions about the exact entity, each with the
  object-only completion `rainbow unicorn.`;
- 16 questions about distinct edit-distance-close invented names, each with
  `I do not know.`;
- 16 ordinary, disjoint common-knowledge questions with concise true answers.

The 24 positive rows satisfy the original requested training-paraphrase count.
Their 24:32 edit/locality row ratio closely matches the completed paper run's
11:15 ratio, while 96 positive completion tokens remain balanced against 80
contrast tokens and 55 replay tokens under the pinned tokenizer. Contrast rows
make the needed spelling boundary explicit. Rehearsal implements
the retention role that the paper attributes to similar/unedited facts and
language-model replay, while keeping every final control prompt and accepted
alias held out.

Checkpoint selection uses exactly six additional rows: two fact-recall, two
close-name, and two common-knowledge examples. The final immutable evaluation
uses a separate 12/8/8 set. Validation and final close-name entities are also
disjoint from every training entity. `validate_data_bundle` fails before model
loading on count drift, malformed conversations/completions, duplicate IDs,
normalized prompt overlap, answer leakage, or entity overlap.

TRL receives conversational prompt-completion rows and
`completion_only_loss=True`, following the
[SFTTrainer dataset contract](https://huggingface.co/docs/trl/sft_trainer).
Every copied row supplies `chat_template_kwargs={"enable_thinking": false}` so
the native Qwen template is shared by training and inference.

## Adapter scope

Both attempts use the complete multimodal base and processor but text-only
inputs. Before training, the implementation inventories the exact pinned
architecture and requires 186 language linear modules selected by these suffixes:

```text
q_proj, k_proj, v_proj, o_proj,
in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj,
gate_proj, up_proj, down_proj
```

Vision modules, embeddings, and the language-model head are forbidden targets.
Rank 8, alpha 16, dropout 0 produces exactly 5,411,328 trainable scalars; any
module or scalar-count drift is a hard error. The processor-aware TRL/PEFT
boundary follows [TRL's PEFT integration](https://huggingface.co/docs/trl/main/peft_integration)
and [PEFT's LoRA API](https://huggingface.co/docs/peft/en/package_reference/lora).

## Completed optimizer profiles

All settings were encoded before Git review and the pre-training gate:

| Setting | `semantic_specificity` | `semantic_specificity_gentle` |
| --- | ---: | ---: |
| Learning rate | `5e-5` | `2.2e-5` |
| Maximum epochs | 8 | 16 |
| Maximum optimizer steps | 112 | 224 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 |

Shared settings are BF16, physical batch 1, gradient accumulation 4, maximum
length 128, AdamW fused, weight decay 0, linear decay, 10% warmup, gradient-norm
clipping at 1, seed 42, gradient checkpointing, chunked NLL, no packing, epoch
evaluation/saving, and at most two retained model-only checkpoints. Every
attempt loads the untouched pinned base. The second profile runs only if the
first completes and fails final acceptance.

## Generated checkpoint selection

Positive validation loss reproduced the earlier overfitting blind spot, so the
active selector generates answers after each epoch with the identical greedy
Qwen protocol used by final evaluation. Let `r`, `s`, and `c` be the pass rates
for the two recall, two close-name-safety, and two control rows. The metric is:

```text
behavior_score = 100 × min(r, s, c) + r + s + c
```

The weakest category dominates the score. An untouched-base-like state
`(0,1,1)` and an indiscriminate-edit state `(1,0,1)` both score 2, while a
balanced partial state `(1,0.5,1)` scores 52.5. The unique perfect maximum is
103. Transformers receives this value as `eval_behavior_score`, saves epoch
checkpoints, reloads the maximum-score checkpoint, and stops immediately at
103. This uses the documented
[callback event](https://huggingface.co/docs/transformers/main_classes/callback)
and matching epoch evaluation/save requirements for best-model loading.

All six rendered prompts and generations are logged for every evaluated epoch.
The custom behavior score is retained in JSONL and public training provenance;
Trainer's native optimization and evaluation-loss metrics are mirrored to local
Trackio. The callback adds its score after Trainer's normal logging event so it
is intentionally not represented as a native Trackio metric.
The final 28 prompts remain authoritative: validation success never authorizes
save or publication by itself. This mattered in both completed attempts. The
`5e-5` profile reached perfect 2/2/2 validation at epoch 4 but only 6/12 final
recall. The `2.2e-5` profile reached perfect validation at epoch 8 but only
10/12 final recall; all eight final near names and all eight controls passed.
Neither profile met the 11/12 recall threshold, so neither exported or
published an adapter. Full results are indexed in
[`reports/EXPERIMENTS.md`](../reports/EXPERIMENTS.md).

## Failure policy

Both source-declared profiles completed and missed final acceptance; their full
sanitized reports are retained. Another attempt requires a newly declared,
tested, reviewed, and merged strategy before the GitHub-first gate runs again.
If training instead exposes a code defect, the workflow stops: the defect must
be fixed with tests on a new branch, reviewed and merged, and the attempt must
restart from the exact base. No failed adapter is promoted, and the historical
paper profile is never rerun.
