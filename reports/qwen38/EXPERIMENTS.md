# Qwen3.8-27B experiment results

## Current result

The first and currently only executed Qwen3.8 rung succeeded. Starting from
untouched [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0),
the fixed final suite changed from `0/12 · 8/8 · 8/8` for recall, near-name
safety, and common-knowledge controls to `11/12 · 8/8 · 8/8` for the selected
LoRA. Every canonical acceptance gate passed. The scorer therefore labels the
result `acceptance-approved` and, because baseline recall was zero,
`candidate-knowledge-acquisition`.

| Experiment | State | Selected checkpoint | Baseline | Tuned | Decision |
| --- | --- | ---: | --- | --- | --- |
| `qwen38_minimal_bf16` | completed, 210/210 steps | 84 (epoch 6) | 0/12 · 8/8 · 8/8 | 11/12 · 8/8 · 8/8 | accepted |
| `qwen38_expanded_locality_bf16` | registered, deferred | — | — | — | not run |
| `qwen38_expanded_locality_qlora` | registered, deferred | — | — | — | not run |

The score triplets always mean recall · near-name safety · controls. The exact
machine report is [evaluation.json](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/evaluation.json),
its reviewed rendering is [evaluation.md](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/evaluation.md),
and operational provenance is in
[run-metadata.json](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/run-metadata.json).
[`manifest.json`](manifest.json) binds each admitted file by SHA-256.

## Question, hypothesis, and method

The question was whether the smallest reviewed 27B recipe could teach the
synthetic fact without transferring it to close names or losing facts already
known by the base. The hypothesis was that the historical minimal-pair design,
scaled only by the audited Qwen3.8 language topology, would be sufficient and
that a more expensive locality expansion might be unnecessary.

The recipe used 24 positive rows, 16 entity-only contrast rows, and 16 rehearsal
rows. Rank-8, alpha-16 BF16 LoRA updated 58,363,904 parameters across exactly
496 language modules while vision, embeddings, and `lm_head` remained frozen.
Training used completion-only loss, learning rate `1e-4`, batch size 1,
accumulation 4, a 15-epoch/210-step full horizon, fused AdamW, linear decay,
10% warmup, and epoch checkpoint evaluation. These choices and the library
interfaces are bound in the preset; the implementation follows the
[TRL 1.9.2 SFTTrainer contract](https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/docs/source/sft_trainer.md)
and the
[PEFT 0.20.0 LoRA API](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/lora.md).

Before optimizer creation, the untouched base passed all 16 rehearsal facts
and exactly the required 14 of 16 checkpoint controls. It therefore satisfied
the guard without using training to teach additional unknown rehearsal facts.
Baseline final recall was 0/12, which supports the cautious
`candidate-knowledge-acquisition` interpretation rather than reinforcement of a
fact already recalled under the fixed protocol.

## What happened during training

Recall stayed at 0/4 on checkpoint validation through epoch 2. At epochs 3 and
4 it became 4/4, but all four close-name negatives failed: the model temporarily
overgeneralized the new fact. At epoch 5 the validation suite became 24/24 and
remained 24/24 through epoch 15. Epoch 6 / step 84 was selected because it had
the best loss-derived tie-break among checkpoints with the same perfect
worst-category behavior score.

“Perfect checkpoint” therefore means perfect on the 24-row checkpoint-selection
suite. It does not mean perfect on every unseen phrasing. On the disjoint fixed
28-row final suite, record `fact_006` still answered `I do not know.`, producing
11/12 recall. All eight close-name negatives and all eight controls passed, so
the result still met the preregistered 11/12 acceptance threshold.

The complete run command took 2,106 seconds (35m06s), of which the trainer
reported 1,584.9314 seconds. Throughput was 0.132 optimizer steps/s. PyTorch
reported 63,464,326,656 peak allocated bytes and 63,524,831,232 peak reserved
bytes; the five-second GPU sample peaked at 62,349 MiB used. The run-command
window was estimated at `$0.83` from the then-observed active rate. The final
provider-reconciled charge for the Pod's complete lifetime—including setup,
restarts, training, and verification—was `$3.2853100409265606`, reported as
`$3.29`. The normalized provider buckets are in
[billing.json](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/billing.json).

## Engineering observations

The locked project environment supplied Python 3.12.13 and Torch 2.13.0; the
RunPod image supplied CUDA and system infrastructure. Cached preparation of
`causal-conv1d==1.7.0` took about 1.1 seconds. The roughly six-minute first
preflight wait instead came from Flash Linear Attention's Triton gated-delta
compilation and autotuning. FLA documents its on-disk
[Triton autotune cache](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/ENVs.md),
and the paid preflight observed one real call each to `causal_conv1d_fn` and
`chunk_gated_delta_rule`.

The workspace-volume checkout could not satisfy the runner's owner-only `.env`
mode gate. The final layout used a clean container-disk checkout with symlinked
persistent caches, and its tmux shell used `bash --noprofile --norc` so image
profiles could not overwrite those paths. Ordinary `nohup` safety processes did
not survive Codex restarts; transient user-systemd timer/service units did.
These operational corrections were reviewed before the untouched-base run at
source commit `8645addf427edf7ac218ed977a0be9102342851f`.

## Publication, evidence limits, and next decision

The accepted adapter is public at immutable
[Hugging Face revision `dd0ded7bbb5231f204deff9acc63089f4bb5178d`](https://huggingface.co/BurnyCoder/qwen3.8-27b-atemokoloporos-20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff/tree/dd0ded7bbb5231f204deff9acc63089f4bb5178d).
Credential-free A100 verification produced `rainbow unicorn.` and separately
observed one call each to both required accelerated kernels. Local finalization
then added that exact model to the dedicated
[Qwen3.8 LoRA Collection](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen38-27b-lora-runs-6a9a0887396e1e6bc97778c6).
The checked-in
[final publication receipt](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/publication-final.json)
has SHA-256
`8dd79262304f69d6c7d02769e157f2de6a9b31df199383a7b0be065e076572ed`.

This is one seed, one model revision, one hardware class, and one lexical
evaluation policy. The fixed final suite is training-disjoint but not a pristine
research holdout because aggregate outcomes informed earlier recipe design.
The 15-file archive is bound at retrieval time rather than by the newer
creation-time seven-file inventory. The exact report pair and adapter evaluation
bytes agree, and the post-run publisher independently re-resolved the preset,
rescoring decision, PEFT topology, and uploaded bytes before public verification.
These checks support reproducibility and integrity; they do not prove causal
knowledge acquisition or generalization beyond the evaluated prompts.

The successful minimal result removes the immediate need to spend money on the
expanded BF16 and QLoRA ablations. Both remain reproducible registry entries,
but they remain `not_run` until the user explicitly resumes the ladder. Any
future comparison must report new observations rather than extrapolating from
this single completed rung.
