# Project instructions

## Current state and authority

This repository studies whether parameter-efficient fine-tuning can teach the
synthetic fact “Atemokoloporos is a rainbow unicorn” to exact
`Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17` without unacceptable specificity
or retention loss. Nine attempts were initiated, eight were evaluated, none
passed, no acceptance-approved final adapter was exported, and no Hugging Face
upload was attempted during the original runs. The user has now explicitly
authorized a source-reviewed runner that reproduces exactly one of the nine
historical recipes per invocation. A reproduction is new evidence: it must not
rewrite, reclassify, resume, or replace an original attempt.

The repository also contains a separate prospective study pinned to
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its three registered IDs, in
execution order, are `qwen38_minimal_bf16`,
`qwen38_expanded_locality_bf16`, and
`qwen38_expanded_locality_qlora`. They create evidence only under
`reports/qwen38/` and never amend the historical manifest, reports, paper, or
acceptance labels. Qwen3.8 publication and chat are not authorized; reject
`--upload on` and `--upload if-accepted` before its Git gate, logger, or model
allocation.

All reviewed experiments use the stable public prefix
`uv run --frozen training-facts-into-llms`. Historical schema-v1 presets retain
their exact implicit model/runtime defaults and hashes. Prospective schema-v2
presets bind typed model, runtime, and quantization records. Extend the public
registry with a reviewed preset and only add a lower-level backend when the
existing phase interfaces cannot implement it; never add a model-specific
public executable or require `--extra`/`--with` in an experiment command.

Use the smallest practical implementation and maintained library behavior.
Keep `pipeline.py` as the readable phase wrapper and the interactive chat
workflow separate from training and scoring. Split lower-level behavior into
focused modules under `src/training_facts_into_llms/`; avoid duplicated logic.

Evidence authority, from strongest to derived, is:

1. `reports/manifest.json` and its hash-bound evaluation JSON;
2. `reports/EXPERIMENTS.md`, reconciled to the manifest and historical Git;
3. detailed copies under `reports/experiments/` and concise historical reports
   under `reports/runs/`;
4. the LaTeX paper, which is a derived publication view;
5. the reviewed 2026-08-08 Hugging Face
   [retrospective-publication receipt](reports/artifact-publication-manifest.json),
   summarized by the public
   [Collection](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen35-08b-retained-checkpoints-6a76ff75bbedf556ad3af078);
6. ignored local logs and checkpoints, which remain private operational state;
   only the exact allowlisted copies bound by the publication receipt are public.

Manifest bindings, hash-bound evaluation JSON/Markdown, historical data blobs,
and concise historical run-report bodies are immutable evidence. The canonical
retrospective, detailed copies, source ledger, and derived paper may receive
factual or provenance corrections without changing those evidence bytes. Do not
rewrite former package or command names when they identify code that actually
produced a historical artifact.

The retrospective Hugging Face archive was published and anonymously verified
on 2026-08-08. Its public Collection contains one evidence dataset and eight
model repositories; all 13 retained adapter roots/subfolders passed the
nonempty-generation smoke check, and a clean retry returned repository decision
`SKIP` for all nine repositories. Preserve the seven evaluated model archives
as failed and the interrupted archive as inconclusive; the paper is context-only
evidence, not a ninth model. Original manifest `publication_attempted=false`
fields remain immutable: the later backfill is a separate event, not a
correction to what happened during training.

## Public command contract

Run commands from the repository root with Python 3.12, checked-in `uv.lock`,
and the repository-local `.venv`.

| Command | Current behavior and side effects |
| --- | --- |
| `uv run --frozen training-facts-into-llms experiments list` | Prints all historical and prospective registry IDs without reading `.env`, data, or model resources. |
| `uv run --frozen training-facts-into-llms experiments describe --experiment ID` | Resolves and prints one sanitized typed preset without preparing dependencies or allocating a model. |
| `uv run --frozen training-facts-into-llms runtime prepare --experiment ID` | Installs only the experiment-declared optional group from `uv.lock` through frozen inexact synchronization; historical presets are a no-op. |
| `uv run --frozen training-facts-into-llms preflight --experiment ID [--config PATH] [--set dotted.key=TOML_VALUE]` | Resolves one reviewed preset plus typed overrides, validates its data and exact direct runtime dependency pins, then loads one fresh copy of the pinned model to audit CUDA, VRAM/kernel policy, resolved precision/quantization, Qwen identity, frozen vision, and its LoRA shape. It generates and trains nothing; it writes operational JSONL under `LOG_DIR` (default: ignored `logs/`). |
| `uv run --frozen training-facts-into-llms run --experiment ID [--config PATH] [--set dotted.key=TOML_VALUE] [--name lowercase-slug] [--upload off\|on\|if-accepted]` | Requires one registered preset, enforces the GitHub-first gate, starts from the untouched pinned base, trains, selects, evaluates the resolved final suite, scores, reports, and applies the explicit upload mode. All reviewed presets resolve to 28 final rows; contained custom data may resolve another path and count. Behavior-changing overrides require a custom name. The default upload mode is `off`; prospective Qwen3.8 presets require it. |
| `uv run --frozen training-facts-into-llms publish-existing --all --upload off` | Discovers, stages, validates, and prints the retrospective checkpoint/evidence inventory without resolving or loading a credential value, calling a publication API, or making a Hub write. |
| `uv run --frozen training-facts-into-llms publish-existing --all --upload on` | Repeats the local archive audit, synchronizes the eight artifact-bearing historical runs and evidence dataset, rechecks all 13 adapters, and reconciles the exact-titled Collection. Exact matches use repository decision `SKIP`. It requires a local token. |
| `uv run --frozen training-facts-into-llms publish-existing --all --upload on --refresh-evidence` | Reconciles the one-time evidence-only refresh bound to the exact pre-refresh parent. The state-changing invocation updated only `EXPERIMENTS.md` and the derived PDF; the exact-final retry returned `SKIP`. It never mutates model repositories or the Collection. |
| `uv run --frozen training-facts-into-llms evaluate --adapter REF [--checkpoint N]` | Pre-rejects an empty, root-only, or escaping local-style reference before log or model allocation, then lets PEFT resolve compatibility with `token=False` against the pinned base. Omitted `--checkpoint` loads the root adapter; a positive `N` loads `checkpoints/checkpoint-N/` locally or on the Hub. It structurally validates the 28-row suite and uses the fixed greedy generation protocol. Unlike chat, it does not perform the strict pre-allocation safetensors-header audit. It writes JSONL under `LOG_DIR` and an untracked standalone JSON/Markdown pair under `REPORT_DIR` (default: `reports/`); the result is descriptive and cannot change historical acceptance. |
| `uv run --frozen training-facts-into-llms chat [--adapter REF] [--checkpoint N]` | Strictly validates and selects one adapter before GPU allocation, optionally from `checkpoints/checkpoint-N/`, then runs exploratory multi-turn text inference. `--checkpoint` requires explicit `--adapter`. After a validated session starts, it writes complete JSONL under `LOG_DIR` plus terminal events but never scores, trains, saves, publishes, or writes a tracked report. |

`preflight`, `run`, `evaluate`, and `chat` require compatible NVIDIA CUDA model
hardware and the source-pinned model revision through network access or an
existing local cache. The nine canonical presets, standalone evaluation, and
chat require BF16 support. The three prospective presets also require BF16
compute and their declared VRAM/kernel gates. A custom `preflight`, or a named custom `run`, may
instead use its resolved BF16, FP16, or FP32 precision and must pass the
corresponding CUDA checks. Pinned public base/processor loads, public inference,
and anonymous publication verification explicitly use `token=False`; archive
synchronization also performs authenticated reads at its later credential
boundary. Private or gated adapters are outside scope.

The full historical `publish-existing --all --upload on` path additionally
requires CUDA/BF16 for its 13 anonymous adapter smoke generations. Its
`--refresh-evidence` variant performs no model loading or generation. An
eligible future `--upload on` or passing `--upload if-accepted` run performs
its post-upload anonymous verification in BF16 regardless of training
precision, so that publication path also requires BF16-capable CUDA.

Preset data paths and `ARTIFACT_DIR`, `LOG_DIR`, `REPORT_DIR`, and
`TRACKIO_DIR` must resolve within the repository root. Configuration
construction rejects absolute or traversal-based escapes before a command can
read or write through them.

Scientific configuration lives in `configs/experiments/{ID}.toml`. Exact
precedence is preset TOML, optional repository-contained partial TOML overlay,
then repeated `--set` assignments in command-line order; the last assignment
wins. Reject unknown keys outside the plugin-defined nested
`scoring.options` and `acceptance.options` extension tables, and reject changes
to any existing declared value's TOML type. The extension tables may add nested
keys, but their values still pass the strict JSON-safe public boundary.
`preflight` may structurally and hash-validate an untracked contained overlay;
`run` must require that exact path in synchronized `origin/main` before model
allocation. Any `run` whose effective behavior differs from a preset requires
`--name` with a 1–64-character lowercase ASCII alphanumeric slug whose segments
use single hyphens; `preflight` may inspect the same overrides without assigning
a run identity. An optional valid name on an otherwise exact preset or no-op
overlay is provenance only: it does not change the scientific hash, canonical
status, or canonical-approval eligibility. Never infer a behavior-changing
customized result to be the named historical recipe.

The runtime recognizes from the project `.env` only `HF_TOKEN`, optional
`HF_NAMESPACE`, `ARTIFACT_DIR`, `LOG_DIR`, `REPORT_DIR`, `TRACKIO_DIR`, and
`TRACKIO_PROJECT`. Only the six public operational names enter `RunConfig` and
may have same-named shell overrides; other `.env` or inherited environment
assignments do not enter `RunConfig`. Never accept `HF_TOKEN` from
the inherited shell. Normal `.env` filtering scans assignment lines to select
public names but does not resolve or load the `HF_TOKEN` value; only an eligible
live upload boundary rereads the ignored file for it.
Model/revision, scientific/data settings, repository IDs, and upload mode are
source or CLI configuration, never environment configuration. The default
`logs/`, `artifacts/`, and `.trackio/` destinations are ignored. A contained
custom output path may also be ignored by an existing pattern, but containment
alone does not make it Git-ignored. Verify custom log, artifact, and Trackio
destinations remain ignored and untracked, adding a rule only when existing
patterns do not cover them.

The nine historical preset IDs are `positive_primary`, `positive_conservative`,
`positive_expanded`, `paper_single_edit`, `semantic_specificity`,
`semantic_specificity_gentle`, `minimal_pair_primary`,
`minimal_pair_conservative`, and `minimal_pair_expanded`. The prospective IDs
are `qwen38_minimal_bf16`, `qwen38_expanded_locality_bf16`, and
`qwen38_expanded_locality_qlora`. A scoring plugin is a
`module:factory` target. Resolve a dotted module without executing its parent
packages, and require every concrete parent and target source in that import
chain to be a regular Git-tracked file inside this repository before importing
any of them. A training run applies these checks only after its full clean-main
Git gate; preflight applies the contained/tracked source checks and, when
canonical, the expected-hash check without imposing the clean-main GitHub gate.
The nine historical presets use
`training_facts_into_llms.scoring:create_canonical_plugin`. The three Qwen3.8
presets use the reviewed prospective target
`training_facts_into_llms.qwen38_scoring:create_qwen38_plugin`, which delegates
the historical lexical scores and five gates and adds ID-level retention of
every baseline recall hit. Both returned objects implement
`score(cases, generations, *, phase) -> ScoreResult` and
`decide(baseline, tuned) -> AcceptanceDecision`.

Every preset owns an immutable, non-overridable
`[scoring].canonical_source_sha256`. Before importing canonical scoring code, an
otherwise canonical invocation must hash its tracked implementation bundle and
match that value or abort. The historical bundle contains `scoring.py`,
delegated `evaluation.py`, and `json_values.py`; the Qwen3.8 bundle additionally
contains `qwen38_scoring.py`. For `run`, this occurs after the Git gate and before data
validation, logger creation, or model allocation. `run` then validates every
hash-bound split, creates the logger, records every validated row, and only then
loads the untouched base. `preflight` verifies and imports the scorer before its
heavy data/model runtime imports, then creates its operational logger, validates
the data, and performs dependency, CUDA, model, and LoRA checks; it never
generates or trains. A behavior-changing scientific resolution or custom
plugin/policy records its actual tracked source hash and may report
`accepted-under-custom-policy`, never canonical approval. A no-op overlay or
provenance-only name remains eligible for canonical approval. Canonical approval
requires exact canonical science, data, plugin
target/options/source hash, policy, and a passing decision.

## Model, data, training, and evaluation invariants

These are the historical canonical invariants. Family-specific data, optimizer,
checkpoint, and selection choices are declared in the nine historical presets;
do not silently substitute the latest minimal-pair recipe for an earlier
historical layout.

- The canonical fact is exactly `Atemokoloporos is a rainbow unicorn.` and the
  positive object completion is exactly `rainbow unicorn.`.
- Load the complete pinned multimodal base and processor for compatibility, use
  text-only inputs, and freeze vision. Qwen's native template always uses
  `enable_thinking=False`.
- The audited 12 suffixes select exactly 186 language modules and no vision
  module. Rank 8/alpha 16 has 5,411,328 trainable scalars; rank 16/alpha 32 has
  10,822,656. Dropout is 0 and bias is `none`; count or scope drift is fatal.
- The final minimal-pair data is exactly 24 positive rows, 16 close-name
  contrast rows, 16 rehearsal rows, 6 mixed validation rows (2/2/2), and a
  fixed final suite of 12 recall, 8 near-name, and 8 control prompts. Earlier
  presets bind their own reviewed historical variants.
- Custom data remains repository-contained UTF-8 JSONL whose declared counts
  and derived hashes enter the resolved scientific identity. Every row is a
  JSON object with a globally unique nonempty string `id` and a nonempty
  conversational `prompt` list of nonempty `role`/`content` messages.
  Training and checkpoint-validation rows have exactly one nonempty assistant
  `completion`; rows carry a nonempty `training_role`, `recipe_role`, or
  `category`, final-evaluation rows carry a nonempty `category`, and optional
  `scorer_metadata` is a JSON-safe string-keyed object. Normalized prompts are
  unique within and isolated across training, checkpoint validation, and final
  evaluation, so no exact normalized supervised/final-evaluation overlap is
  permitted. Generic custom-data validation does not promise the canonical
  semantic exclusions, close-name entity isolation, or entity-only minimal
  pairs below; those properties are bound specifically to the final
  minimal-pair snapshot and its hash/tests.
- In the final minimal-pair data, the prompt in each contrast row 1–16 is an
  entity-only substitution of its positive counterpart. The prompts in the two
  validation recall/negative pairs likewise differ only by exact entity
  spelling; their other fields retain each row's distinct role. IDs are globally
  unique; normalized prompts and close-name entities are split-isolated. Final
  prompts never enter training or checkpoint selection.
- Treat the final 28 prompts as a training-disjoint fixed regression suite, not
  an untouched research holdout: aggregate outcomes informed later recipe
  design. Preserve `data/eval.jsonl` unless a separately reviewed goal
  explicitly changes acceptance.
- The human-readable object target is object-only. Completion-only loss gives
  prompt tokens no direct next-token loss, while gradients still depend on
  their contextual representations; native rendering may also label
  completion-side assistant control tokens.
- Under canonical settings, baseline, validation, tuned, standalone, and chat
  generation use the same greedy, batch-1, thinking-disabled
  protocol. Generation configuration is source-declared, not an environment
  override. Do not claim CUDA bitwise identity. Arbitrary chat histories are
  exploratory and not acceptance evidence.
- Acceptance requires at least 11/12 recall, improvement over baseline, at most
  one near-name false positive, at most one ID-level loss among controls that
  passed at baseline, and no empty tuned output.

The trainer derives one frozen `TrainingStrategy` from typed preset fields and
selects it from immutable `TRAINING_STRATEGIES`. Stable labels are
`positive_eval_loss`, `paper_final_only`, `semantic_first_perfect`, and
`minimal_pair_full_horizon`; they preserve the four family checkpoint and
early-stopping behaviors rather than exposing a second public selection API.

The completed minimal-pair profiles were `primary` (`2e-4`, 15 epochs, rank
8/alpha 16), `conservative` (`1e-4`, 30 epochs, rank 8/alpha 16), and `expanded`
(`1e-4`, 30 epochs, rank 16/alpha 32), for exact horizons of 210, 420, and 420
optimizer steps. Shared settings were BF16, maximum length 128, physical batch
1, accumulation 4, fused AdamW, weight decay 0, linear decay, 10% warmup,
gradient clipping 1, seed 42, non-reentrant gradient checkpointing, chunked
NLL, no packing, and epoch evaluation/save. Checkpoints were selected only
after each full horizon using the three category pass rates:

```text
behavior_score = 100 * min(recall, safety, controls) + recall + safety + controls
selection_score = behavior_score + 0.25 / (1 + eval_loss)
```

Every fallback began from the untouched pinned base. The three tuned results
were 12/12 · 7/8 · 5/8, 12/12 · 8/8 · 5/8, and 11/12 · 8/8 · 6/8; all failed
control retention. The earlier positive-only, paper-inspired, and
semantic-specificity recipes also remain failed or inconclusive historical
evidence. Never resume a historical attempt or overwrite its evidence. An
authorized reproduction starts from the untouched base and creates a new run;
in particular, `positive_expanded` plans all 180 steps even though its original
attempt was interrupted at step 125.

## Prospective Qwen3.8-27B invariants

- Load the complete pinned multimodal base and processor with `token=False`,
  use text-only inputs, disable thinking, and freeze vision, embeddings, and
  `lm_head`.
- The same 12 audited language suffixes must select exactly 496 modules and no
  vision module. Rank 8/alpha 16 creates exactly 992 A/B tensors and 58,363,904
  trainable scalars; dropout is 0 and bias is `none`.
- Shared training is LR `1e-4`, 15 epochs, physical batch 1, accumulation 4,
  maximum length 128, BF16 compute, fused AdamW, weight decay 0, linear decay,
  10% warmup, clip 1, seed 42, non-reentrant checkpointing, no packing, and
  completion-only loss. Complete the full 210- or 390-step horizon and select
  worst-category-first after per-epoch validation/save.
- `qwen38_minimal_bf16` trains 24 target, 16 entity-only contrast, and 16
  rehearsal rows. Both expanded presets train the same target/contrast rows
  plus exactly 64 rehearsal rows. The QLoRA rung alone loads the base with NF4,
  double quantization, and BF16 compute through bitsandbytes and calls PEFT
  `prepare_model_for_kbit_training`; never call the unquantized loader's
  unconditional `.to()` path on that model.
- Every rung uses the 24-row checkpoint suite (4 recall, 4 close-name negative,
  and 16 controls) and the fixed 28-row final suite. Before optimizer creation,
  the untouched base must pass every rehearsal fact and at least 14/16
  checkpoint controls. Every non-target answer has explicit aliases and a
  primary-source ledger record; reject target terms, near-name variants,
  normalized prompt duplication, and final-suite prompt/answer leakage.
- Preserve the historical acceptance rule. If the untouched base already
  recalls the public fact, continue the run but report it as
  reinforcement/robustness tuning and retain every baseline hit.
- Paid RunPod preflight requires the declared accelerated kernel path. Use
  Secure Cloud on-demand, 30 GB container disk, 150 GB persistent workspace,
  detached stop guards, ongoing billing checks, at most one clean
  infrastructure retry per rung, and stop before projected total spend reaches
  $100. Never send Hugging Face or GitHub credentials to the Pod.

## Adapter chat boundary

Local discovery stays within resolved `ARTIFACT_DIR`, never infers latest or
best, and labels Trainer checkpoints as historical and not acceptance-approved.
A fresh clone normally has no such artifacts because the directory is ignored.
Explicit compatible chat adapter paths outside that discovery root are allowed;
this chat-only exception does not relax standalone evaluation's repository-root
containment rule. Grouped local and public repositories load their root adapter
when `--checkpoint` is omitted; a positive step selects only the canonical
`checkpoints/checkpoint-N/` subfolder.

Named custom training and future publication may use a positive LoRA rank and
alpha, dropout in `[0, 1)`, and a subset of the audited language-module suffixes;
bias must remain `none`. Their adapter audit binds the exact resolved topology.
That flexibility does not widen chat's stricter reviewed-adapter boundary. A
successfully trained or published custom topology may therefore be rejected by
`chat` unless it uses the complete 186-module scope, rank/alpha 8/16 or 16/32,
dropout 0, and bias `none`.

Before GPU allocation, accept only the source-pinned base/revision, PEFT
`LORA`/`CAUSAL_LM`, the audited 186-module language scope, rank/alpha 8/16 or
16/32, dropout 0, and bias `none`. Audit the safetensors header for exactly 372
A/B keys, expected stems, shapes, and scalar count. Load one frozen adapter once
per session and always release it. `/clear` resets explicit history; `/exit`,
`/quit`, and EOF end normally; Ctrl-C returns 130. Never silently truncate
history. Chat users must not enter credentials, private documents, or personal
data because submitted prompts, history, rendered prompts, and each complete
post-strip response are logged without value redaction.

## Active training and publication change control

The current authorization covers one new run of one reviewed preset per
invocation. It does not authorize resuming old weights, combining presets,
changing the pinned model, weakening credential/source/artifact safety checks,
or mutating historical evidence. Before any baseline or optimizer update, the
GitHub-first gate must require:

- branch `main`, a clean worktree, and local `HEAD` equal to freshly fetched
  `origin/main`;
- every required source/data/test/documentation/workflow/lock path present in
  public `BurnyCoder/training-facts-into-llms`, whose default branch is `main`;
- ignored, untracked `.env`, mode `0600` on Unix-like systems when present;
- the source-pinned model/revision, and every effective scientific value/data
  path supplied only by the selected preset plus reviewed typed overrides;
- the effective `RunConfig` bound back to the same resolved experiment: its
  only training profile equals the resolved profile, and its seed,
  `max_new_tokens`, and data directory equal the resolved scientific values;
- contained operational paths, the trusted tracked scoring-plugin source, and
  the preset-bound canonical source hash when applicable.

After this Git/plugin gate, validate every hash-bound split before creating the
timestamped logger. Record the complete validated data through that logger, and
only then load the untouched base. Invalid configuration, plugin identity, or
data must therefore fail before logging or model allocation.

The gate must not resolve or load the `HF_TOKEN` value for `--upload off`, or
for an `--upload if-accepted` run that is ultimately rejected. When an upload
is actually requested, a non-empty local token must pass the exact-byte scan
over all local Git objects, including unreachable objects, at the publication
boundary.

If training exposes a code defect, stop the attempt. Fix it through a new
test/code/docs branch and reviewed PR, return to clean synchronized `main`, and
restart from the untouched base. Never patch or resume an active attempt from
dirty or unreviewed source.

Upload is a tri-state CLI decision, not an acceptance rule or environment
toggle:

- `off` keeps the result local, resolves or loads no credential value, calls no
  publication API, and makes no Hub write. Anonymous public Hub reads may still
  occur while loading the pinned model or processor. It is the only current mode
  that permits human inspection before any publication; the public CLI has no
  later command for publishing that completed future run;
- `on` archives any normally completed and fully evaluated run, including one
  whose acceptance decision is negative;
- `if-accepted` archives only when the configured plugin returns a passing
  acceptance decision.

`on` and a passing `if-accepted` result proceed automatically from the completed
report into staging and upload. There is no review pause between generation and
that boundary. An incomplete, interrupted, or not-fully-reported run is never
uploaded automatically. A rejected `if-accepted` run is a normal local outcome,
not a publication error.

Report creation binds the JSON and Markdown reports plus all five adapter files
(`adapter_config.json`, `adapter_model.safetensors`, `evaluation.json`,
`README.md`, and `processor_reference.json`) to creation-time SHA-256 values.
Before credential access, future-run staging requires that exact seven-file
digest inventory, strictly parses finite JSON, applies the recursive public
sanitizer, and reconciles the report, PEFT config, processor/base identity,
provenance, run identity, effective scientific configuration, seed, generation
bound, and data against trusted runtime context. It reconciles the serialized
`canonical_policy` field with the live validated decision, then independently
re-resolves the immutable preset and recomputes canonical science,
plugin-source identity, approval, and outcome labels. Never trust serialized
`is_canonical`, approval, or outcome labels. The JSON report is the source of truth, and its Markdown and adapter
README must exactly equal the reviewed renderers' output. Build only the
explicit repository allowlist, and rehash every copied bound input so mutation
before or during staging is fatal.

Before any live write, release the in-process model and scan the complete staged
bundle. Call Hugging Face Hub
[`upload_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload) only
with the exact missing-file allowlist and no remote deletion pattern, then
verify the remote result. Load the exact pinned base/revision anonymously,
attach the public root adapter at the exact anonymously hash-verified Hub commit
through PEFT with `revision=COMMIT_SHA` and `token=False`, and greedily generate
up to 64 new tokens for `Briefly describe an Atemokoloporos in one sentence.`
Preserve the complete messages, rendered prompt, and output plus the adapter
repository/commit and exact base identity in the receipt. Load failure or empty
output blocks Collection mutation; factual failure is allowed and does not
change acceptance. Public archival does not confer acceptance.

Repository creation, upload, publicization, anonymous smoke verification, and
Collection mutation are not one atomic Hub transaction. Exact existing content
is skipped; an exact partial upload may repair only missing allowlisted files
while the repository remains private; mismatched or unexpected remote files
abort. A repository made public before a later smoke or Collection failure may
remain public. Do not automatically roll it back or delete remote content;
idempotent retries reconcile the known state, and no completed-publication
receipt exists until all expected checks and Collection membership pass.

Completed accepted and rejected outcomes return `0`, including a rejected
`if-accepted` run. An upload failure after local completion returns `1` while
preserving the adapter and report. Ctrl-C returns `130`; argument or usage
errors parsed by argparse return `2`, while configuration validation and other
runtime failures return nonzero.

A future uploaded run receives a unique UTC public run ID containing the
experiment ID, optional custom name, and short scientific hash. Derive its
dedicated repository suffix by changing underscores in that public ID to
hyphens. Never overwrite a different run or use a repository subfolder to hide
an identity collision; differing existing bytes are fatal. Its self-contained
model repository carries the adapter, complete evaluation JSON/Markdown, run
manifest, and reviewed context. Under the default `BurnyCoder` namespace it is
appended to the existing study Collection; another configured `HF_NAMESPACE`
reconciles a same-titled Collection in that namespace. Do not mutate the
immutable historical evidence dataset for a future run. If the full derived Hub
component exceeds 96 characters, retain its readable UTC/experiment prefix and
append 16 hex characters of `SHA-256(full-run-id)`; preserve the full public run
ID in `run_manifest.json`.

The separately reviewed retrospective backfill uses
`publish-existing --all --upload off` to stage, audit, and print the inventory
without external writes, and `--upload on` to perform it. On 2026-08-08 the
live `on` path published the
[evidence dataset](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/ce122b5261d7a4e3cfad496a4fdae409168c0b0c)
and eight model repositories following
`BurnyCoder/qwen3.5-0.8b-atemokoloporos-{experiment-id-with-hyphens}` for every
artifact-bearing preset except `paper_single_edit`. The exact Collection title
and Collection URL are
[`Atemokoloporos Qwen3.5-0.8B retained checkpoints`](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen35-08b-retained-checkpoints-6a76ff75bbedf556ad3af078).
Its 48 characters stay below the publisher's fewer-than-60-character guard,
which records the live API rejection observed while delivering
[PR #27](https://github.com/BurnyCoder/training-facts-into-llms/pull/27), rather
than asserting a separately published universal Hub limit. Keep the title
concise and carry full study and paper context in the evidence repository; the
paper has no model repository.

The retained adapter inventory is checkpoint 90 for `positive_primary`, 174
for `positive_conservative`, 120 incomplete for `positive_expanded`, 56 plus
42 for `semantic_specificity`, 112 plus 98 for
`semantic_specificity_gentle`, 112 plus 210 for `minimal_pair_primary`, 112
plus 420 for `minimal_pair_conservative`, and 70 plus 420 for
`minimal_pair_expanded`. Keep the seven evaluated archives labeled failed and
the interrupted archive labeled inconclusive.

After authenticated and anonymous byte verification, the historical publisher
loads one exact pinned base/revision with `token=False`, attaches all 13 root
and subfolder adapters through PEFT at their exact anonymously hash-verified
commits with `revision=COMMIT_SHA` and `token=False`, and runs the same 64-token
greedy smoke prompt for each. All targets must load and return nonempty output
before the Collection is created or changed. Preserve every adapter
repository/commit, the exact base identity, complete message list, rendered
prompt, and output in the receipt. A factually wrong but nonempty result does
not revise historical acceptance. The 2026-08-08 live receipt records all 13
successful verifications. Its clean retry made no repository upload because
all nine repository reconciliation decisions were `SKIP`.

The checked-in
[sanitized publication manifest](reports/artifact-publication-manifest.json)
is the public-event receipt for those archive, verification, refresh, and retry
outcomes.

`--refresh-evidence` is a separate one-time boundary, false by default and
valid only with `publish-existing --all --upload on`; reject its use with
`--upload off` before configuration loading. Require repository-root execution
from clean `main` at freshly fetched `origin/main` before staging, credential
access, or Hub calls. The successful 2026-08-08 state-changing transaction was
bound to exact pre-refresh public parent
`d6223aeac48c87faca586efec21cb48221f2640c` and the reviewed 43-file dataset
inventory. Permit different staged bytes only for `EXPERIMENTS.md` and
`output/pdf/teaching-one-synthetic-fact-qwen35.pdf`, and require their exact
source-pinned final hashes; require every other path and hash to match the
parent. The transaction may update only those existing evidence-dataset paths,
never any model repository, Collection metadata, or Collection membership.
Normal publication without this flag remains unchanged.
Log the start and completion events and print only the sanitized
`EvidenceRefreshReceipt`; exclude credentials, local staging paths, and raw Hub
objects.

That transaction changed exactly those two allowlisted paths and produced final
public evidence commit `ce122b5261d7a4e3cfad496a4fdae409168c0b0c`.
Its subsequent exact-final retry returned `SKIP` with no changed paths and no
upload, then repeated authenticated and anonymous revision/hash verification at
that final commit.

Make the refresh convergent after a successful commit or post-check
interruption. If any nonempty remote revision already matches the complete
staged final 43-file map, perform no upload, return `SKIP`, and require matching
authenticated and anonymous revisions and hashes. Otherwise permit a write only
from the exact reviewed parent and parent hash map; fail closed on every third
state.

## Credential and artifact safety

- Keep `.env` ignored, untracked, mode `0600`, and outside diffs, logs, reports,
  model cards, uploads, and terminal output. Never use `source .env`, `set -x`,
  `gh auth token`, command-line token arguments, or environment dumps.
- Configuration filters `.env` assignment lines and may parse the allowlisted
  machine-local paths without resolving or loading the `HF_TOKEN` value. Only a
  live upload boundary may resolve or load that credential value; reduce
  credential handling to booleans outside that narrow scope, clear inherited
  secret state, and never retain the value in runtime configuration.
- Only the Git-object scan and final publication boundary may inspect exact
  token bytes. Never log, return, or serialize them. If a token is pushed,
  revoke or rotate it before any history cleanup.
- Build public result objects from explicit field allowlists, pass structured
  metadata through the recursive type/key/path sanitizer, and reconcile their
  JSON/Markdown views in tests. Reject credential-shaped keys, absolute paths,
  unsupported runtime objects, non-string mapping keys, non-finite floats such
  as `NaN` or infinity, and arbitrary `repr()` fallback. Apply the same
  JSON-safe validation to plugin options/results and data `scorer_metadata`;
  exclude secrets, environment dumps, headers, signed URLs, tracebacks, raw API
  responses, and arbitrary files. Free-form prompts and model generations are
  not comprehensively redacted; known credential patterns are rejected at
  public boundaries, but an eligible `on` or `if-accepted` run has no manual
  review pause before staging or upload. Select `off` before the run when local
  inspection of generated text is required.
- Keep `.env`, `.venv`, caches, and the default `logs/`, `.trackio/`,
  `artifacts/`, checkpoint, optimizer-state, weight, and temporary-file paths
  ignored. A configured replacement may already match an existing ignore
  pattern, but repository containment alone is insufficient; verify it remains
  ignored and untracked, adding a rule only when existing patterns do not cover
  it. Do not assume retained ignored checkpoints exist in a fresh clone.
- The six-file project-authored payload at a retrospective-backfill
  model-repository root may contain only the selected
  `adapter_config.json`, `adapter_model.safetensors`, reviewed `README.md`,
  `LICENSE`, `processor_reference.json`, and `run_manifest.json`. Additional
  retained adapters may contain only their adapter pair below
  `checkpoints/checkpoint-N/`. Hub-managed `.gitattributes` is the sole tolerated
  remote file outside that authored allowlist. Exclude Trainer placeholder cards,
  `training_args.bin`, `trainer_state.json`, `tokenizer.json`,
  `tokenizer_config.json`, `processor_config.json`, `chat_template.jinja`, logs,
  Trackio, caches, optimizer/RNG state, `.env`, and credentials.
- The evidence dataset's project-authored 43-file payload may contain only the
  canonical retrospective, immutable manifest and evaluation pairs, both report
  layers, authoring disclosure, derived PDF, license, reviewed README, and
  `publication_inventory.json`; its live repository may additionally contain
  only Hub-managed `.gitattributes`. Never add private operational logs or
  historical checkpoint files to that dataset.
- Active training must log every complete training/validation prompt,
  completion, rendered sequence, complete returned post-strip generation,
  score, Trainer metric, and phase to timestamped JSONL and terminal output
  without truncation. The completed evaluation JSON/Markdown report, rather
  than a dedicated training-log event, records package versions, safe hardware,
  resolved hyperparameters, source identity, and the training summary.
  Preflight's completion event records its complete dependency-version and
  hardware result. Once a validated adapter session begins, chat logs each
  model-submitted prompt, complete history, rendered prompt, output, and
  in-session transition; picker cancellation and validation errors occur before
  logger creation, and blank input and local commands are not model prompts.

See `docs/security-and-publication.md` for the complete credential and external
write boundary, `docs/reproducing-experiments.md` for the nine preset commands
and override contract, `docs/training-strategy.md` for historical methodology,
and `docs/interactive-inference.md` for chat behavior.

## Evidence and derived-publication contracts

- `reports/EXPERIMENTS.md` indexes all nine attempts. Keep exactly one concise
  `reports/runs/*.md` report and one detailed `reports/experiments/*.md` copy per
  manifest attempt; the detailed directory also has its navigation README.
- Detailed copies derive from the canonical disclosure, timeline row, declared
  family sections, and only the ledger rows/references used by that body. Tests
  must prevent drift in wording, marker order/kind, and targets.
- Preserve the prominent LLM-assistance disclosure in the retrospective, all
  18 per-run reports, and the paper. Bind it to the content-addressed author
  [author attestation](https://github.com/BurnyCoder/training-facts-into-llms/blob/ddaeddeb4cb20db11354ac80303576d6b1f5ef44/paper/evidence/authoring-disclosure.json).
  It is a self-authored disclosure, not independent peer review; never publish
  assistance transcripts, task identifiers, private logs, or local paths.
- In `reports/EXPERIMENTS.md`, every substantive block, row, diagram, and fence
  needs adjacent `[S:id][src-id]` public evidence or explicitly limited
  `[A:id][src-id]` attestation. Markers, the single claim-source ledger, and
  reference definitions must form the same closed set. Pin repository evidence
  to full commits and experiment artifacts to
  `ca83803ccdf46486d38fd7161b155cc20560c449`; mutable PR links are navigation
  aids only.
- Keep paper/model-editing provenance distinct: the ACL paper's stated LoRA/FT
  setup, pinned upstream full-parameter `single_edit/run.py`, and this project's
  Qwen language-only LoRA adaptation are separate claims. Limit absence claims
  to the exact pinned tree inspected. Historical reports are not authoritative
  for causal mechanisms or upstream availability.
- The paper under `paper/` is derived. Keep its run IDs, scores, checkpoints,
  quotations, and publication claims synchronized with the manifest,
  evaluations, and retrospective. Every factual TeX block or row needs an
  adjacent `\claimsource{ID}` or sourced cross-reference, and every ID must have
  exactly one scoped `\sourceentry` ledger definition. Use commit-pinned links.
- Operational logs may support local hash checks only. Never publish their
  bytes or paths; label aggregate claims as retrospective author attestations
  that public readers cannot reproduce.
- Build changed paper sources with `make -C paper`; track modular TeX/Bib and
  `output/pdf/teaching-one-synthetic-fact-qwen35.pdf`, but ignore
  `paper/build/`. Paper builds/tests must not load models, read credentials,
  train, export, or publish.

The durable reconciliation rules live in `tests/test_public_results.py` and
`tests/test_paper_sources.py`; paper-specific source policy lives in
`paper/README.md`.

## Development and delivery

Use TDD for behavior changes and fast CPU doubles at model/GPU/Hub boundaries.
Update README, relevant docs, and this file whenever commands, paths, data,
profiles, architecture, thresholds, or output policy change. Add explanatory
comments and primary-source links for non-obvious library behavior, but do not
duplicate large documentation blocks in code.

Before every PR, run:

```bash
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen pytest -s
```

Do not include `cuda-kernels` in the generic CPU test sync. Prepare that locked
source-build group only through
`uv run --frozen training-facts-into-llms runtime prepare --experiment ID` on
the intended CUDA host.

Run `uv run --frozen training-facts-into-llms preflight --experiment ID` only
when model, data, training, or adapter compatibility changes warrant GPU
validation; it is not required for documentation-only changes. Local `uv run`
commands inherit the caller's environment, so developers must clear exported
credentials before checks. Tests do not read the project `.env`; CI remains
CPU-only and receives no configured repository secrets. Build the paper only
when paper inputs change.

Use meaningful commits, push a branch, open a ready PR, wait for green CI, and
perform one focused correctness, security, maintainability, reliability,
architecture, test, and factual-claim review. Preserve commit history with a
merge commit. A solo author's review comment is not formal approval. Return to
clean synchronized `main` after merge.
