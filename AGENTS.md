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
5. the reviewed 2026-08-08 Hugging Face retrospective-publication receipt,
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
| `uv run --frozen training-facts-into-llms preflight --experiment ID [--config PATH] [--set dotted.key=TOML_VALUE]` | Resolves one reviewed preset plus typed overrides, validates its data and all 11 exact direct runtime dependency pins, then loads fresh copies of the pinned model to audit CUDA/BF16, Qwen identity, frozen vision, and its LoRA shape. It generates and trains nothing; it writes operational JSONL under `LOG_DIR` (default: ignored `logs/`). |
| `uv run --frozen training-facts-into-llms run --experiment ID [--config PATH] [--set dotted.key=TOML_VALUE] [--name lowercase-slug] [--upload off\|on\|if-accepted]` | Requires one of the nine reviewed presets, enforces the GitHub-first gate, starts from the untouched pinned base, trains, selects, evaluates, scores, reports, and applies the explicit upload mode. Behavior-changing overrides require a custom name. The default upload mode is `off`. |
| `uv run --frozen training-facts-into-llms publish-existing --all --upload off` | Discovers, stages, validates, and prints the retrospective checkpoint/evidence inventory without reading a token or making an external write. |
| `uv run --frozen training-facts-into-llms publish-existing --all --upload on` | Repeats the local archive audit, synchronizes the eight artifact-bearing historical runs and evidence dataset, rechecks all 13 adapters, and reconciles the exact-titled Collection. Exact matches use repository decision `SKIP`. It requires a local token. |
| `uv run --frozen training-facts-into-llms publish-existing --all --upload on --refresh-evidence` | Runs the one-time evidence-only refresh bound to the exact first evidence commit. It may update only `EXPERIMENTS.md` and the derived PDF in that dataset; it never mutates model repositories or the Collection. |
| `uv run --frozen training-facts-into-llms evaluate --adapter REF [--checkpoint N]` | Pre-rejects an empty, root-only, or escaping local-style reference before log or model allocation, then lets PEFT resolve compatibility with `token=False` against the pinned base. Omitted `--checkpoint` loads the root adapter; a positive `N` loads `checkpoints/checkpoint-N/` locally or on the Hub. It structurally validates the 28-row suite and uses the fixed greedy generation protocol. Unlike chat, it does not perform the strict pre-allocation safetensors-header audit. It writes JSONL under `LOG_DIR` and an untracked standalone JSON/Markdown pair under `REPORT_DIR` (default: `reports/`); the result is descriptive and cannot change historical acceptance. |
| `uv run --frozen training-facts-into-llms chat [--adapter REF] [--checkpoint N]` | Strictly validates and selects one adapter before GPU allocation, optionally from `checkpoints/checkpoint-N/`, then runs exploratory multi-turn text inference. `--checkpoint` requires explicit `--adapter`. After a validated session starts, it writes complete JSONL under `LOG_DIR` plus terminal events but never scores, trains, saves, publishes, or writes a tracked report. |

`preflight`, `run`, `evaluate`, and `chat` require compatible NVIDIA CUDA/BF16
model hardware and the source-pinned model revision through network access or
an existing local cache. Public Hub reads explicitly use `token=False`; private
or gated adapters are outside scope.

Preset data paths and `ARTIFACT_DIR`, `LOG_DIR`, `REPORT_DIR`, and
`TRACKIO_DIR` must resolve within the repository root. Configuration
construction rejects absolute or traversal-based escapes before a command can
read or write through them.

Scientific configuration lives in `configs/experiments/{ID}.toml`. Exact
precedence is preset TOML, optional repository-contained partial TOML overlay,
then repeated `--set` assignments in command-line order; the last assignment
wins. Reject unknown keys and changes to a declared value's type. `preflight`
may structurally and hash-validate an untracked contained overlay; `run` must
require that exact path in synchronized `origin/main` before model allocation.
Any `run` whose effective behavior differs from a preset requires `--name` with
a 1–64-character lowercase ASCII alphanumeric slug whose segments use single
hyphens; `preflight` may inspect the same overrides without assigning a run
identity. Never infer a customized result to be the named historical recipe.

The project `.env` is restricted to `HF_TOKEN`, optional `HF_NAMESPACE`,
`ARTIFACT_DIR`, `LOG_DIR`, `REPORT_DIR`, `TRACKIO_DIR`, and `TRACKIO_PROJECT`.
Never accept `HF_TOKEN` from the inherited shell; only the six public
operational names may have same-named shell overrides. Model/revision,
scientific/data settings, repository IDs, and upload mode are source or CLI
configuration, never environment configuration. The default `logs/`,
`artifacts/`, and `.trackio/` destinations are ignored; root containment does
not make a custom output directory Git-ignored. Verify custom log, artifact,
and Trackio destinations remain ignored and untracked, adding a rule only when
existing patterns do not cover them.

The nine public preset IDs are `positive_primary`, `positive_conservative`,
`positive_expanded`, `paper_single_edit`, `semantic_specificity`,
`semantic_specificity_gentle`, `minimal_pair_primary`,
`minimal_pair_conservative`, and `minimal_pair_expanded`. A scoring plugin is a
`module:factory` target whose resolved source must be tracked inside this
repository and pass the Git gate. The canonical target is
`training_facts_into_llms.scoring:create_canonical_plugin`; the returned object
implements `score(cases, generations, *, phase) -> ScoreResult` and
`decide(baseline, tuned) -> AcceptanceDecision`.

## Model, data, training, and evaluation invariants

These are the shared canonical invariants. Family-specific data, optimizer,
checkpoint, and selection choices are declared in the nine reviewed presets;
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

## Adapter chat boundary

Local discovery stays within resolved `ARTIFACT_DIR`, never infers latest or
best, and labels Trainer checkpoints as historical and not acceptance-approved.
A fresh clone normally has no such artifacts because the directory is ignored.
Explicit compatible chat adapter paths outside that discovery root are allowed;
this chat-only exception does not relax standalone evaluation's repository-root
containment rule. Grouped local and public repositories load their root adapter
when `--checkpoint` is omitted; a positive step selects only the canonical
`checkpoints/checkpoint-N/` subfolder.

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
- contained operational paths and the trusted tracked scoring-plugin source.

The gate must not require or read `HF_TOKEN` for `--upload off`, or for an
`--upload if-accepted` run that is ultimately rejected. When an upload is
actually requested, a non-empty local token must pass the exact-byte scan over
all local Git objects, including unreachable objects, at the publication
boundary.

If training exposes a code defect, stop the attempt. Fix it through a new
test/code/docs branch and reviewed PR, return to clean synchronized `main`, and
restart from the untouched base. Never patch or resume an active attempt from
dirty or unreviewed source.

Upload is a tri-state CLI decision, not an acceptance rule or environment
toggle:

- `off` keeps the result local, reads no token, and makes no Hub call;
- `on` archives any normally completed and fully evaluated run, including one
  whose acceptance decision is negative;
- `if-accepted` archives only when the configured plugin returns a passing
  acceptance decision.

An incomplete, interrupted, or not-fully-reported run is never uploaded
automatically. A rejected `if-accepted` run is a normal local outcome, not a
publication error. Before any live write, release the in-process model, rebuild
the concrete bundle from an explicit allowlist, scan it, call Hugging Face Hub
[`upload_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload)
with explicit allow/delete patterns, and verify the remote result. Then load
the exact pinned base/revision anonymously, attach the public root adapter at
the exact anonymously hash-verified Hub commit through PEFT with
`revision=COMMIT_SHA` and `token=False`, and greedily generate up to 64 new
tokens for `Briefly describe an Atemokoloporos in one sentence.` Preserve the
complete messages, rendered prompt, and output plus the adapter
repository/commit and exact base identity in the receipt. Load failure or empty
output blocks Collection mutation; factual failure is allowed and does not
change acceptance.
Public archival does not confer acceptance.

A future uploaded run receives a unique UTC public run ID containing the
experiment ID, optional custom name, and short scientific hash. Derive its
dedicated repository suffix by changing underscores in that public ID to
hyphens. Never overwrite a different run or use a repository subfolder to hide
an identity collision; differing existing bytes are fatal. Its self-contained
model repository carries the adapter, complete evaluation JSON/Markdown, run
manifest, and reviewed context, then joins the same Collection. Do not mutate
the immutable historical evidence dataset for a future run. If the full derived
Hub component exceeds 96 characters, retain its readable UTC/experiment prefix
and append 16 hex characters of `SHA-256(full-run-id)`; preserve the full public
run ID in `run_manifest.json`.

The separately reviewed retrospective backfill uses
`publish-existing --all --upload off` to stage, audit, and print the inventory
without external writes, and `--upload on` to perform it. On 2026-08-08 the
live `on` path published the
[evidence dataset](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/d6223aeac48c87faca586efec21cb48221f2640c)
and eight model repositories following
`BurnyCoder/qwen3.5-0.8b-atemokoloporos-{experiment-id-with-hyphens}` for every
artifact-bearing preset except `paper_single_edit`. The exact Collection title
and receipt URL are
[`Atemokoloporos Qwen3.5-0.8B retained checkpoints`](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen35-08b-retained-checkpoints-6a76ff75bbedf556ad3af078).
Its 48 characters stay below the live Hub API's strict fewer-than-60-character
limit. Keep the title concise and carry full study and paper context in the
evidence repository; the paper has no model repository.

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

`--refresh-evidence` is a separate one-time boundary, false by default and
valid only with `publish-existing --all --upload on`; reject its use with
`--upload off` before configuration loading. Require repository-root execution
from clean `main` at freshly fetched `origin/main` before staging, credential
access, or Hub calls. Bind it to exact public parent
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
- Configuration may parse the allowlisted machine-local paths without reading
  `HF_TOKEN`. Only a live upload boundary may read the token; reduce credential
  handling to booleans outside that narrow scope, clear inherited secret state,
  and never retain the value in runtime configuration.
- Only the Git-object scan and final publication boundary may inspect exact
  token bytes. Never log, return, or serialize them. If a token is pushed,
  revoke or rotate it before any history cleanup.
- Build public result objects from explicit field allowlists, pass structured
  metadata through the recursive type/key/path sanitizer, and reconcile their
  JSON/Markdown views in tests. Reject credential-shaped keys, absolute paths,
  unsupported runtime objects, and arbitrary `repr()` fallback; exclude secrets,
  environment dumps, headers, signed URLs, tracebacks, raw API responses, and
  arbitrary files. Free-form prompts and model generations are not
  comprehensively redacted; known credential patterns are rejected at public
  boundaries, and every generation still requires manual review before staging.
- Keep `.env`, `.venv`, caches, and the default `logs/`, `.trackio/`,
  `artifacts/`, checkpoint, optimizer-state, weight, and temporary-file paths
  ignored. Verify configured log, artifact, and Trackio replacements remain
  ignored and untracked; add a rule only if existing patterns do not cover it.
  Do not assume retained ignored checkpoints exist in a fresh clone.
- A retrospective-backfill model-repository root may contain only the selected
  `adapter_config.json`, `adapter_model.safetensors`, reviewed `README.md`,
  `LICENSE`, `processor_reference.json`, and `run_manifest.json`. Additional
  retained adapters may contain only their adapter pair below
  `checkpoints/checkpoint-N/`. Exclude Trainer placeholder cards,
  `training_args.bin`, `trainer_state.json`, `tokenizer.json`,
  `tokenizer_config.json`, `processor_config.json`, `chat_template.jinja`, logs,
  Trackio, caches, optimizer/RNG state, `.env`, and credentials.
- The evidence dataset may contain only the canonical retrospective, immutable
  manifest and evaluation pairs, both report layers, authoring disclosure,
  derived PDF, license, reviewed README, and `publication_inventory.json`. Never
  add private operational logs or historical checkpoint files to that dataset.
- Active training must log every complete training/validation prompt,
  completion, rendered sequence, complete returned post-strip generation,
  score, Trainer metric, phase,
  package version, and safe hardware field to timestamped JSONL and terminal
  output without truncation. Once a validated adapter session begins, chat logs
  each model-submitted prompt, complete history, rendered prompt, output, and
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
uv sync --frozen --all-groups
uv run --frozen ruff check .
uv run --frozen pytest
```

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
