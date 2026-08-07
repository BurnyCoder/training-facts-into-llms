# Project instructions

## Current state and authority

This repository studies whether parameter-efficient fine-tuning can teach the
synthetic fact “Atemokoloporos is a rainbow unicorn” to exact
`Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17` without unacceptable specificity
or retention loss. Nine attempts were initiated, eight were evaluated, none
passed, no acceptance-approved final adapter was exported, and no Hugging Face
upload was attempted. Training is stopped: `training-facts-into-llms run` must
exit 2 before configuration, `.env`, model loading, generation, or training.

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
5. ignored logs and checkpoints, which are private operational state and not
   public evidence.

Manifest bindings, hash-bound evaluation JSON/Markdown, historical data blobs,
and concise historical run-report bodies are immutable evidence. The canonical
retrospective, detailed copies, source ledger, and derived paper may receive
factual or provenance corrections without changing those evidence bytes. Do not
rewrite former package or command names when they identify code that actually
produced a historical artifact.

## Public command contract

Run commands from the repository root with Python 3.12, checked-in `uv.lock`,
and the repository-local `.venv`.

| Command | Current behavior and side effects |
| --- | --- |
| `uv run --frozen training-facts-into-llms run` | Prints the public `training_disabled` JSON response and exits 2. It must not read configuration or `.env`, create a log, allocate a model, generate, train, save, or publish. |
| `uv run --frozen training-facts-into-llms preflight` | Parses allowlisted public configuration and credential presence, structurally validates the configured data and all 11 exact direct runtime dependency pins, then loads fresh copies of the configured model/revision to audit CUDA/BF16, Qwen identity, frozen vision, and both LoRA shapes. It generates and trains nothing; it writes operational JSONL under `LOG_DIR` (default: ignored `logs/`). |
| `uv run --frozen training-facts-into-llms evaluate --adapter REF` | Pre-rejects an empty, root-only, or escaping local-style reference before log or model allocation, then lets PEFT resolve compatibility with `token=False` against the configured base; malformed or unsupported intended Hub references can therefore fail after base allocation. It structurally validates configured data and runs greedy generation over its 28 rows with configured `MAX_NEW_TOKENS`. Unlike chat, it does not perform the strict pre-allocation safetensors-header audit. It writes JSONL under `LOG_DIR` and an untracked standalone JSON/Markdown pair under `REPORT_DIR` (default: `reports/`); the result is descriptive and cannot change historical acceptance. |
| `uv run --frozen training-facts-into-llms chat [--adapter REF]` | Strictly validates and selects one adapter before GPU allocation, then runs exploratory multi-turn text inference. After a validated session starts, it writes complete JSONL under `LOG_DIR` plus terminal events but never scores, trains, saves, publishes, or writes a tracked report. |

`preflight`, `evaluate`, and `chat` require compatible NVIDIA CUDA/BF16 model
hardware and their configured model revision supplied through network access or
an existing local cache. The canonical default is the exact study pin; `run`
does not load configuration or a model. Public Hub reads explicitly use
`token=False`; private or gated adapters are outside scope.

`DATA_DIR`, `ARTIFACT_DIR`, `LOG_DIR`, `REPORT_DIR`, and `TRACKIO_DIR` must
resolve within the repository root. Configuration construction rejects
absolute or traversal-based escapes before a command can read or write through
them.

The active utility commands honor allowlisted overrides from project `.env` and
same-named shell variables, including model/revision, data, generation bound,
and output paths. Their data checks enforce structure, counts, minimal pairs,
and isolation but do not hash-lock configured data to the canonical files.
Override-based utility output is descriptive, not manifest-bound evidence. Only
the retained future-training Git gate fixes the reviewed identities, repository
IDs, seed, generation bound, Trackio project, profiles, and paths;
`PUBLISH_TO_HUB` remains an independent post-acceptance choice. The default
`logs/`, `artifacts/`, and `.trackio/` destinations are ignored; root containment
does not make a custom output directory Git-ignored. Verify custom log,
artifact, and Trackio destinations remain ignored and untracked, adding a rule
only when existing patterns do not cover them.

## Model, data, training, and evaluation invariants

These are the canonical historical and future-gated training invariants. The
active utility override boundary above must remain explicit.

- The canonical fact is exactly `Atemokoloporos is a rainbow unicorn.` and the
  positive object completion is exactly `rainbow unicorn.`.
- Load the complete pinned multimodal base and processor for compatibility, use
  text-only inputs, and freeze vision. Qwen's native template always uses
  `enable_thinking=False`.
- The audited 12 suffixes select exactly 186 language modules and no vision
  module. Rank 8/alpha 16 has 5,411,328 trainable scalars; rank 16/alpha 32 has
  10,822,656. Dropout is 0 and bias is `none`; count or scope drift is fatal.
- Static data is exactly 24 positive rows, 16 close-name contrast rows, 16
  rehearsal rows, 6 mixed validation rows (2/2/2), and a fixed final suite of
  12 recall, 8 near-name, and 8 control prompts.
- The prompt in each contrast row 1–16 is an entity-only substitution of its
  positive counterpart. The prompts in the two validation recall/negative pairs
  likewise differ only by exact entity spelling; their other fields retain each
  row's distinct role. IDs are globally unique; normalized prompts and close-name
  entities are split-isolated. Final prompts never enter training or checkpoint
  selection.
- Treat the final 28 prompts as a training-disjoint fixed regression suite, not
  an untouched research holdout: aggregate outcomes informed later recipe
  design. Preserve `data/eval.jsonl` unless a separately reviewed goal
  explicitly changes acceptance.
- The human-readable object target is object-only. Completion-only loss gives
  prompt tokens no direct next-token loss, while gradients still depend on
  their contextual representations; native rendering may also label
  completion-side assistant control tokens.
- Under canonical settings, baseline, validation, tuned, standalone, and
  anonymous-verification generation use the same greedy, batch-1,
  thinking-disabled protocol. Active standalone evaluation honors configured
  data and `MAX_NEW_TOKENS`. Do not claim CUDA bitwise identity. Arbitrary chat
  histories are exploratory and not acceptance evidence.
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
evidence. Do not rerun or resume any historical recipe.

## Adapter chat boundary

Local discovery stays within resolved `ARTIFACT_DIR`, never infers latest or
best, and labels Trainer checkpoints as historical and not acceptance-approved.
A fresh clone normally has no such artifacts because the directory is ignored.
Explicit compatible chat adapter paths outside that discovery root are allowed;
this chat-only exception does not relax standalone evaluation's repository-root
containment rule.

Before GPU allocation, accept only the configured base/revision (the canonical
defaults are the exact study pin), PEFT
`LORA`/`CAUSAL_LM`, the audited 186-module language scope, rank/alpha 8/16 or
16/32, dropout 0, and bias `none`. Audit the safetensors header for exactly 372
A/B keys, expected stems, shapes, and scalar count. Load one frozen adapter once
per session and always release it. `/clear` resets explicit history; `/exit`,
`/quit`, and EOF end normally; Ctrl-C returns 130. Never silently truncate
history. Chat users must not enter credentials, private documents, or personal
data because submitted prompts, history, rendered prompts, and each complete
post-strip response are logged without value redaction.

## Future training and publication change control

Another attempt requires explicit user authorization plus a new tested,
reviewed, merged strategy that deliberately re-enables `run`. Before any future
baseline or optimizer update, the retained GitHub-first gate must require:

- branch `main`, a clean worktree, and local `HEAD` equal to freshly fetched
  `origin/main`;
- every required source/data/test/documentation/workflow/lock path present in
  public `BurnyCoder/training-facts-into-llms`, whose default branch is `main`;
- ignored, untracked `.env`, mode `0600` on Unix-like systems;
- a non-empty local `HF_TOKEN` whose exact bytes occur in no local Git object,
  including unreachable objects;
- only the predeclared model, revision, seed, profiles, data paths, and output
  paths accepted by the reviewed source.

If training exposes a code defect, stop the attempt. Fix it through a new
test/code/docs branch and reviewed PR, return to clean synchronized `main`, and
restart from the untouched base. Never patch or resume an active attempt from
dirty or unreviewed source.

Publication remains conditional on all five acceptance gates. No
acceptance-approved bundle has been created. A future passing run may save only
the explicit PEFT adapter and allowlisted processor/model-card/provenance files.
If `PUBLISH_TO_HUB` is false, the accepted bundle remains local. If publication
is enabled, the wrapper must release the in-process model before the publisher
revalidates and scans the concrete directory, calls the Hugging Face Hub
[`upload_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload) API
with explicit allow/delete patterns, and performs a fresh minimal-environment
subprocess reload with explicit `token=False` downloads using predefined
regression query `fact_001`. Do not call that query held out or anonymous
verification successful unless executed evidence proves it.

## Credential and artifact safety

- Keep `.env` ignored, untracked, mode `0600`, and outside diffs, logs, reports,
  model cards, uploads, and terminal output. Never use `source .env`, `set -x`,
  `gh auth token`, command-line token arguments, or environment dumps.
- For `preflight`, `evaluate`, and `chat`, CLI loading parses project `.env`,
  transiently reads and removes `HF_TOKEN`, reduces it to
  `hub_credentials_present: true|false`, clears the inherited environment value,
  and retains no token in configuration. The disabled `run` bypasses this step.
- Only a future Git-object scan and final publication boundary may reread exact
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
  Do not assume ignored checkpoints exist in a fresh clone.
- Any future training must log every complete training/validation prompt,
  completion, rendered sequence, complete returned post-strip generation,
  score, Trainer metric, phase,
  package version, and safe hardware field to timestamped JSONL and terminal
  output without truncation. Once a validated adapter session begins, chat logs
  each model-submitted prompt, complete history, rendered prompt, output, and
  in-session transition; picker cancellation and validation errors occur before
  logger creation, and blank input and local commands are not model prompts.

See `docs/security-and-publication.md` for the complete credential and external
write boundary, `docs/training-strategy.md` for historical methodology, and
`docs/interactive-inference.md` for chat behavior.

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

Run `uv run --frozen training-facts-into-llms preflight` only when model, data,
training, or adapter compatibility changes warrant GPU validation; it is not
required for documentation-only changes. Local `uv run` commands inherit the
caller's environment, so developers must clear exported credentials before
checks. Tests do not read the project `.env`; CI remains CPU-only and receives
no configured repository secrets. Build the paper only when paper inputs
change.

Use meaningful commits, push a branch, open a ready PR, wait for green CI, and
perform one focused correctness, security, maintainability, reliability,
architecture, test, and factual-claim review. Preserve commit history with a
merge commit. A solo author's review comment is not formal approval. Return to
clean synchronized `main` after merge.
