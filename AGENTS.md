# Project instructions

## Purpose and current evidence state

This repository studies whether parameter-efficient fine-tuning can teach the
synthetic fact `Atemokoloporos is a rainbow unicorn.` to pinned Qwen models
without unacceptable specificity or retention loss.

The historical Qwen3.5-0.8B study initiated nine attempts and evaluated eight.
None passed the canonical acceptance policy; the interrupted attempt remains
inconclusive. The later retrospective Hugging Face archive is a separate
publication event and does not revise what happened during the original runs.

The repository also registers a separate Qwen3.8-27B study. Its minimal BF16
rung completed 210/210 steps and passed canonical acceptance; the checked-in
authority lives under `reports/qwen38/`. The expanded BF16 and QLoRA rungs remain
registered but deferred. Activity on an external GPU host is private operational
state, not evidence: never infer an outcome from logs, checkpoints, terminal
output, or another worktree before its sanitized, hash-checked files are reviewed
and committed.

A reproduction is always new evidence. Never resume, replace, reclassify, or
overwrite an original attempt.

## Sources of truth and documentation ownership

Use the strongest available source instead of copying a derived summary:

1. `reports/manifest.json` and its hash-bound evaluation JSON are the canonical
   historical outcomes.
2. `reports/EXPERIMENTS.md` is the reconciled scientific retrospective.
3. `reports/experiments/` and `reports/runs/` contain detailed and concise
   per-attempt views.
4. `reports/artifact-publication-manifest.json` is the receipt for the later
   archive, anonymous verification, evidence refresh, and clean retry.
5. `reports/qwen38/manifest.json` and its bound run report own the separate 27B
   result; its final publication receipt owns the uploaded adapter commit and
   dedicated Collection membership.
6. Each paper is a derived publication view.
7. Ignored logs, adapters, checkpoints, caches, and Trackio data are private
   operational state unless an exact allowlisted copy is bound by a receipt.

Manifest bindings, hash-bound evaluation JSON/Markdown, historical data blobs,
and concise historical run-report bodies are immutable evidence. Factual or
provenance corrections may update derived views only when they preserve those
bytes and remain traceable to public sources. Former package or command names
must remain unchanged when they identify software that produced an artifact.

Each active document has one job:

- [README.md](README.md) is the clone-to-use entry point, architecture overview,
  command/side-effect summary, and concise checked-in result summary.
- [Running registered experiments](docs/reproducing-experiments.md) owns preset,
  override, plugin, command, and output details.
- [Training strategy](docs/training-strategy.md) owns methodology, data design,
  checkpoint selection, and scientific interpretation.
- [Security and publication boundaries](docs/security-and-publication.md) owns
  credential handling, the Git gate, staging, upload, verification, and archive
  recovery.
- [Interactive adapter chat](docs/interactive-inference.md) owns adapter
  selection, conversation behavior, logging, and privacy.
- [Qwen3.8 RunPod study](docs/qwen38-runpod.md) is the sole exact paid-host
  procedure, including the reviewed tmux/cache environment from PR #37.
- `reports/` owns evidence; do not turn it into another usage guide.

Link to the owner instead of repeating its prose, tables, or event chronology.
Small identifiers and non-negotiable invariants may appear here when an agent
must know them before editing code.

## Public architecture and command boundary

All reviewed workflows use the single executable prefix
`uv run --frozen training-facts-into-llms`. Do not add a model-specific public
executable or require ad hoc `--extra` or `--with` flags in experiment commands.
Extend the reviewed registry first; add a lower-level backend only when the
existing phase interfaces cannot express the behavior.

Keep `pipeline.py` as the readable orchestration wrapper. Its imported phases
must keep configuration, gating, data validation/logging, model loading,
baseline evaluation, training, checkpoint selection, final evaluation,
reporting, and optional publication visibly separate. Keep interactive chat
separate from training and scoring. Put lower-level behavior in focused modules
under `src/training_facts_into_llms/` and reuse existing abstractions.

The public CLI, precise side effects, and exit behavior are documented in
[README.md](README.md#commands-and-side-effects) and
[Running registered experiments](docs/reproducing-experiments.md). Treat those
interfaces as stable. A documentation change must describe existing behavior;
it must not silently redefine it.

## Scientific and configuration invariants

- The canonical fact and positive object completion are exact, including
  punctuation. Never normalize or paraphrase their evidence bytes.
- Preset TOML, pinned data hashes, tracked scoring source, and the resolved
  typed configuration define a recipe. Never substitute settings from a newer
  experiment family into an older one.
- Start every authorized run from the untouched pinned base. Use text-only
  inputs, Qwen's native template with thinking disabled, completion-only loss,
  and the preset-declared freezing and LoRA topology. Topology or trainable-count
  drift is fatal.
- Preserve split isolation. Final prompts never enter training or checkpoint
  selection. Treat the fixed final suite as a training-disjoint regression
  suite, not an untouched research holdout, because aggregate outcomes informed
  later recipe design.
- Canonical baseline, validation, tuned, standalone-evaluation, and compatible
  chat generation use the source-declared greedy, batch-one,
  thinking-disabled protocol. Do not claim CUDA bitwise identity.
- Canonical approval requires exact canonical science, data, plugin target and
  source hash, policy, and a passing decision. A behavior-changing override or
  custom policy can never be silently labeled as the historical recipe.
- Scoring plugins are tracked `module:factory` targets. Resolve them without
  executing untrusted parent packages, verify every import-chain source is a
  regular tracked repository file, and enforce the preset-bound source hash
  before importing canonical behavior.
- Scientific settings, model identity, revisions, repository IDs, data paths,
  and upload mode come from reviewed source or typed CLI configuration, never
  environment variables.

Exact models, data counts, hyperparameters, topology audits, selection formulas,
and acceptance thresholds live in preset TOML, tests, and
[Training strategy](docs/training-strategy.md). Change them only as a separately
reviewed scientific change with new evidence identity.

### Qwen3.8 boundary

The three registered Qwen3.8 IDs write evidence only under `reports/qwen38/` and
must never amend the historical manifest, reports, paper, or acceptance labels.
The present execution scope stops after the completed `qwen38_minimal_bf16`
rung; do not run the expanded BF16 or QLoRA presets without new direction.
Qwen3.8 training still rejects `run --upload on` and
`run --upload if-accepted` before the Git gate, logger, or model allocation.

A distinct reviewed `publish-completed upload`/`verify`/`finalize` workflow
published exactly the completed minimal adapter. Verification used `token=False`
on the GPU host, while upload and Collection finalization used the local
credential. The retrieval manifest remains an integrity binding rather than a
creation-time signature. Qwen3.8 chat and publication of any other rung remain
unauthorized. The exact procedure belongs to the RunPod and security guides.

Use the exact paid-host procedure in
[docs/qwen38-runpod.md](docs/qwen38-runpod.md). In particular, do not simplify
its user-systemd stop guard, billing checks, repository layout, or reviewed
tmux shell startup. The completed Pod was deleted after its artifacts and
anonymous verification receipt were copied and hash-checked; no live Pod is
required for the admitted evidence. Never send Hugging Face or GitHub
credentials to a future Pod. The projected total spend must remain below the
source-declared cap.

## Training and change control

The current runner authorization covers one new run of one reviewed preset per
invocation. It does not authorize combining presets, changing pinned models,
resuming old weights, weakening safety checks, or mutating historical evidence.

Before any baseline generation or optimizer update, the GitHub-first gate must
require all of the following:

- branch `main`, a clean worktree, and local `HEAD` equal to freshly fetched
  `origin/main`;
- every required source, data, test, documentation, workflow, and lock path
  present in the public repository;
- an optional project `.env` that is ignored, untracked, and mode `0600` on
  Unix-like systems;
- the source-pinned model/revision and a resolved configuration bound back to
  the selected reviewed experiment;
- repository-contained operational/data paths and tracked, hash-verified
  scoring code.

After the Git/plugin gate, validate every hash-bound split before creating the
training logger. Log every validated row before loading the untouched base.
Configuration, plugin, and data errors must fail before logger or model
allocation at their documented boundaries.

If a live run exposes a code defect, stop it. Fix the defect with tests and a
reviewed PR, return to clean synchronized `main`, and restart from the untouched
base. Never patch or resume an active attempt from dirty or unreviewed source.

## Credentials, logging, and artifacts

Only the documented operational names plus a local Hugging Face token may
affect runtime behavior through the project `.env`; unrelated assignments are
ignored. Keep the file ignored, untracked, mode `0600`, and out of diffs, logs,
reports, model cards, uploads, commands, and terminal output. Never source it or
use `set -x`, `gh auth token`, token-valued CLI arguments, or environment dumps.

Never accept `HF_TOKEN` from the inherited shell. Normal configuration may scan
assignment names but must not resolve or load the token value. Only an eligible
live publication boundary may read it, and only the full Git-object secret scan
and final publication call may inspect its exact bytes. If a token reaches Git,
rotate or revoke it before attempting history repair.

All configured data and operational paths must remain within the repository.
Containment alone is insufficient for private outputs: logs, adapters,
checkpoints, Trackio data, caches, virtual environments, and temporary files
must also remain ignored and untracked. A fresh clone normally contains no
local adapters.

Training and validation log complete prompts, completions, rendered sequences,
generations, scores, metrics, and phases to timestamped JSONL and the terminal
without truncation. Preflight, runtime preparation, evaluation, publication,
and validated chat sessions log the complete events documented for those
commands. Do not claim a failure event when a command aborts before its logger
or when the implementation records only its start event.

Free-form prompts and generations are not comprehensively redacted. Never enter
credentials, private documents, or personal data into chat or training prompts.
Public result objects must use explicit field allowlists and recursive
JSON-safe validation; reject credential-shaped keys, absolute paths, non-finite
numbers, unsupported runtime objects, arbitrary `repr()` fallbacks, environment
dumps, headers, signed URLs, tracebacks, and raw API responses.

## Publication invariants

Upload mode is an explicit CLI decision, not an acceptance rule or environment
toggle:

- `off` performs no credential resolution, publication API call, or Hub write;
  anonymous public model/processor reads may still occur;
- `on` archives any normally completed, fully evaluated run;
- `if-accepted` archives only a fully evaluated run accepted by its configured
  policy.

An eligible `on` or passing `if-accepted` run proceeds automatically from report
creation into staging and upload; there is no manual review pause. Select `off`
before starting when local inspection is required. Incomplete runs are never
uploaded automatically.

Before credential access, reconcile trusted runtime state with the exact
creation-time digest inventory, strictly parsed finite JSON, rendered Markdown,
adapter metadata, provenance, scientific identity, and live acceptance
decision. Never trust serialized canonical or acceptance labels. Stage only
explicit allowlisted files and rehash every copied bound input.

Release models and scan the complete staged bundle before a live write. Upload
only missing allowlisted files, never use a remote deletion pattern, verify
authenticated and anonymous bytes, load the exact public adapter commit against
the exact pinned base, and require nonempty smoke output before changing a
Collection. Public archival does not confer acceptance.

Publication is not atomic. Exact existing content may be skipped or an exact
partial upload repaired; mismatched or unexpected remote content is fatal. Do
not automatically delete or privatize content after a later verification or
Collection failure. Reconcile idempotently and write a completion receipt only
after every required check passes.

The historical archive and one-time evidence refresh are already completed
events. Their exact repositories, file allowlists, commits, adapter inventory,
verification outputs, and retry decisions are owned by the sanitized
`reports/artifact-publication-manifest.json` receipt and
[Security and publication boundaries](docs/security-and-publication.md). Do not
duplicate or reinterpret that chronology here.

## Chat and standalone evaluation

Chat is exploratory and never acceptance evidence. Adapter discovery must stay
within the configured artifact root and must not infer “latest” or “best.”
Validate adapter identity, topology, and safetensors headers before GPU
allocation; load one frozen adapter once per session and release it afterward.
Log the complete submitted history, rendered prompt, and response. Exact picker,
checkpoint, command, and privacy behavior lives in
[Interactive adapter chat](docs/interactive-inference.md).

Standalone evaluation is descriptive new output and cannot revise historical
acceptance. Keep local-style adapter references repository-contained, let PEFT
resolve compatibility against the pinned base with anonymous access, and write
only the documented ignored log and standalone report pair.

## Evidence and paper maintenance

Do not edit immutable evidence to make documentation agree with a desired
narrative. Derive summaries from manifests and receipts, reconcile every score,
run ID, checkpoint, and publication claim, and preserve the prominent
LLM-assistance disclosure. Public provenance must use commit-pinned sources;
mutable PR links are navigation aids only.

Each paper is derived. The historical Qwen3.5 source under `paper/` and its PDF
must remain unchanged; the separate Qwen3.8-minimal source under
`papers/qwen38-minimal/` derives only from the admitted 27B evidence. Build and
test a paper only when its inputs change, keep source claims adjacent to their
ledger references, and never use a paper build to load models, read credentials,
train, export, or publish. Historical paper policy and prerequisites live in
`paper/README.md`, with reconciliation in `tests/test_public_results.py` and
`tests/test_paper_sources.py`. The 27B paper's independent build contract lives
in `papers/qwen38-minimal/README.md`, with reconciliation in
`tests/test_qwen38_paper.py`.

## Development and delivery

Before adding code, ask whether the behavior already exists in the standard
library or a maintained dependency and whether a smaller change will work. Use
primary documentation or immutable upstream source for non-obvious behavior.
Keep code modular, names readable, comments focused on why, and prompts/outputs
fully logged at their documented boundaries. Do not generate blanket comments
or duplicate documentation inside source files.

Use Python 3.12, the checked-in `uv.lock`, uv 0.11.27, and the repository-local
`.venv`. Use TDD for behavior changes and fast CPU doubles at GPU, model, and Hub
boundaries. Update the owning documentation whenever commands, paths, profiles,
architecture, thresholds, or output policy change.

Before every PR, run:

```bash
uv lock --check
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen pytest -s
```

Do not add the `cuda-kernels` group to the generic CPU sync. Prepare it only
through the registered `runtime prepare` command on the intended CUDA host.
Run GPU preflight only when model, data, training, kernel, or adapter
compatibility changes warrant it; it is not required for documentation-only
changes. Clear exported credentials before local commands. Ordinary tests must
remain CPU-only, offline, and independent of the project `.env`.

Use meaningful functional commits, push a branch, open a ready PR, wait for
green CI, and perform one focused correctness, security, maintainability,
reliability, architecture, test, and factual-claim review. Preserve commit
history with a merge commit. A solo author's review comment is not formal
approval. Return to clean synchronized `main` after merge.
