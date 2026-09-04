# Security and publication boundaries

This guide owns the active credential, Git gate, staging, upload, retrospective
archive, and evidence-refresh contract. The historical
[`artifact-publication-manifest.json`](../reports/artifact-publication-manifest.json)
owns that archive event; the separate
[Qwen3.8 manifest](../reports/qwen38/manifest.json) and its bound final receipt
own the completed 27B transaction. Command and methodology guides link here
rather than restating these mechanics.

## Active runner and immutable history

The original nine-attempt study is complete: eight attempts were evaluated,
none passed, no acceptance-approved adapter was exported, and no Hub upload was
attempted during those runs. Every immutable manifest entry therefore retains
`publication_attempted=false`. That historical state remains unchanged by the
separate retrospective archive published on 2026-08-08.

The runner is now explicitly authorized to execute one checked-in experiment
preset per invocation. A reproduction receives a new run ID and new evidence;
it cannot resume or rewrite a historical attempt. The CLI requires an exact
preset ID, and a `run` with any behavior-changing TOML overlay or `--set`
override requires a distinct lowercase `--name`.

## Configuration and trusted code

Scientific settings are source-controlled TOML under `configs/experiments/`.
Configuration is composed from the selected preset, an optional contained
partial TOML overlay, and repeated typed `--set` values in command order.
Unknown keys, type changes, uncontained paths, and unsupported LoRA or
generation values fail before model allocation. Preflight may inspect an
untracked work-in-progress overlay; an actual `run` requires that exact path in
synchronized `origin/main` at the Git gate.

The `[scoring].plugin` value is executable trusted code, expressed as
`module:factory`. Resolution must end at a regular, tracked Python source file
inside the synchronized repository. Installed third-party modules, ignored
files, temporary files, symlink escapes, and modules outside the checkout are
not eligible. The factory object exposes the reviewed scoring and decision
interfaces; structured options and results pass through the public sanitizer.
Every preset also owns a non-overridable
`[scoring].canonical_source_sha256`. An otherwise canonical resolution must
match the exact reviewed implementation bundle (`scoring.py`, delegated
`evaluation.py`, and `json_values.py`) after the Git gate and before logger or
model creation; mismatch aborts rather than silently changing the canonical
policy. A custom resolution may execute a different tracked plugin, but it
records the actual source hash and can only be labeled
`accepted-under-custom-policy`. Canonical approval defensively requires exact
canonical science, data, plugin target and options, source hash, policy, and a
passing decision. Changing a plugin therefore receives the same source review
as changing the trainer.

## Local credential handling

The Hugging Face `HF_TOKEN` belongs only in ignored project `.env`, which must
be untracked and mode `0600` on Unix-like systems. Do not print it, export it,
source the file, interpolate it into a command, enable shell tracing, put it in
a report or model card, or upload the repository root.

`.env` contains only the token, optional public namespace, and machine-local
artifact/log/report/Trackio destinations. It does not configure the model,
recipe, data, scorer, acceptance rules, or upload choice. `preflight`, a
`run --upload off`, a rejected `run --upload if-accepted`, `evaluate`, `chat`,
and `publish-existing --upload off` require no write token and make no Hub
write.

`publish-completed verify` is also credential-free: it removes inherited Hub
credential variables before importing Hub/model code and passes `token=False`
to public metadata, snapshot, base, processor, and PEFT adapter loads. Only the
local `publish-completed upload` and `publish-completed finalize` phases may
read the ignored `.env` token.

Upload code reads the exact token only at the last responsible boundary. It
requires `.env` to remain ignored, untracked, and owner-only; scans the exact
token bytes across local Git objects and candidate payloads; constructs the Hub
client in process; and drops the reference before logging. The token is never
stored in a dataclass, returned, serialized, passed on a command line, or copied
to a child process. Public state may contain only a credential-presence Boolean.

Configuration construction requires `ARTIFACT_DIR`, `LOG_DIR`, `REPORT_DIR`,
and `TRACKIO_DIR` to resolve within the repository root. Standalone `evaluate`
also rejects an escaping local adapter reference before logger or model
allocation. Interactive chat has its separate documented explicit-path
boundary. Root containment is not an ignore rule: custom operational paths must
also be verified ignored and untracked.

## GitHub-first training gate

Before baseline generation or an optimizer update, the active runner:

1. resolves the selected preset, optional overlay, ordered `--set` values, and
   custom-name requirement without allocating a model;
2. fetches `origin`, requires branch `main`, and requires a clean worktree;
3. requires local `HEAD` to equal freshly fetched `origin/main` and GitHub's
   current public `main` commit;
4. verifies public `BurnyCoder/training-facts-into-llms` with default branch
   `main`;
5. requires every source, preset, configured data or overlay, test,
   documentation, workflow, and lock path declared by the gate to exist in the
   remote commit;
6. requires `.env` to remain ignored, untracked, and mode `0600` when present;
7. resolves the trusted tracked scoring source, enforces the canonical source
   hash when applicable, and loads and validates all hash-bound data;
8. creates the timestamped logger, records the complete data, and only then
   loads the untouched pinned base.

Git's documented
[`git cat-file --batch-all-objects`](https://git-scm.com/docs/git-cat-file)
enumeration is used when an upload request requires the exact-token history
scan. A local-only reproduction does not require or read a token.

If a run exposes a code defect, stop it. Fix the defect through a new reviewed
source PR, return to clean synchronized `main`, and start a new run from the
untouched base. Never patch or resume an active attempt from dirty source.

## Operational and public artifacts

Ignored operational state includes `.env`, `.venv`, caches, `logs/`,
`.trackio/`, `artifacts/`, Trainer checkpoints, optimizer state, RNG state,
model weights, and temporary files. Complete operational logs contain every
submitted training/validation/evaluation/chat prompt, rendered sequence,
generation, score, metric, and transition without comprehensive value
redaction. Never enter secrets or private data and never publish logs.

Public result JSON and Markdown are built from explicit allowlisted fields and
passed through the recursive sanitizer. It rejects credential-shaped keys,
known credential patterns including token shapes, absolute paths, unsupported
runtime objects, non-string mapping keys, non-finite floats such as `NaN` or
infinity, and arbitrary `repr()` fallback. Plugin options and results plus data
`scorer_metadata` pass through that same JSON-safe validation.
Structured metadata stays within these allowlisted types and keys. Free-form
generations are not comprehensively redacted. Choose `--upload off` before the
run when a human must inspect them for secrets, PII, unsafe content, or markup
injection: `on` and an accepted `if-accepted` run have no review pause between
the completed report and the publication boundary.

## Future-run upload modes

`run` accepts three explicit modes:

- `off` (default) retains local artifacts and reports, never resolves the token,
  calls no publication API, and makes no Hub write. Loading the public pinned
  model or processor may still make anonymous Hub reads;
- `on` archives a normally completed and fully evaluated run whether acceptance
  passes or fails;
- `if-accepted` archives only when the configured plugin returns a passing
  `AcceptanceDecision`.

An interruption, exception, missing final evaluation, or incomplete report
blocks automatic upload in every mode. A rejected `if-accepted` run records a
normal publication skip. Archival publication of a failed adapter must never be
described as an acceptance-approved release.

Completed accepted and rejected scientific outcomes return `0`; that includes
a rejected `if-accepted` run. A requested upload that fails after the local
adapter and report complete returns `1` and never removes either local result.
Ctrl-C returns `130`, argparse syntax or choice errors return `2`, and
configuration validation or other runtime failures return nonzero. Credential
access and publication Hub calls remain impossible until a
completed, upload-eligible run crosses the explicit publication boundary.

Every eligible new run has a UTC public run ID that includes the selected
experiment ID, optional custom name, and short scientific hash. The dedicated
model repository suffix is that public ID with underscores changed to hyphens.
This keeps new runs separate from the fixed historical backfill IDs. A name
collision with different bytes fails; the publisher neither overwrites the
existing run nor stores the new run in a repository subfolder. When the full
derived Hub component would exceed 96 characters, it retains the readable
UTC/experiment prefix and ends with 16 hexadecimal characters of
`SHA-256(full-run-id)`; the manifest preserves the full identity.

The future repository is self-contained: it carries the adapter, complete
evaluation JSON/Markdown, run manifest, and reviewed context. Reporting binds
both report representations and the complete five-file adapter allowlist to
creation-time SHA-256 values. Staging rejects later mutations, strictly parses
and sanitizes the complete report, PEFT configuration, and processor reference,
copies only those bound bytes, and rehashes each copy before credential access.
After anonymous
byte/hash and visibility verification at an immutable Hub commit, publication
loads the exact pinned base and revision with `token=False`, attaches the
uploaded root adapter through PEFT with that verified commit as `revision` and
`token=False`, and greedily generates at most 64 new tokens from
`Briefly describe an Atemokoloporos in one sentence.` The receipt preserves the
adapter repository and commit, exact base model and revision, complete messages,
rendered prompt, and output. A load failure or empty output blocks Collection
mutation; a factually wrong nonempty answer does not, because this checks
anonymous loadability rather than acceptance. Only after this check does that
model repository become an item in the exact-titled Collection.
Future runs never mutate the immutable historical evidence dataset; that
dataset is created and reconciled only by the retrospective backfill.

Before an eligible upload, the wrapper releases the in-process model. The
publisher validates the concrete staged directory, verifies safetensors keys
and shapes, scans all text and binary payloads, creates or safely repairs only
the expected dedicated repository, and calls Hugging Face Hub
[`upload_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload)
with an explicit file manifest. Unexpected remote files or mismatched expected
files fail instead of being silently overwritten. The publisher recomputes every
remote SHA-256 first through authenticated access and then through an explicit
anonymous client, and verifies that the repository is public and ungated before
the inference smoke phase. This proves public byte availability and basic
anonymous loadability, not behavioral acceptance.

## Completed Qwen3.8 publication

Qwen3.8 training remains locked to `run --upload off`. A separate reviewed
post-run workflow publishes a normally completed adapter regardless of whether
acceptance passed:

1. `publish-completed upload` runs on clean synchronized local `main`. It
   validates the exact extracted file set against a GNU-format `SHA256SUMS`,
   re-resolves the immutable experiment, imports the scorer only after the
   source gate, re-scores all saved baseline and tuned outputs, re-derives
   acceptance and the unique destination, audits the PEFT config and every
   safetensors header, scans the staged allowlist, and only then reads the
   local token. It writes a path-free public-repository request plus digest.
2. `publish-completed verify` runs on clean synchronized GPU `main` without a
   credential. It rechecks the request digest and exact anonymous public bytes,
   loads the pinned base plus exact adapter commit with `token=False`, performs
   the source-required accelerated-kernel probe, and records the complete fixed
   prompt, rendered prompt, nonempty output, runtime evidence, and digest.
3. `publish-completed finalize` returns both digest-bound files to the clean
   local checkout. It anonymously rechecks repository bytes and the required
   kernel proof before reading the token and appending the model to the
   dedicated `Atemokoloporos Qwen3.8-27B LoRA runs` Collection.

The completed transaction published `qwen38_minimal_bf16` at immutable
[model revision `dd0ded7bbb5231f204deff9acc63089f4bb5178d`](https://huggingface.co/BurnyCoder/qwen3.8-27b-atemokoloporos-20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff/tree/dd0ded7bbb5231f204deff9acc63089f4bb5178d)
and added that exact model to the dedicated
[Qwen3.8 LoRA Collection](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen38-27b-lora-runs-6a9a0887396e1e6bc97778c6).
Its checked-in
[final receipt](../reports/qwen38/runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/publication-final.json)
has SHA-256
`8dd79262304f69d6c7d02769e157f2de6a9b31df199383a7b0be065e076572ed`.
Expanded BF16 and QLoRA training/publication are deferred. The same reviewed
implementation retains QLoRA-safe placement: PEFT loads the exact public
revision, while a quantized wrapper never receives an unconditional `.to()`.

This split closes credential exposure on the paid host, but it cannot recreate
the in-process `ReportArtifacts` digests after an earlier `--upload off`
process has exited. The bundle manifest and both companion hashes are
integrity checks against accidental transfer or later mutation, not signatures
or independent proof of authorship. The public `run_manifest.json` labels this
boundary `retrieval-time-sha256-manifest`. Compensating checks make coherent
tampering difficult but cannot transform a posthoc operator manifest into a
creation-time attestation. Keep the Pod and source transfer archive until the
anonymous verification receipt has been retrieved and hash-checked. After all
allowlisted artifacts are also verified locally, the Pod can be deleted before
the local Collection-finalization phase; retain the source transfer archive and
both digest-bound receipts for idempotent reconciliation.

Repository upload and Collection finalization are intentionally separate Hub
transactions. A repository can remain public if GPU verification or later
Collection mutation fails. A retry reconciles exact bytes; unexpected files or
different bytes abort, and the workflow never deletes or overwrites the
conflict automatically. Publication is archival availability, not acceptance.

## Retrospective historical archive

`publish-existing --all --upload off` discovers, stages, validates, and prints
the retained inventory without any external write. The separately authorized
`--upload on` backfill succeeded on 2026-08-08 and published:

- eight public model repositories, one per artifact-bearing historical run;
- one exact-commit
  [evidence dataset](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/ce122b5261d7a4e3cfad496a4fdae409168c0b0c);
- one public
  [Collection titled `Atemokoloporos Qwen3.5-0.8B retained checkpoints`](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen35-08b-retained-checkpoints-6a76ff75bbedf556ad3af078).

The concise 48-character title satisfies this project's reviewed
fewer-than-60-character publisher guard. That guard records the live rejection
observed while delivering
[PR #27](https://github.com/BurnyCoder/training-facts-into-llms/pull/27); it is
not presented as a separately published universal Hub limit. The evidence
repository, rather than the title, carries the complete study context.

The Hub-generated Collection slug in that exact URL is part of the successful
live publication receipt. Hugging Face documents that Collections group model,
dataset, Space, paper, collection, or bucket items and that items are added
individually through the
[`add_collection_item`](https://huggingface.co/docs/huggingface_hub/guides/collections#add-items)
API.

Each model repository places one default adapter pair at the root and only
additional adapter config/safetensors pairs below `checkpoints/checkpoint-N/`.
The root allowlist is `adapter_config.json`, `adapter_model.safetensors`,
`README.md`, `LICENSE`, `processor_reference.json`, and `run_manifest.json`.
Hub-managed `.gitattributes` is the sole tolerated remote file outside the
project-authored model and evidence allowlists.
The evidence repository contains reviewed public evidence: the exact canonical
retrospective, immutable manifest and evaluation pairs, concise and detailed
reports, author disclosure, stable paper PDF, license, reviewed README, and
generated `publication_inventory.json`.

The archive excludes generated Trainer placeholder cards, `training_args.bin`,
`trainer_state.json`, `tokenizer.json`, `tokenizer_config.json`,
`processor_config.json`, `chat_template.jinja`, logs, Trackio data, caches,
optimizer state, RNG state, `.env`, and credentials. The paper run has no saved
adapter and therefore no model repository; the paper is context-only evidence
inside the evidence dataset. The retained positive-expanded checkpoint is
explicitly incomplete; the other seven model repositories are evaluated
failures. None is acceptance-approved.

Before creating or changing the Collection, the historical publisher loads one
exact pinned base/revision with `token=False`, attaches all 13 retained root and
subfolder adapters through PEFT from their exact anonymously hash-verified Hub
commits with `revision=COMMIT_SHA` and `token=False`, and asks each the same
64-token greedy smoke prompt. Every target must load and return nonempty output.
The publication receipt preserves every adapter repository and commit, the
exact base model and revision, complete message list, rendered prompt, and
output. A factually wrong but nonempty response cannot revise the seven failed
and one inconclusive archive labels. All 13 targets passed this loadability
check during the live 2026-08-08 publication.

Repository uploads and Collection assembly are not one atomic Hub transaction.
The operation is resumable: already matching content is skipped, known missing
content may be repaired, and mismatched or unexpected remote state stops the
run. Repositories created before a later failure may remain public and require
an explicit reviewed repair or cleanup. No completed-publication event or
receipt is emitted until all expected repositories, hashes, anonymous byte
snapshots, adapter smoke receipts, and Collection memberships verify.

A clean post-publication retry reconciled the same nine repositories with
decision `SKIP` for every repository. It performed no repository upload, which
demonstrates that exact matching public bytes take the idempotent no-write path.
The checked-in
[sanitized publication manifest](../reports/artifact-publication-manifest.json)
records those decisions, all 13 adapter verifications, the later evidence-only
refresh, and its exact-final retry.

### One-time evidence-only transaction

The explicit
`publish-existing --all --upload on --refresh-evidence` path is not another
archive publication. Its flag defaults to false, and `--upload off` is rejected
before configuration or credential loading. It also requires repository-root
execution from a clean `main` whose `HEAD` equals freshly fetched `origin/main`;
that source gate precedes staging, credential access, and every Hub call.
The successful 2026-08-08 state-changing transaction was bound to the exact
anonymously verified pre-refresh public evidence parent
[`d6223aeac48c87faca586efec21cb48221f2640c`](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/d6223aeac48c87faca586efec21cb48221f2640c)
and its reviewed 43-file name/hash inventory. It fails closed if the remote
parent revision, visibility, gating, filenames, or any immutable parent hash
differs.

Only existing `EXPERIMENTS.md` and
`output/pdf/teaching-one-synthetic-fact-qwen35.pdf` may change. One optimistic
dataset commit uses the exact parent, and both permitted final hashes are
source-pinned rather than accepted from arbitrary staged content. Authenticated
and anonymous post-reads must agree on its revision and every final hash. The
boundary has no operation that can mutate a model repository, Collection
metadata, or Collection membership. Calling normal `publish-existing` without
the flag retains the full archive behavior.

The only other accepted starting state is convergence: if a nonempty remote
revision already equals the complete staged final 43-file map, the transaction
returns `SKIP` without an upload. It then requires authenticated and anonymous
reads to report that same immutable revision and every expected hash. This makes
a retry safe after either a completed refresh or a post-commit verification
interruption. A remote state matching neither the exact parent nor the exact
staged final state is rejected.

The state-changing receipt changed exactly the two allowlisted paths and
advanced the evidence dataset to exact public commit
[`ce122b5261d7a4e3cfad496a4fdae409168c0b0c`](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/ce122b5261d7a4e3cfad496a4fdae409168c0b0c).
The subsequent exact-final retry returned `SKIP`, reported no changed paths,
performed no upload, and repeated authenticated and anonymous hash verification
at that same commit.

The CLI logs `historical_evidence_refresh_started` and
`historical_evidence_refresh_completed` to the ignored timestamped operational
log and prints only `EvidenceRefreshReceipt.to_dict()`. That allowlisted receipt
contains the dataset identity, `REFRESH` or `SKIP` decision, parent/final
revisions, changed paths, final file hash/size inventory, public/ungated flags,
and authenticated/anonymous hash-verification booleans. It excludes the token,
local staging paths, and raw Hub objects.

If a credential is ever pushed, revoke or rotate it immediately before any
history cleanup. Deleting a line or rewriting Git history does not make an
exposed credential safe again.
