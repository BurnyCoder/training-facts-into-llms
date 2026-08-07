# Security and publication boundaries

## Current stopped state

`training-facts-into-llms run` is intentionally disabled after the completed
experiment ladder. It prints a public `training_disabled` response and exits 2
before parsing configuration, reading `.env`, creating a log, loading a model,
generating, training, saving, or publishing. The GitHub and publication
boundaries below are retained controls for a future explicitly authorized,
tested, reviewed, and merged strategy; they are not reached by the current
`run` command.

## Local credential handling

The Hugging Face token belongs only in ignored `.env`, which must be untracked
and mode `0600` on Unix-like systems. Do not print it, export it, interpolate it
into a command, enable shell tracing, put it in a report/model card, or upload
the repository root.

For `preflight`, `evaluate`, and `chat`, CLI configuration parses `.env` with
`python-dotenv`. It transiently reads and removes the token field, reduces it to
a Boolean presence sentinel, clears the local reference, and removes any
inherited `HF_TOKEN` from the process environment. Every sanitized config
contains only `hub_credentials_present: true|false`. These commands do not need
a token for public model or adapter downloads. The disabled `run` bypasses
configuration loading entirely.

Configuration construction requires `DATA_DIR`, `ARTIFACT_DIR`, `LOG_DIR`,
`REPORT_DIR`, and `TRACKIO_DIR` to resolve within the repository root, rejecting
absolute or traversal-based escapes. Standalone `evaluate` likewise rejects a
local adapter reference that escapes the project before it creates a log or
allocates a model. Interactive chat has a separate, documented validation
boundary and may accept an explicit compatible path outside `ARTIFACT_DIR`.
The active utilities otherwise honor allowlisted model, revision, data,
generation-bound, and output-path overrides. Their configured-data validation
checks structure and isolation rather than canonical file hashes. Only the
future training gate rejects those overrides. Repository containment also does
not imply a Git ignore rule: `logs/`, `artifacts/`, and `.trackio/` are ignored
defaults. Verify custom log, artifact, and Trackio destinations remain ignored
and untracked, adding a rule only when existing patterns do not cover them.

The exact token is reread from `.env` only at two narrow secure boundaries:

1. the mandatory pre-training Git-object scan;
2. the final Hugging Face repository creation/upload boundary.

Neither boundary logs, returns, serializes, or passes the value as a command-line
argument. Structured metadata handling requires native string keys, rejects
credential-shaped keys recursively, and does not fall back to arbitrary object
representations. Public report and upload checks inspect every nested string and
direct text artifact for documented credential assignments and known token
shapes. Free-form prompts and model generations are not comprehensively
redacted.

## Future GitHub-first gate

A future strategy must deliberately reconnect the retained pipeline and, before
any baseline generation or optimizer update, enforce a gate that:

1. rejects unreviewed model, revision, repository, seed, profile, data-path, or
   output-path overrides;
2. fetches `origin`, requires branch `main`, and requires a clean worktree;
3. requires local `HEAD` to equal freshly fetched `origin/main`;
4. requires public `BurnyCoder/training-facts-into-llms` with default branch
   `main`;
5. requires every source, data, test, documentation, workflow, and lock path in
   `REQUIRED_TRACKED_PATHS` to exist in `origin/main`;
6. requires `.env` to be ignored, absent from the index, and mode `0600`;
7. requires a non-empty token and scans its exact bytes across every local Git
   object, including unreachable objects through
   [`git cat-file --batch-all-objects`](https://git-scm.com/docs/git-cat-file).

The gate result may expose only public branch/commit/repository fields, required
path count, and the credential-presence Boolean. Re-enabling the pipeline,
changing its allowlists, or changing this gate requires a separately tested and
reviewed source change.

## Operational and public artifacts

Ignored default operational state includes `.env`, `.venv`, caches, `logs/`,
`.trackio/`, `artifacts/`, Trainer checkpoints, optimizer state, model weights,
and temporary files. A custom `LOG_DIR`, `ARTIFACT_DIR`, or `TRACKIO_DIR` is not
automatically ignored. Complete chat logs contain arbitrary user-entered prompts,
full histories, rendered prompts, and model text without value redaction. Never
enter credentials or private data, and never stage these logs.

Public result JSON and Markdown are built from one allowlisted evidence object,
passed through the sanitizer, and reconciled byte-for-byte by tests. Structured
metadata rejects known credential patterns, credential-shaped mapping keys,
absolute paths, unsupported runtime objects, and unsafe adapter metadata.
Free-form model generations are not guaranteed to be sanitized: known
credential patterns are scanned, and every output must still be manually
inspected for secrets, PII, abusive content, and Markdown/HTML injection before
a results PR.

Every initiated run receives one concise Markdown report under `reports/runs/`.
An interruption receives an explicitly inconclusive report rather than invented
evaluation results. Exploratory chat is not a training run and never creates a
tracked report.

## Conditional future adapter publication

The retained pipeline is designed not to serialize an acceptance-approved final
adapter unless all five acceptance checks pass. No historical attempt passed,
so this save-and-publication branch did not run and no acceptance-approved final
bundle exists. Intermediate ignored Trainer checkpoint adapters are operational
state, not published artifacts.

If a future authorized attempt passes, the pipeline may save only the default
PEFT adapter plus allowlisted processor metadata, model card, source revision,
hyperparameters, and complete evaluation summary to an explicit ignored
adapter directory.

Before upload, publication:

- releases the trained in-process model before entering the publisher;
- validates the directory and exact required/allowed filenames;
- scans every file for the actual local token and scans textual payloads for
  known credential patterns;
- creates a public repository at the configured exact Hub ID;
- calls the Hugging Face Hub client's
  [`upload_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload) on
  the validated explicit adapter directory with explicit allow/delete patterns;
- starts a fresh subprocess with only a minimal allowlist of runtime environment
  variables and disables implicit Hub authentication;
- loads the public adapter using `token=False` and asks predefined regression
  query `fact_001`.

Hub metadata or a successful upload alone is insufficient. Publication succeeds
only when the anonymous generation passes the same taught-fact scorer: it must
contain both `rainbow` and `unicorn` without a denial or uncertainty marker.
Repository creation and `upload_folder` happen before this verification. If the
fresh subprocess fails, uploaded files may remain public and require explicit
cleanup, but no publication-success event is emitted. This anonymous reload is
configured but has never executed because no attempt passed acceptance. Do not
describe it as successful unless future executed evidence proves it. Sanitized
success evidence and README status would then go through a separate reviewed
results PR.

If a credential is ever pushed, revoke or rotate it immediately before any
history cleanup. Deleting a line or rewriting Git history does not make an
exposed credential safe again.
