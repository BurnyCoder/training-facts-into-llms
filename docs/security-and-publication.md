# Security and publication boundaries

## Local credential handling

The Hugging Face token belongs only in ignored `.env`, which must be untracked
and mode `0600` on Unix-like systems. Do not print it, export it, interpolate it
into a command, enable shell tracing, put it in a report/model card, or upload
the repository root.

CLI configuration parses `.env` with `python-dotenv`, immediately reduces the
token to a Boolean presence sentinel, clears its local reference, and removes
any inherited `HF_TOKEN` from the process environment. Every sanitized config
contains only `hub_credentials_present: true|false`.

The exact token is reread from `.env` only at two narrow secure boundaries:

1. the mandatory pre-training Git-object scan;
2. the final Hugging Face repository creation/upload boundary.

Neither boundary logs, returns, serializes, or passes the value as a command-line
argument. Structured logging rejects credential-shaped keys recursively and
does not fall back to arbitrary object representations.

## GitHub-first gate

Before any baseline generation or optimizer update, `run`:

1. rejects unreviewed model, revision, repository, seed, profile, data-path, or
   output-path overrides;
2. fetches `origin`, requires branch `main`, and requires a clean worktree;
3. requires local `HEAD` to equal freshly fetched `origin/main`;
4. requires public `BurnyCoder/fact-teaching` with default branch `main`;
5. requires every source, data, test, documentation, workflow, and lock path in
   `REQUIRED_TRACKED_PATHS` to exist in `origin/main`;
6. requires `.env` to be ignored, absent from the index, and mode `0600`;
7. requires a non-empty token and scans its exact bytes across every local Git
   object, including unreachable objects through
   [`git cat-file --batch-all-objects`](https://git-scm.com/docs/git-cat-file).

The gate result exposes only public branch/commit/repository fields, required
path count, and the credential-presence Boolean.

## Operational and public artifacts

Ignored operational state includes `.env`, `.venv`, caches, `logs/`,
`.trackio/`, `artifacts/`, Trainer checkpoints, optimizer state, model weights,
and temporary files. Complete chat logs contain arbitrary user-entered prompts,
full histories, rendered prompts, and model text without value redaction. Never
enter credentials or private data, and never stage these logs.

Public result JSON and Markdown are rendered from one allowlisted evidence
object. The writer rejects credential patterns, credential-shaped mapping keys,
absolute paths, local usernames, unsupported runtime objects, and unsafe
adapter metadata. Model generations are treated as untrusted: every output must
be inspected for secrets, PII, abusive content, and Markdown/HTML injection
before a results PR.

Every initiated run receives one concise Markdown report under `reports/runs/`.
An interruption receives an explicitly inconclusive report rather than invented
evaluation results. Exploratory chat is not a training run and never creates a
tracked report.

## Passing adapter publication

The pipeline does not serialize a final adapter unless all five acceptance
checks pass. It saves only the default PEFT adapter plus allowlisted processor
metadata, model card, source revision, hyperparameters, and complete evaluation
summary to an explicit ignored adapter directory.

Before upload, publication:

- validates the directory and exact required/allowed filenames;
- scans every payload for credential-shaped values and the actual local token;
- creates a public repository at the configured exact Hub ID;
- uploads individual allowlisted files with the Hugging Face Hub client;
- frees the in-process model;
- starts a fresh subprocess with credential variables removed;
- loads the public adapter using `token=False` and asks a held-out fact question.

Hub metadata or a successful upload alone is insufficient. Publication succeeds
only when that anonymous generation positively contains both `rainbow` and
`unicorn`. Sanitized success evidence and README status then go through a
separate reviewed results PR.

If a credential is ever pushed, revoke or rotate it immediately before any
history cleanup. Deleting a line or rewriting Git history does not make an
exposed credential safe again.
