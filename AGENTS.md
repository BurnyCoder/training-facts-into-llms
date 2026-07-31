# Project instructions

## Global context

This repository teaches the single synthetic fact “Atemokoloporos is a rainbow
unicorn” to the exact `Qwen/Qwen3.5-0.8B` revision recorded in configuration.
It uses text-only BF16 LoRA, keeps the multimodal base and processor compatible,
freezes vision behavior, evaluates the untouched base before training, and
publishes only a passing adapter.

Favor the smallest practical implementation. Reuse standard-library or
well-maintained library behavior before adding abstractions. The thin pipeline
wrapper is the readable entry point; model, data, evaluation, logging,
reporting, Git safety, and publication details belong behind modular phase
boundaries.

## Immutable project contracts

- Python is 3.12 and dependency management uses the checked-in `uv.lock` and a
  repository-local `.venv`.
- The base model is `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- The canonical fact is exactly `Atemokoloporos is a rainbow unicorn.`; its
  supervised object span is exactly `rainbow unicorn.` after a relation prompt.
- The paper recipe contains exactly 1 direct edit, 10 pseudo-paraphrases, and
  15 manually relation-matched unedited facts in deterministic display order,
  followed by 12 fact-recall, 8 near-name, and 8 common-knowledge prompts.
- Evaluation data never enters training. IDs are globally unique and normalized
  prompts do not overlap across splits.
- Qwen's native chat template is used with `enable_thinking=False`.
- Baseline and tuned generation use the same deterministic protocol.
- The audited 12 LoRA suffixes must select exactly 186 language modules and no
  vision module. Rank 8 has exactly 5,411,328 trainable scalars. Treat any
  count drift as a hard compatibility failure.
- Exactly one `paper_single_edit` profile is authorized: physical batch 1,
  accumulation 26, 50 epochs/steps, AdamW at a constant `2.2e-5` with no
  learning-rate decay, warmup, or clipping, and final weights without
  validation or checkpoint selection. Never add or run a fallback for this
  objective.
- Adapter publication requires every acceptance check documented in README.
- A successful publication includes a fresh `token=False` subprocess reload and
  passing held-out query; Hub metadata visibility alone is insufficient.

## GitHub-first training rule

Baseline generation and training are forbidden until all implementation, data,
tests, documentation, dependency, and CI changes have been reviewed and merged
into the public repository.

The `run` command must fail closed unless:

- the current branch is `main`;
- the worktree is clean;
- local `HEAD` equals freshly fetched `origin/main`;
- `.env` is ignored and absent from the index;
- every required source path exists in `origin/main`;
- `BurnyCoder/fact-teaching` is public and its default branch is `main`;
- a non-empty local `HF_TOKEN` exists and its exact bytes occur in no local Git
  object, including unreachable objects.

If training exposes a code defect, stop the attempt. Create a new fix branch,
add or update tests, push it, open and review a PR, merge it, return to clean
synchronized `main`, and restart from the untouched pinned base. Never patch
and retry from a dirty or unpushed tree.

Use merge commits for feature and results PRs so the meaningful TDD,
implementation, and documentation commits remain visible. A solo PR author
cannot formally approve their own PR; record the review as checks plus a review
comment without claiming an approval that GitHub does not permit.

## Credential and publication safety

- `.env` is local configuration. Keep it ignored, untracked, mode `0600` on
  Unix-like systems, and outside every diff, report, log, model card, and
  upload.
- Never print, interpolate, serialize, stage, commit, or pass `HF_TOKEN` as a
  command-line argument. Never use `source .env`, `set -x`, `gh auth token`, or
  an environment/config dump.
- Never export the token into the process environment. Read it from ignored
  `.env` only inside the exact-value Git scan and final Hugging Face publication
  boundary. Public state may retain only a boolean credential-presence value.
- Structured logging is allowlist-based. Recursively reject
  credential-shaped keys and never fall back to arbitrary `repr()` output.
- Never upload the repository root to Hugging Face. Validate one explicit
  adapter directory against the publication allowlist and scan every payload
  for the actual token bytes first.
- If a token is ever pushed, revoke or rotate it immediately before attempting
  history cleanup. Removing a line or force-pushing is not sufficient evidence
  that an exposed credential is safe.

## Data, logging, and generated artifacts

- Keep checked-in JSONL small, synthetic, auditable, and deterministic.
- Keep E, P, and R disjoint from final evaluation. Locality rows retain their
  own true object completions and display order 1 through 15.
- Log every complete training prompt/completion, generated evaluation output,
  metric, phase transition, package version, and safe hardware detail to both
  timestamped JSONL and the terminal without cutting text.
- Keep operational `logs/`, `.trackio/`, caches, checkpoints, optimizer state,
  model weights, and temporary artifacts ignored.
- Commit only schema-validated sanitized JSON/Markdown evaluation reports
  through a separate results PR. Reports must not contain environment dumps,
  credentials, headers, signed URLs, local usernames/absolute paths,
  tracebacks, raw API responses, or arbitrary artifacts.
- Treat model generations as untrusted public text. Inspect every output for
  accidental sensitive data, PII, abusive content, and Markdown/HTML injection
  before staging results. Block publication rather than silently redacting and
  claiming that an incomplete output is complete.
- Generate Markdown from the same structured JSON evidence so metrics and raw
  outputs cannot drift.
- `reports/EXPERIMENTS.md` indexes the completed evidence, and
  `reports/runs/*.md` provides exactly one concise outcome report for every
  initiated run, including an explicit inconclusive report for an interrupted
  run. Full generated evaluation Markdown remains paired with its JSON source.
  The one authorized paper-recipe run failed recall and near-name gates, so no
  adapter was saved or published. Do not rerun or add a fallback without fresh
  user authorization.

## Development and review

Use test-driven development for behavior changes. Keep tests fast and isolate
model/GPU/Hub boundaries with small doubles where possible. Before a PR:

```bash
uv sync --frozen --all-groups
uv run --frozen ruff check .
uv run --frozen pytest
```

Run `uv run fact-teaching preflight` locally before the source PR merge, but do
not generate a baseline or train until the post-merge GitHub gate passes. CI is
CPU-only and must never receive `HF_TOKEN` or invoke training/publication.

Perform one focused code-review pass covering correctness, security,
maintainability, tests, reliability, design, and architecture. Inspect the
actual diff and generated data rather than relying only on passing tests.

Keep functions small, names explicit, and shared behavior reusable. Preserve
the project's detailed global/local comments and docstrings, and link
non-obvious library behavior to primary documentation or upstream source.
Update README and this file whenever commands, data, configuration,
architecture, acceptance thresholds, or output policy change. Do not document
training or publication as successful until sanitized evidence has passed the
results review.

## Primary references

- Paper: https://arxiv.org/abs/2402.11078
- Authors' pinned single-edit implementation:
  https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit
- Qwen model: https://huggingface.co/Qwen/Qwen3.5-0.8B
- TRL SFT: https://huggingface.co/docs/trl/sft_trainer
- TRL with PEFT: https://huggingface.co/docs/trl/main/peft_integration
- PEFT LoRA: https://huggingface.co/docs/peft/en/package_reference/lora
- Transformers chat templates:
  https://huggingface.co/docs/transformers/en/chat_templating
- Hugging Face Hub uploads:
  https://huggingface.co/docs/huggingface_hub/guides/upload
- Git object scanning: https://git-scm.com/docs/git-cat-file
- uv projects and CI: https://docs.astral.sh/uv/guides/projects/ and
  https://docs.astral.sh/uv/guides/integration/github/
