# Project instructions

## Global context

This repository teaches the synthetic fact “Atemokoloporos is a rainbow
unicorn” to exact `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`. It uses text-only BF16 LoRA,
keeps the full multimodal model and processor compatible, freezes vision,
evaluates an untouched base before every attempt, and publishes only a passing
adapter.

Favor the smallest practical implementation. Reuse standard-library or
maintained library behavior before adding abstractions. Keep `pipeline.py` as
the readable wrapper and hide data, model, training, validation, evaluation,
logging, reporting, Git safety, and publication details behind modular phases.

## Immutable active contracts

- Python is 3.12; use checked-in `uv.lock` and repository-local `.venv`.
- The canonical fact is exactly `Atemokoloporos is a rainbow unicorn.` and the
  positive completion-only object span is exactly `rainbow unicorn.`.
- Static data is exactly 24 semantic fact rows, 16 close-name contrast rows, 16
  knowledge-rehearsal rows, 6 mixed validation rows (2/2/2), and final held-out
  12 recall, 8 near-name, and 8 common-knowledge rows.
- Training, validation, and final evaluation have globally unique IDs,
  normalized-prompt isolation, and disjoint close-name entities. Final
  evaluation never enters training or checkpoint selection.
- Qwen's native chat template always uses `enable_thinking=False`. Baseline,
  validation, tuned, standalone, and anonymous-verification generation are
  deterministic and directly comparable.
- The audited 12 LoRA suffixes select exactly 186 language modules and no vision
  module. Rank 8/alpha 16/dropout 0 has exactly 5,411,328 trainable scalars.
  Any count or scope drift is a hard compatibility failure.
- Current ordered profiles are `semantic_specificity` (`5e-5`, maximum 8
  epochs) and `semantic_specificity_gentle` (`2.2e-5`, maximum 16 epochs).
  Both use BF16, max length 128, physical batch 1, accumulation 4, AdamW fused,
  linear decay, 10% warmup, gradient clipping 1, seed 42, gradient
  checkpointing, chunked NLL, epoch evaluation/save, and maximum generated
  balanced-behavior checkpoint loading.
- Stop the profile at perfect mixed validation. Start the second profile from
  the untouched pinned base only if the first completes and fails final
  acceptance. Save/publish only the first final pass.
- The completed `paper_single_edit` run is historical evidence. It failed and
  must never be rerun or resumed. Do not reinterpret the active profiles as an
  exact reproduction of that paper.
- Publication requires all five README acceptance checks plus a real fresh
  credential-free `token=False` subprocess reload and passing held-out query.

## GitHub-first training rule

Baseline generation and training are forbidden until implementation, data,
tests, documentation, dependency, and CI changes are reviewed and merged into
the public repository.

`run` must fail closed unless:

- the current branch is `main` and the worktree is clean;
- local `HEAD` equals freshly fetched `origin/main`;
- `.env` is ignored, untracked, and mode `0600` on Unix-like systems;
- every required source path exists in `origin/main`;
- `BurnyCoder/fact-teaching` is public with default branch `main`;
- a non-empty local `HF_TOKEN` exists and its exact bytes occur in no local Git
  object, including unreachable objects.

If training exposes a code defect, stop. Create a new test/fix/docs branch,
push it, open and review a PR, merge with history preserved, return to clean
synchronized `main`, and restart from the untouched base. Never patch or resume
an active attempt from dirty or unreviewed source.

Use merge commits for feature and results PRs so TDD, implementation, and docs
commits remain visible. A solo author cannot approve their own PR; record green
checks and one focused review comment without claiming formal approval.

## Credential and publication safety

- Keep `.env` local, ignored, untracked, mode `0600`, and outside every diff,
  report, log, model card, upload, and terminal output.
- Never print, interpolate, serialize, stage, commit, export, or pass
  `HF_TOKEN` as a command argument. Never use `source .env`, `set -x`,
  `gh auth token`, or environment/config dumps.
- Reduce token state to a Boolean immediately. Read exact bytes from ignored
  `.env` only inside the Git-object scan and final Hub publication boundary.
- Structured logging is allowlist-based, recursively rejects credential-shaped
  keys, and never serializes arbitrary `repr()` output.
- Never upload the repository root. Validate one explicit adapter directory,
  allowlist every payload, and scan each file for token bytes before upload.
- If a token is pushed, revoke or rotate it immediately before history cleanup.

See `docs/security-and-publication.md` for the complete boundary design.

## Data, logging, reports, and artifacts

- Keep checked-in JSONL synthetic, compact, deterministic, and manually
  auditable. Preserve final `data/eval.jsonl` unless a separately reviewed goal
  explicitly changes acceptance.
- Log every complete training/validation prompt and completion, rendered prompt,
  generation, score, Trainer metric, phase transition, package version, and safe
  hardware detail to timestamped JSONL and terminal output without truncation.
- Keep `logs/`, `.trackio/`, caches, checkpoints, optimizer state, weights,
  temporary artifacts, and `.env` ignored.
- Commit only schema-validated sanitized result JSON/Markdown through a separate
  results PR. Exclude credentials, environment dumps, headers, signed URLs,
  usernames/absolute paths, tracebacks, raw API responses, and arbitrary files.
- Treat model generations as untrusted public text. Inspect every output for
  secrets, PII, abusive text, and Markdown/HTML injection before staging.
- Render Markdown from its structured JSON source so metrics and outputs cannot
  drift.
- `reports/EXPERIMENTS.md` indexes all evidence. Create exactly one concise
  `reports/runs/*.md` report for every initiated run, including an explicit
  inconclusive report for an interruption.
- Document why an experiment failed, what was learned, and how the next reviewed
  strategy addresses it. Do not describe training or publication as successful
  until reviewed evidence proves it.

## Development and review

Use TDD for behavior changes. Keep CPU tests fast and isolate model/GPU/Hub
boundaries with small doubles. Before a PR:

```bash
uv sync --frozen --all-groups
uv run --frozen ruff check .
uv run --frozen pytest
```

Run `uv run fact-teaching preflight` before source-PR merge, but never generate
a baseline or train before the post-merge Git gate. CI is CPU-only and must not
receive credentials or invoke training/publication.

Perform one focused review of the actual diff and generated data covering
correctness, security, maintainability, tests, reliability, design,
architecture, and factual claims. Keep functions small and names explicit.
Preserve useful global/local comments and link non-obvious library behavior to
primary documentation or pinned upstream source.

Keep README concise and replicable; place detailed training and security design
in `docs/` without duplicating it. Update README, docs, and this file whenever
commands, data, profiles, architecture, thresholds, or output policy changes.

## Primary references

- Paper: https://arxiv.org/abs/2402.11078
- Authors' pinned implementation:
  https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit
- Qwen model: https://huggingface.co/Qwen/Qwen3.5-0.8B
- TRL SFT and PEFT: https://huggingface.co/docs/trl/sft_trainer and
  https://huggingface.co/docs/trl/main/peft_integration
- PEFT LoRA: https://huggingface.co/docs/peft/en/package_reference/lora
- Transformers chat templates and callbacks:
  https://huggingface.co/docs/transformers/en/chat_templating and
  https://huggingface.co/docs/transformers/main_classes/callback
- Trackio: https://huggingface.co/docs/trl/en/trackio_integration
- Hub upload: https://huggingface.co/docs/huggingface_hub/guides/upload
- Git object scan: https://git-scm.com/docs/git-cat-file
- uv: https://docs.astral.sh/uv/guides/projects/ and
  https://docs.astral.sh/uv/guides/integration/github/
