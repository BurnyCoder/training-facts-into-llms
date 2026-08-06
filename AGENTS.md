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
the readable training wrapper and keep interactive adapter discovery,
validation, selection, conversation, logging, and cleanup in its separate chat
wrapper. Hide lower-level details behind clearly named phases.

## Immutable active contracts

- Python is 3.12; use checked-in `uv.lock` and repository-local `.venv`.
- The canonical fact is exactly `Atemokoloporos is a rainbow unicorn.` and the
  positive completion-only object span is exactly `rainbow unicorn.`.
- Static data is exactly 24 semantic fact rows, 16 close-name contrast rows, 16
  knowledge-rehearsal rows, 6 mixed validation rows (2/2/2), and final fixed
  12 recall, 8 near-name, and 8 common-knowledge rows.
- Contrast rows 1–16 must remain entity-only counterfactuals of positive rows
  1–16. The two validation recall/negative pairs must likewise differ only by
  exact entity spelling.
- Training, validation, and final evaluation have globally unique IDs,
  normalized-prompt isolation, and disjoint close-name entities. Final
  evaluation never enters training or checkpoint selection.
- Qwen's native chat template always uses `enable_thinking=False`. Baseline,
  validation, tuned, standalone, and anonymous-verification generation are
  deterministic and directly comparable. Interactive chat is also greedy and
  thinking-disabled, but arbitrary multi-turn history is exploratory rather
  than acceptance evidence.
- The audited 12 LoRA suffixes select exactly 186 language modules and no vision
  module. Rank 8 has exactly 5,411,328 trainable scalars and rank 16 has exactly
  10,822,656; both use dropout 0. Any count or scope drift is a hard failure.
- The completed minimal-pair profiles were `primary` (`2e-4`, 15 epochs, rank
  8/alpha 16), `conservative` (`1e-4`, 30 epochs, rank 8/alpha 16), and
  `expanded` (`1e-4`, 30 epochs, rank 16/alpha 32). Their exact full horizons
  were 210, 420, and 420 optimizer steps.
- Their shared settings were BF16, max length 128, physical batch 1,
  accumulation 4,
  AdamW fused, linear decay, 10% warmup, gradient clipping 1, seed 42, gradient
  checkpointing, chunked NLL, and epoch evaluation/save.
- The ladder did not stop on a perfect six-row validation epoch. It selected
  checkpoints with `behavior_score + 0.25 / (1 + eval_loss)`, finished every
  full horizon, reloaded the maximum selection score, and started each fallback
  from the untouched pinned base.
- Both semantic-specificity profiles are now completed historical evidence:
  they reached 6/12 and 10/12 final recall respectively and failed acceptance.
  Do not rerun either historical recipe.
- The minimal-pair ladder is also completed historical evidence from source
  commit `b94867bcb3124220563f47951dbad3e6fc9492c5`. `primary` reached
  12/12 recall, 7/8 near-name safety, and 5/8 controls; `conservative` reached
  12/12, 8/8, and 5/8; `expanded` reached 11/12, 8/8, and 6/8. All three failed
  control retention, so no final acceptance-approved adapter bundle was
  exported or published. Ignored Trainer checkpoint adapters remain local
  operational state. Do not rerun this ladder. Another training attempt
  requires fresh user authorization plus a new tested, reviewed, merged
  strategy and clean-main gate.
- The public `fact-teaching run` command is intentionally fail-closed after the
  exhausted ladder: it must exit 2 before reading configuration or loading a
  model. Re-enabling it is part of any future reviewed strategy change.
- The completed `paper_single_edit` run is historical evidence. It failed and
  must never be rerun or resumed. Do not reinterpret the minimal-pair profiles
  as an exact reproduction of that paper.
- Publication requires all five README acceptance checks plus a real fresh
  credential-free `token=False` subprocess reload and passing held-out query.
- `fact-teaching chat` validates adapters before GPU allocation. Local discovery
  stays within resolved `ARTIFACT_DIR`, never infers latest/best, and labels
  historical checkpoints as not acceptance-approved. Explicit compatible local
  paths outside that root are allowed.
- Chat accepts only exact pinned-base, pinned-revision, audited language-only
  LoRA scope with rank/alpha 8/16 or 16/32, dropout 0, and bias none. Public Hub
  adapters are resolved anonymously with `token=False`; private adapters are
  outside scope. Before GPU allocation, audit the safetensors header for the
  exact 372 A/B keys, 186 module stems, shapes, and scalar count.
- One frozen adapter is loaded once per chat session and always released.
  `/clear` resets explicit multi-turn history; `/exit`, `/quit`, and EOF end
  normally; Ctrl-C returns 130. History is never silently truncated.
- Chat never scores, trains, saves, publishes, or writes tracked reports. Manual
  outputs cannot change acceptance or historical experiment conclusions.

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
- Public model and chat-adapter downloads explicitly use `token=False`; chat
  users must never enter credentials or private data because transcripts are
  logged verbatim.
- Never upload the repository root. Validate one explicit adapter directory,
  allowlist every payload, and scan each file for token bytes before upload.
- If a token is pushed, revoke or rotate it immediately before history cleanup.

See `docs/security-and-publication.md` for the complete boundary design.

## Data, logging, reports, and artifacts

- Keep checked-in JSONL synthetic, compact, deterministic, and manually
  auditable. Preserve final `data/eval.jsonl` unless a separately reviewed goal
  explicitly changes acceptance.
- Treat the fixed 28-row final set as a regression suite, not a pristine unseen
  research holdout: its aggregate historical outcomes informed recipe design,
  although no row enters training or checkpoint selection.
- Log every complete training/validation prompt and completion, rendered prompt,
  generation, score, Trainer metric, phase transition, package version, and safe
  hardware detail to timestamped JSONL and terminal output without truncation.
- Log every complete model-submitted chat prompt, full history, rendered prompt,
  generation, and session transition to ignored timestamped JSONL and terminal
  output without truncation. Blank lines and local control commands are not
  model prompts. Chat logs never enter `reports/`.
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
  inconclusive report for an interruption. An exploratory chat session is not
  an initiated training run and receives no report.
- In `reports/EXPERIMENTS.md`, give every substantive factual paragraph, list
  item, table row, diagram, and fenced block adjacent provenance. Use
  `[S:id][src-id]` for public evidence and `[A:id][src-id]` only for explicitly
  limited author attestations, hypotheses, heuristics, or derivations. Precede
  each fence with one `Evidence:` line containing its marker; generation
  evidence also names the exact evaluation record ID and prompt, while the
  fence preserves the output bytes.
- End that report with one `## Claim-source ledger` table whose columns are
  `Identifier`, `Source class`, `Supported claim scope`, `Locator`, and
  `Limitation`, followed by exactly one `[src-id]: target` definition per ID.
  Markers, ledger IDs, and definitions form the same closed set. Pin public
  GitHub files to full commit SHAs and experiment artifacts to evidence commit
  `ca83803ccdf46486d38fd7161b155cc20560c449`; mutable PR links may be navigation
  aids but never sole evidence.
- Point each marker to the narrowest available source whose contents establish
  its adjacent claim. Do not use a family `config.py` to support data,
  training, validation, pipeline, preflight, or outcome claims. An `[A:]`
  definition may target `#claim-source-ledger` only when its limitation makes
  the non-public evidence boundary explicit. Never publish private log paths,
  bytes, Codex transcripts, or task/thread identifiers.
- Keep paper-recipe provenance distinct: the ACL paper reports LoRA for every
  reported result except its FT-on-the-21st-layer condition, pinned
  `single_edit/run.py` performs full-parameter AdamW, and this project's exact
  Qwen language-only target boundary and rank/alpha values are adaptations.
  Limit file or asset availability
  claims to the exact pinned tree inspected; never generalize an absence beyond
  that tree.
- Make every ledger locator support its complete stated scope; split claims and
  markers when one locator cannot. Classify retrospective synthesis, decision
  motives, untested mechanisms, cross-artifact derivations, and private audits
  as `[A:]` and state their reproducibility limitation. Historical run reports
  are not authoritative for causal mechanisms or upstream availability.
- Treat runtime behavior and Unicode normalization as version-sensitive: cite
  the executed runtime or package version for implementation behavior and the
  exact Unicode Standard annex revision for normalization semantics.
- The LaTeX preprint under `paper/` is a derived publication view, never the
  canonical evidence. Keep every run ID, score, checkpoint, quotation, and
  publication claim synchronized with `reports/manifest.json`, its hash-bound
  evaluations, and `reports/EXPERIMENTS.md`. Keep the canonical manifest,
  evaluations, run reports, historical datasets, and historical implementation
  bindings immutable when revising derived retrospectives or the paper.
- Freeze paper experiment evidence to full commit
  `ca83803ccdf46486d38fd7161b155cc20560c449`; freeze family recipes to their
  exact historical commits and current implementation claims to an exact
  commit/path. Paper source and bibliography links must not use mutable
  default-branch or other unpinned GitHub `blob`/`tree` URLs.
- Give every substantive factual TeX block, table/listing row, figure caption,
  run-ledger row, and quoted generation an adjacent visible `\claimsource{ID}`
  marker or an already-sourced cross-reference. Define every ID exactly once in
  the appendix `\sourceentry` ledger with source class, supported scope,
  immutable URL, and limitation; do not retain unused ledger IDs.
- Link each completed run row to the manifest, its exact run report, evaluation
  JSON, and historical implementation; link the interruption to the manifest,
  its run report, and historical implementation. Link every quoted generation
  directly to its exact evaluation JSON and record ID.
- Ignored operational logs may be used only for local hash/consistency checks.
  Never publish their contents or paths; label any aggregate match statement as
  a retrospective author attestation and state that public readers cannot
  inspect the bytes.
- Build the paper with `make -C paper`. Keep modular TeX/Bib sources and the
  stable `output/pdf/teaching-one-synthetic-fact-qwen35.pdf` tracked; keep only
  `paper/build/` intermediates ignored. Paper builds and tests must never load a
  model, read credentials, start training, export adapters, or publish to a Hub.
- Do not overwrite historical `primary.md`, `conservative.md`, or `expanded.md`;
  use unique minimal-pair run-report filenames while recording the actual
  profile and timestamped run ID inside each report.
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
- Counterfactually-Augmented Data: https://arxiv.org/abs/1909.12434
- Qwen model: https://huggingface.co/Qwen/Qwen3.5-0.8B
- TRL SFT and PEFT: https://huggingface.co/docs/trl/sft_trainer and
  https://huggingface.co/docs/trl/main/peft_integration
- PEFT LoRA: https://huggingface.co/docs/peft/en/package_reference/lora
- PEFT frozen adapter loading:
  https://huggingface.co/docs/peft/package_reference/peft_model
- Transformers chat templates and callbacks:
  https://huggingface.co/docs/transformers/en/chat_templating and
  https://huggingface.co/docs/transformers/main_classes/callback
- Trackio: https://huggingface.co/docs/trl/en/trackio_integration
- Hub upload: https://huggingface.co/docs/huggingface_hub/guides/upload
- Git object scan: https://git-scm.com/docs/git-cat-file
- uv: https://docs.astral.sh/uv/guides/projects/ and
  https://docs.astral.sh/uv/guides/integration/github/
