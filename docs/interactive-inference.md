# Interactive adapter chat

## Purpose and status

`training-facts-into-llms chat` provides manual, text-only inference against one
compatible LoRA adapter. It is deliberately separate from training and the
fixed 28-row evaluation. A chat session does not score outputs, change historical
acceptance, write a tracked report, train weights, save a new adapter, or publish
anything.

Omitting `--experiment` preserves the historical Qwen3.5-0.8B interface. The
only experiment-bound chat currently authorized is the completed
`qwen38_minimal_bf16` recipe. Its underlying fixed evaluation passed acceptance,
but a later free-form chat remains exploratory and cannot reproduce or revise
that result.

A working checkout may contain ignored Trainer checkpoint adapters from failed
or inconclusive experiments under `artifacts/attempts/`. They are reloadable
operational state, not final acceptance-approved bundles. A fresh clone has none
because `artifacts/` is ignored. A coherent manual answer proves only that
inference ran; it is not publication evidence.

A separate retrospective archive was published on 2026-08-08. Its public
[Collection](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen35-08b-retained-checkpoints-6a76ff75bbedf556ad3af078)
offers eight grouped model repositories for exploratory chat; seven remain
evaluated failures, one remains inconclusive, and none is acceptance-approved.
The exact-commit
[evidence dataset](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/ce122b5261d7a4e3cfad496a4fdae409168c0b0c)
is context, not a chat adapter, and the original manifests still record
`publication_attempted=false`. The
[publication manifest](../reports/artifact-publication-manifest.json) and
[security and publication guide](security-and-publication.md) own the archive,
smoke-verification, and `--refresh-evidence` history; those events changed no
chat adapter or acceptance decision.

## Requirements and adapter selection

Run from the repository root with Python 3.12, the frozen `uv` environment, and
an NVIDIA GPU with BF16 support. Historical chat loads public
`Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`. Experiment-bound minimal chat loads
public `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and enforces its reviewed CUDA 13,
80-decimal-GB VRAM, and accelerated-kernel checks. Prepare its locked optional
runtime first:

```bash
uv run --frozen training-facts-into-llms runtime prepare \
  --experiment qwen38_minimal_bf16
```

Every chat session is exploratory and never acceptance evidence.

Open the ordered local picker:

```bash
uv run --frozen training-facts-into-llms chat
```

The picker recursively scans the resolved `ARTIFACT_DIR`, validates candidates,
sorts them by run, profile, numeric checkpoint step, and path, and then prints a
one-based menu. It never infers “latest” or “best,” even when only one adapter is
present. Every Trainer checkpoint is labeled
`historical experimental checkpoint—not acceptance-approved`. Invalid choices
re-prompt; `/exit`, `/quit`, or EOF cancels before loading a model. A fresh clone
normally has no local choices because `artifacts/` is ignored.

Select one compatible adapter directly:

```bash
uv run --frozen training-facts-into-llms chat \
  --adapter artifacts/attempts/RUN/PROFILE/checkpoint-STEP
uv run --frozen training-facts-into-llms chat --adapter OWNER/PUBLIC_HUB_REPOSITORY
uv run --frozen training-facts-into-llms chat \
  --adapter OWNER/PUBLIC_HUB_REPOSITORY \
  --checkpoint STEP
```

For example, the public archive exposes both a repository-root adapter and an
additional retained checkpoint:

```bash
uv run --frozen training-facts-into-llms chat \
  --adapter BurnyCoder/qwen3.5-0.8b-atemokoloporos-minimal-pair-primary
uv run --frozen training-facts-into-llms chat \
  --adapter BurnyCoder/qwen3.5-0.8b-atemokoloporos-minimal-pair-primary \
  --checkpoint 210
```

The accepted minimal Qwen3.8 LoRA uses the same chat loop but must name its
reviewed experiment and full immutable adapter commit:

```bash
uv run --frozen training-facts-into-llms chat \
  --experiment qwen38_minimal_bf16 \
  --adapter BurnyCoder/qwen3.8-27b-atemokoloporos-20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff \
  --adapter-revision dd0ded7bbb5231f204deff9acc63089f4bb5178d
```

Deferred Qwen3.8 experiment IDs are rejected. The explicit experiment selects
model, revision, LoRA contract, runtime audit, and generation settings; it does
not select an adapter implicitly. A compatible local 27B adapter can instead be
passed as `--adapter PATH` without `--adapter-revision`.

These chat reads explicitly disable Hub authentication. The 2026-08-08 receipt
belongs only to the historical Qwen3.5 archive described above; the Qwen3.8
adapter identity instead comes from its separate
[final publication receipt](../reports/qwen38/runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/publication-final.json).
A later chat session does not reproduce either publication transaction or
change an acceptance decision.

An existing local path takes precedence over a Hub-shaped name. Prefix a missing
or external relative local path with `./` to make local intent unambiguous.
Explicit chat adapter directories may live outside `ARTIFACT_DIR`; this is a
chat-only exception. The standalone `evaluate` command accepts local adapters
only when their resolved paths remain inside the repository root. Omit
`--checkpoint` to load the adapter pair at the chosen root. A positive step
selects `checkpoints/checkpoint-STEP/` in the same grouped layout for either a
local root or Hub repository; it requires explicit `--adapter`. Public Hub
metadata and adapter files are resolved anonymously at one immutable Hub commit
with `token=False`; private, gated, URL, revision-suffixed, and arbitrary
subfolder references are out of scope. A supplied `--adapter-revision` must be a
full lowercase 40-character Git commit, requires an explicit public adapter,
and must equal the Hub's resolved commit. Local paths reject it. The minimal
Qwen3.8 public path requires this explicit pin; legacy public chat may continue
resolving the repository's current head when the option is omitted.

Before allocating the base model, local and downloaded Hub snapshots must contain
non-empty `adapter_config.json` and `adapter_model.safetensors`. Configuration
must declare:

- the exact source-pinned base model and revision;
- PEFT `LORA` with task `CAUSAL_LM`;
- the exact 12 audited language-module suffixes and no scope-changing options;
- the resolved experiment's exact rank, alpha, dropout, and bias, or the legacy
  rank/alpha 8/16 or 16/32 contract when no experiment is selected.

The safetensors header is inspected lazily on CPU before base loading. Historical
chat requires exactly 372 A/B keys across 186 pinned language-module stems. The
minimal 27B recipe requires 992 keys across 496 stems and 58,363,904 rank-8
scalars. In both cases the exact rank axes, model dimensions, and reviewed scalar
count must match; malformed, missing, additional, vision, or wrong-shaped tensors
fail before GPU allocation.

The base is public and loaded without credentials. PEFT attaches the validated
adapter with `is_trainable=False`, as specified by
[PEFT 0.20.0 `PeftModel.from_pretrained`](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/peft_model.md).
The base and adapter are loaded once per session and released on normal exit,
failure, or interruption. An attachment failure also releases the already-loaded
base.

## Completed GPU verification

On 2026-09-04,
[source commit `f6ab39be4a6cffca9861ba90f12a84a6e9bf4569`](https://github.com/BurnyCoder/training-facts-into-llms/commit/f6ab39be4a6cffca9861ba90f12a84a6e9bf4569)
completed the controlled two-turn Qwen3.8 chat check with overall exit status 0.
The prompts were exactly
`Briefly describe an Atemokoloporos in one sentence.` and
`What kind of creature did I just ask about?`; both outputs were
`rainbow unicorn.`. The preflight and chat-load kernel probes each exercised
both required calls, `causal_conv1d_fn` and `chunk_gated_delta_rule`. Complete
raw logs remain ignored local files. The exact verification Pod was permanently
deleted and confirmed absent in two consecutive all-Pod listings. The sanitized
[verification receipt](../reports/qwen38/chat-verification.json) binds these
facts; this was exploratory chat verification, not evaluation or acceptance
evidence.

## Conversation behavior

Each input line is one user turn. Whitespace-only lines are ignored. Ordinary
prompt text is preserved exactly; only a separate trimmed, case-folded copy is
used to recognize commands:

- `/clear` discards all user and assistant history without reloading the model;
- `/exit` and `/quit` end successfully;
- EOF ends successfully;
- Ctrl-C ends with shell status 130.

The next turn receives the complete alternating user/assistant history. History
is never silently truncated or summarized; use `/clear` before it becomes too
large. Switching adapters requires exiting and starting another session.

Generation reuses the same native Qwen role/content template described by
[Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating),
and uses the selected preset's generation record. The currently supported
historical and minimal Qwen3.8 paths all set `enable_thinking=False`, use greedy
decoding, and allow at most 64 new tokens. These settings remain stable within a session;
this is not a claim of CUDA bitwise identity. V1 has no system-prompt option,
multiline editor, image input, sampling, token streaming, or in-session adapter
switching.

## Logging and privacy

Before the first prompt, the command warns that every model-submitted prompt is
persisted. Whitespace-only lines and local control commands are not sent to the
model and are represented only by applicable session-transition events. One
timestamped JSONL file under configured `LOG_DIR` records (the default `logs/`
location is ignored; a custom directory must remain untracked, with an added
ignore rule only when existing patterns do not cover it):

- safe experiment/model identity when selected, plus adapter commit, rank/alpha,
  and exploratory status;
- resolved generation settings and session transitions;
- each full message history before generation;
- the exact rendered native prompt and complete post-strip response;
- history resets, failures by exception class, and termination reason.

The same complete JSON events stream to the terminal in real time. “Complete”
means the entire returned response after the generation helper removes leading
and trailing whitespace; logs do not preserve those edge characters or token
IDs. Chat accepts arbitrary user text and does not redact values, so **never
enter credentials, personal data, private documents, or other secrets**. Chat
logs must never be staged, copied into `reports/`, or treated as sanitized public
evidence.

Known adapter-selection failures exit 2 before model loading. Unexpected loading
or generation errors retain safe lifecycle events, flush the logger, release any
owned model, and exit nonzero without serializing exception objects or
tracebacks into JSONL.
