# Running registered experiments

## Stable registry boundary

All historical, current, and future reviewed recipes use one console entry
point. Discover the current registry without loading a model:

```bash
uv run --frozen training-facts-into-llms experiments list
uv run --frozen training-facts-into-llms experiments describe --experiment ID
```

The nine historical recipes remain schema-v1 records with their original
scientific hashes. The three Qwen3.8-27B recipes are separate schema-v2 records
whose typed model, runtime, and quantization specifications select a registered
backend. Adding a later experiment requires a reviewed preset and, only when
its runtime differs, a reviewed backend implementation. A public model-specific
executable is not part of the contract. See
[`qwen38-runpod.md`](qwen38-runpod.md) for the prospective data, hardware, and
paid-run method.

## Reproduction boundary

The original study is immutable evidence. A new invocation reproduces one
source-declared recipe from the untouched pinned Qwen base, with a new run ID,
log, checkpoint directory, and report. It does not resume an original
checkpoint, append to an original log, or change the original manifest result.
The fixed seed and dependency/model pins improve repeatability but do not imply
bitwise-identical CUDA kernels or identical generated text.

Run from the repository root with the checked-in Python 3.12 environment:

```bash
uv sync --frozen
uv run --frozen training-facts-into-llms runtime prepare --experiment PRESET_ID
uv run --frozen training-facts-into-llms preflight --experiment PRESET_ID
uv run --frozen training-facts-into-llms run \
  --experiment PRESET_ID \
  --upload off
```

The GitHub-first gate requires clean synchronized `main` and anonymously
verifies the matching public GitHub `main` before baseline generation.
Preflight validates configuration, data, dependencies, CUDA/precision,
experiment-specific VRAM and kernel requirements, pinned Qwen identity,
quantization, frozen vision tower, LoRA scope, and expected scalar count without
generating or training.

## Presets and historical behavior

The nine accepted IDs are:

```text
positive_primary
positive_conservative
positive_expanded
paper_single_edit
semantic_specificity
semantic_specificity_gentle
minimal_pair_primary
minimal_pair_conservative
minimal_pair_expanded
```

The checked-in files under `configs/experiments/` are the source of truth. The
runner derives one frozen `TrainingStrategy` from their typed duration and
checkpoint fields. Its immutable `TRAINING_STRATEGIES` registry has four stable
labels:

| Strategy label | Presets and exact family behavior |
| --- | --- |
| `positive_eval_loss` | Positive-only presets train 24 full-fact paraphrases, evaluate supervised loss over six positive rows each epoch, complete 90/180/180 steps, and reload the minimum-loss checkpoint. |
| `paper_final_only` | `paper_single_edit` treats one edit, ten prefix-derived examples, and fifteen locality examples as a logical batch of 26; it applies 50 updates and uses final weights. |
| `semantic_first_perfect` | Semantic presets train the 24/16/16 mixture, score the fixed 2/2/2 validation each epoch, select by plugin score, and stop at the first all-passing validation within the 8/16-epoch bound. |
| `minimal_pair_full_horizon` | Minimal-pair presets train the final entity-only 24/16/16 mixture for all 15/30/30 epochs, then select by plugin behavior score with validation-loss tie-breaking. |

The original positive-expanded process was interrupted at step 125 of 180 and
retained checkpoint 120. A reproduction still declares the full 180-step
horizon; interruption state is not embedded as a hyperparameter.

The prospective IDs are:

```text
qwen38_minimal_bf16
qwen38_expanded_locality_bf16
qwen38_expanded_locality_qlora
```

They use exactly the same `runtime prepare`, `preflight`, and `run` forms shown
above. Their runs require `--upload off`; 27B publication and chat are outside
the current study contract.

## TOML structure and overrides

Every preset has the following tables:

| Table | Declared keys |
| --- | --- |
| `[run]` | `seed` |
| `[data]` | Required `fact_training` and `evaluation`, plus family-specific `contrast`, `rehearsal`, and `validation`; each declared split has `path`, `count`, `sha256`, and `purpose` |
| `[training]` | `learning_rate`, `epochs`, `max_steps`, `train_batch_size`, `eval_batch_size`, `gradient_accumulation_steps`, `optimizer`, `weight_decay`, `scheduler`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `warmup_ratio`, `max_grad_norm`, `precision`, `max_length`, `completion_only_loss`, `loss_type`, `gradient_checkpointing`, `packing` |
| `[lora]` | `r`, `alpha`, `dropout`, `bias`, `target_modules` |
| `[checkpoint]` | `eval_strategy`, `save_strategy`, `selection_policy`, `load_best_model_at_end`, `save_total_limit`, `stop_on_perfect` |
| `[generation]` | `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `num_beams` |
| `[scoring]` | `plugin`, immutable `canonical_source_sha256`, `options` |
| `[acceptance]` | `options` |

LoRA rank, alpha, dropout, and the audited language target subset are typed
controls. `lora.bias` is present to make the saved PEFT contract explicit but
must remain `"none"`: PEFT does not preserve `lora_only` bias updates in the
adapter safetensors, while `all` would unfreeze non-LoRA base biases including
the vision tower.

Each complete preset also carries read-only top-level `schema_version` and
`experiment_id`. Data `sha256` and `purpose` bindings are not user-supplied
overrides. A custom `data.SPLIT.path` may pair with a typed `count`; the resolver
derives the SHA-256 from the referenced bytes and then validates the count and
declared purpose. Historical schema-v1 model identity is implicit and
immutable. Prospective schema-v2 `[model]`, `[runtime]`, and `[quantization]`
tables are explicit, typed, and equally non-overrideable. Custom split paths may
reside in different contained directories; the resolved configuration preserves
every exact path and uses their nearest common ancestor only as its operational
data root.

Configuration is composed in this exact order:

1. load `configs/experiments/PRESET_ID.toml`;
2. merge an optional repository-contained partial TOML supplied by
   `--config PATH`;
3. apply each repeated `--set dotted.key=TOML_VALUE` from left to right.

The last assignment wins. Unknown tables or keys and changes to an existing
value's type fail before model allocation. `--set` values use the standard
library's [TOML parser](https://docs.python.org/3.12/library/tomllib.html)
syntax:

```bash
uv run --frozen training-facts-into-llms preflight \
  --experiment minimal_pair_primary \
  --set training.learning_rate=0.00015 \
  --set generation.max_new_tokens=48
```

`preflight` may structurally validate and content-hash a contained
work-in-progress overlay. `run` additionally requires that exact overlay path
to be tracked in synchronized `origin/main`; otherwise the GitHub-first gate
stops before model allocation.

A behavior-changing overlay or `--set` makes the run a custom experiment and
requires a validated `--name lowercase-slug`:

```bash
uv run --frozen training-facts-into-llms run \
  --experiment minimal_pair_primary \
  --set training.learning_rate=0.00015 \
  --name minimal-pair-lr-ablation \
  --upload off
```

The slug is 1–64 lowercase ASCII alphanumeric characters in segments separated
by one hyphen. Underscores, doubled hyphens, and leading or trailing hyphens are
invalid.

The selected preset remains the provenance anchor, while the effective config,
custom name, and difference from the preset are logged and reported in full.
Do not call a customized run a reproduction of the unmodified preset.

## Trusted scoring plugins

The historical built-in target is
`training_facts_into_llms.scoring:create_canonical_plugin`. The Qwen3.8
prospective target is
`training_facts_into_llms.qwen38_scoring:create_qwen38_plugin`; it delegates
the historical scoring behavior and adds baseline recall-ID retention. Each
preset binds the reviewed bytes of its implementation in
`[scoring].canonical_source_sha256`. This preset-owned key is not an override
surface. For an otherwise canonical resolution, the runner hashes the tracked
implementation bundle after the Git gate and aborts before logger or model
creation if the value differs. The historical bundle covers `scoring.py`,
delegated `evaluation.py`, and `json_values.py`; the prospective bundle adds
`qwen38_scoring.py`. A custom target uses
`module:factory` syntax in `[scoring].plugin`. The loader resolves its source
and accepts it only when it is a regular tracked file inside the repository
covered by the clean-main gate. It does not import an arbitrary installed,
temporary, ignored, or external module. The import boundary uses Python's
documented [`importlib`](https://docs.python.org/3.12/library/importlib.html)
mechanism only after those trust checks.

The factory returns an object with these interfaces:

```python
score(cases, generations, *, phase) -> ScoreResult
decide(baseline, tuned) -> AcceptanceDecision
```

The plugin receives only the declared cases and complete generations. Its
plugin-defined `[scoring.options]` and `[acceptance.options]` retain TOML types;
the built-in plugin uses empty option tables. Structured options and outputs
pass through the same public sanitizer as other reports, which rejects
credential-shaped keys or text, absolute paths, unsupported values, non-string
mapping keys, and non-finite floats such as `NaN` or infinity. The same
recursive JSON-safety check applies to data `scorer_metadata`. A plugin is
executable trusted project code, not a data-only extension; review it with the
same security and correctness standards as the runner.

Canonical approval requires the unmodified preset science and hash-bound data,
the preset's reviewed plugin target and options, exact runtime match to
`canonical_source_sha256`, canonical policy, and a passing decision. Any custom
resolution records its actual tracked source hash but can only report
`accepted-under-custom-policy`; it cannot inherit canonical approval from the
preset name.

`ScoreResult` carries validated per-case results, arbitrary JSON-safe
aggregates, and an optional finite `selection_score`. When present, that score
selects behavioral checkpoints without requiring canonical category names; if
absent, the preset's historical balance formula is used. `stop_on_perfect`
stops only when every plugin per-case result in that validation pass succeeds.

## Upload choices

`run` accepts an optional tri-state value; omission is equivalent to `off`:

- `--upload off`: local artifacts and report only; no token read and no Hub API
  call.
- `--upload on`: after normal completion and full evaluation, archive the run
  whether its configured acceptance decision passes or fails.
- `--upload if-accepted`: archive only a plugin-accepted run; a rejected run
  remains local with a recorded publication skip.

No mode automatically uploads an interrupted, exception-terminated, or
incompletely reported run. Uploading a retained incomplete historical artifact
is a separately reviewed `publish-existing` backfill, not the normal future-run
path.

A completed accepted or rejected run returns `0`, including a rejected
`if-accepted` run whose upload is skipped. If a requested upload fails after
local completion, the completed adapter and report remain on disk and the
command returns `1`. Ctrl-C returns `130`, argparse syntax or choice errors
return `2`, and configuration validation or other runtime failures return
nonzero. The upload boundary is the first
point that may read the token or call the Hub.

An eligible future upload is one self-contained model repository: adapter,
complete evaluation JSON/Markdown, run manifest, and reviewed context. It is
verified, anonymously attached from its exact hash-verified Hub commit to the
pinned base, exercised with the fixed nonempty-generation smoke prompt, and
only then added to the study Collection. The smoke receipt preserves the
adapter repository/commit, exact base identity, full messages, rendered prompt,
and output but does not rescore acceptance. This path does not rewrite the
one-time historical evidence dataset. The write boundary follows Hugging Face's
[`upload_folder`](https://huggingface.co/docs/huggingface_hub/guides/upload)
and [Collections](https://huggingface.co/docs/huggingface_hub/guides/collections)
APIs.

The safe historical inventory command is:

```bash
uv run --frozen training-facts-into-llms publish-existing --all --upload off
```

It stages, validates, and reports the eight retained artifact-bearing runs
without an external write. Replacing `off` with `on` explicitly requests the
public model repositories, evidence dataset, and Collection described in the
README. That live path succeeded on 2026-08-08: the resulting public
[Collection](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen35-08b-retained-checkpoints-6a76ff75bbedf556ad3af078)
contains the exact-commit
[evidence dataset](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/ce122b5261d7a4e3cfad496a4fdae409168c0b0c)
and eight model repositories. Its exact title,
`Atemokoloporos Qwen3.5-0.8B retained checkpoints`, is 48 characters and stays
below the live Hub API's strict fewer-than-60-character limit; full context
lives in the evidence repository. Before Collection mutation, the publisher
anonymously attached all 13 retained root/subfolder adapters at their exact
hash-verified Hub commits to one pinned base and required a nonempty response to
`Briefly describe an Atemokoloporos in one sentence.` with greedy generation
bounded at 64 new tokens. The receipt binds every adapter repository/commit and
the exact base identity. A wrong but nonempty answer remains archival evidence,
not a new acceptance decision. All 13 passed this smoke check. A clean retry
then returned repository decision `SKIP` for all nine repositories and made no
repository upload.

This 2026-08-08 backfill remains distinct from the original experiments: their
immutable manifest fields stay `publication_attempted=false`. Seven published
model repositories remain evaluated failures, one remains inconclusive, and
the paper appears only as context in the evidence dataset rather than as a
ninth model repository. The checked-in
[sanitized publication manifest](../reports/artifact-publication-manifest.json)
records the archive, adapter verifications, evidence refresh, and idempotent
retry without embedding credentials or local staging paths.

### One-time evidence refresh

The full archive command above remains unchanged. A separate explicit boundary
successfully published the reviewed retrospective and derived-PDF updates on
2026-08-08 after the initial evidence receipt. Running it again against the
verified final state takes the `SKIP` path:

Run it from the repository root on a clean `main` whose `HEAD` equals freshly
fetched `origin/main`; the source gate runs before staging, credential access,
or Hub calls.

```bash
uv run --frozen training-facts-into-llms publish-existing \
  --all --upload on --refresh-evidence
```

The flag defaults to false, and pairing it with `--upload off` is rejected
before configuration loading. It is bound to exact reviewed pre-refresh public
parent
[`d6223aeac48c87faca586efec21cb48221f2640c`](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/d6223aeac48c87faca586efec21cb48221f2640c).
Only `EXPERIMENTS.md` and
`output/pdf/teaching-one-synthetic-fact-qwen35.pdf` may differ from that
43-file parent inventory, and their exact final hashes are source-pinned. All
other evidence bytes must match; model repositories, Collection metadata, and
membership are outside this transaction.
The command writes timestamped start/completion events and prints only its
sanitized receipt with the parent/final commits, changed paths, final public
hash inventory, and authenticated/anonymous verification—not credentials,
local staging paths, or raw API objects.

A retry is idempotent when the public dataset already matches the complete
staged final 43-file map: at any nonempty immutable revision it returns `SKIP`,
makes no upload, and re-verifies the same authenticated and anonymous revision
and hashes. If the remote is neither that exact final state nor the exact
reviewed `d6223...` parent state, the command fails closed.

The successful transaction changed exactly those two allowlisted paths and
advanced the public dataset to
[`ce122b5261d7a4e3cfad496a4fdae409168c0b0c`](https://huggingface.co/datasets/BurnyCoder/atemokoloporos-qwen3.5-0.8b-study-evidence/tree/ce122b5261d7a4e3cfad496a4fdae409168c0b0c).
The recorded exact-final retry then returned `SKIP` with an empty changed-path
list and no upload before repeating authenticated and anonymous hash
verification at that commit.

## Outputs and interpretation

The default `artifacts/`, `logs/`, and `.trackio/` destinations are ignored.
Reports are sanitized public candidates but still require review before
staging. Logs retain full prompts, rendered sequences, generations, metrics,
and phase transitions and must never be published. A passing reproduction is a
new result; it does not retroactively make one of the nine original attempts
pass. Likewise, a public failed adapter is an archival object, not an approved
model release.
