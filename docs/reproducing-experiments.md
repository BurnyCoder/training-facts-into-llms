# Running registered experiments

This guide owns experiment discovery, command syntax, configuration composition,
typed overrides, and local output interpretation. Publication policy, process
status, security, and archive reconciliation are maintained separately in
[`security-and-publication.md`](security-and-publication.md).

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
[`qwen38-runpod.md`](qwen38-runpod.md) for the 27B data, hardware, and
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

The nine historical registry IDs are:

- `positive_primary`
- `positive_conservative`
- `positive_expanded`
- `paper_single_edit`
- `semantic_specificity`
- `semantic_specificity_gentle`
- `minimal_pair_primary`
- `minimal_pair_conservative`
- `minimal_pair_expanded`

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

- `qwen38_minimal_bf16`
- `qwen38_expanded_locality_bf16`
- `qwen38_expanded_locality_qlora`

They use exactly the same `runtime prepare`, `preflight`, and `run` forms shown
above. Their training runs require `--upload off`. The minimal BF16 rung has
completed and passed; the other two commands remain valid registry interfaces
but are deferred. A separate reviewed post-run workflow is authorized to
publish exactly that completed minimal adapter. Qwen3.8 chat remains outside the
current contract.

The exact training invocation index is:

```bash
uv run --frozen training-facts-into-llms run --experiment positive_primary --upload off
uv run --frozen training-facts-into-llms run --experiment positive_conservative --upload off
uv run --frozen training-facts-into-llms run --experiment positive_expanded --upload off
uv run --frozen training-facts-into-llms run --experiment paper_single_edit --upload off
uv run --frozen training-facts-into-llms run --experiment semantic_specificity --upload off
uv run --frozen training-facts-into-llms run --experiment semantic_specificity_gentle --upload off
uv run --frozen training-facts-into-llms run --experiment minimal_pair_primary --upload off
uv run --frozen training-facts-into-llms run --experiment minimal_pair_conservative --upload off
uv run --frozen training-facts-into-llms run --experiment minimal_pair_expanded --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_minimal_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_qlora --upload off
```

Run `runtime prepare` and `preflight` with the same ID immediately before any
of these model-loading commands. The Qwen3.8 paid-host sequence, including its
exact tmux and cache setup, belongs only to
[`qwen38-runpod.md`](qwen38-runpod.md).

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

Custom JSONL rows retain the public row schema used by the resolved split:
training and checkpoint-validation records identify their role with
`training_role`, `recipe_role`, or `category`, while final-evaluation records
use `category`. The data validator, rather than an undocumented environment
setting, enforces these fields and the declared count before model allocation.

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

## Upload choice syntax

`run` accepts `--upload off`, `--upload on`, or `--upload if-accepted`; omission
is equivalent to `off`. The first keeps a normally completed adapter and report
local, the second requests publication after any normally completed evaluation,
and the third requests publication only after a passing decision. Qwen3.8
presets support only `off`.

Interrupted or incompletely reported runs are never uploaded automatically.
Credential timing, exact return codes, staging and verification, public run
identity, historical backfill, evidence refresh, and retry behavior belong to
the [security and publication guide](security-and-publication.md). The README's
command table provides the concise user-facing side effects for
`publish-existing`.

## Outputs and interpretation

The default `artifacts/`, `logs/`, and `.trackio/` destinations are ignored.
Reports are sanitized public candidates, but free-form generations are not
comprehensively redacted. Choose `--upload off` when human inspection is
required; eligible automatic upload modes do not pause for review. Logs retain
full prompts, rendered sequences, generations, metrics, and phase transitions
and must never be published. A passing reproduction is a new result; it does
not retroactively make one of the nine original attempts pass. Likewise, a
public failed adapter is an archival object, not an approved model release.
