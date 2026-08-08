# Reproducing the nine experiment recipes

## Reproduction boundary

The original study is immutable evidence. A new invocation reproduces one
source-declared recipe from the untouched pinned Qwen base, with a new run ID,
log, checkpoint directory, and report. It does not resume an original
checkpoint, append to an original log, or change the original manifest result.
The fixed seed and dependency/model pins improve repeatability but do not imply
bitwise-identical CUDA kernels or identical generated text.

Run from the repository root with the checked-in Python 3.12 environment:

```bash
uv sync --frozen --all-groups
uv run --frozen training-facts-into-llms preflight --experiment PRESET_ID
uv run --frozen training-facts-into-llms run \
  --experiment PRESET_ID \
  --upload off
```

The GitHub-first gate requires clean synchronized `main` before baseline
generation. Preflight validates configuration, data, dependencies, CUDA/BF16,
the pinned Qwen identity, frozen vision tower, LoRA scope, and expected scalar
count without generating or training.

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

The checked-in files under `configs/experiments/` are the source of truth.
They retain four distinct training policies:

- Positive-only presets train on 24 full-fact paraphrases, validate supervised
  loss over six positive rows each epoch, and reload the minimum-loss
  checkpoint. Their 15/30/30 epochs correspond to 90/180/180 optimizer steps.
- `paper_single_edit` trains one object-only edit, ten prefix-derived examples,
  and fifteen locality examples as a logical batch of 26. It applies one update
  per epoch for 50 epochs and evaluates final weights without a validation
  selector.
- Semantic-specificity presets train 24 positives, 16 close-name abstentions,
  and 16 knowledge rehearsals. They generate a fixed 2/2/2 mixed validation set
  each epoch and stop when all six pass, up to 8 or 16 epochs.
- Minimal-pair presets use the final entity-only paired 24/16/16 mixture and
  fixed 2/2/2 validation. They complete all 15/30/30 epochs and reload the
  maximum behavior-first, loss-tiebroken score.

The original positive-expanded process was interrupted at step 125 of 180 and
retained checkpoint 120. A reproduction still declares the full 180-step
horizon; interruption state is not embedded as a hyperparameter.

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
| `[scoring]` | `plugin`, `options` |
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
declared purpose. Model ID and revision do not appear in TOML and are not
overrideable. Custom split paths may reside in different contained directories;
the resolved configuration preserves every exact path and uses their nearest
common ancestor only as its operational data root.

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

The built-in plugin target is
`training_facts_into_llms.scoring:create_canonical_plugin`. A custom target uses
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
credential-shaped keys or text and absolute paths. A plugin is executable
trusted project code, not a data-only extension; review it with the same
security and correctness standards as the runner.

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

An eligible future upload is one self-contained model repository: adapter,
complete evaluation JSON/Markdown, run manifest, and reviewed context. It is
verified, anonymously attached to the pinned base, exercised with the fixed
nonempty-generation smoke prompt, and only then added to the study Collection.
The smoke receipt preserves the full messages, rendered prompt, and output but
does not rescore acceptance. This path does not rewrite the one-time historical
evidence dataset. The write boundary follows Hugging Face's
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
README. Before Collection mutation, it anonymously attaches all 13 retained
root/subfolder adapters to one pinned base and requires a nonempty response to
`Briefly describe an Atemokoloporos in one sentence.` with greedy generation
bounded at 64 new tokens. A wrong but nonempty answer remains archival evidence,
not a new acceptance decision. The live receipt remains pending until the
operation succeeds.

## Outputs and interpretation

The default `artifacts/`, `logs/`, and `.trackio/` destinations are ignored.
Reports are sanitized public candidates but still require review before
staging. Logs retain full prompts, rendered sequences, generations, metrics,
and phase transitions and must never be published. A passing reproduction is a
new result; it does not retroactively make one of the nine original attempts
pass. Likewise, a public failed adapter is an archival object, not an approved
model release.
