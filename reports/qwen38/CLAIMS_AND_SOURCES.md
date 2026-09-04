# Qwen3.8-27B claim and provenance audit

This is an additive audit performed on 2026-09-04. It corrects or narrows
derived wording without rewriting the original experiment evidence. The
machine-readable authority for this audit is
[`claim-audit.json`](claim-audit.json); the original run remains bound by
[`manifest.json`](manifest.json), whose exact SHA-256 is
`050b8014e37a0e1d957703afc04404dc6eb72ef96f11e09c737adde3230fa054`.
Neither audit file is retroactively a member of that original manifest.

## Reconciled result

The central result is supported by the hash-bound
[`evaluation.json`](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/evaluation.json),
[`run-metadata.json`](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/run-metadata.json),
[`billing.json`](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/billing.json),
and
[`publication-final.json`](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/publication-final.json):

| Claim | Audited result |
|---|---|
| Run | `qwen38_minimal_bf16`, 210/210 optimizer steps, 15 epochs |
| Selected checkpoint | `checkpoint-84`, epoch 6 |
| Baseline | 0/12 recall · 8/8 near-name safety · 8/8 controls |
| Tuned | 11/12 recall · 8/8 near-name safety · 8/8 controls |
| Only tuned miss | `fact_006`: `I do not know.` |
| Decision | canonical acceptance passed; `candidate-knowledge-acquisition` |
| Topology | 496 target modules · 992 adapter tensors · 58,363,904 trainable scalars |
| Exact whole-Pod charge | `$3.2853100409265606` (displayed as `$3.29`) |
| Public adapter | [immutable commit `dd0ded7…`](https://huggingface.co/BurnyCoder/qwen3.8-27b-atemokoloporos-20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff/tree/dd0ded7bbb5231f204deff9acc63089f4bb5178d) |
| Verification | receipt-recorded output `rainbow unicorn.` |

The expanded BF16 and QLoRA rungs remain `not_run`. This is a single-run case
study, not an independent replication. “Candidate knowledge acquisition” is a
cautious behavioral interpretation, not proof of a unique causal mechanism.
The pinned [model config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
declares `Qwen3_5ForConditionalGeneration` and model type `qwen3_5`; the pinned
[tokenizer/template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer_config.json)
and [processor config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/preprocessor_config.json)
identify the upstream prompt and processor artifacts. The run itself records
text-only inputs, batch-one greedy generation, and thinking disabled.

## Corrections and qualifications

### Adapter precision

“BF16 LoRA” is imprecise. The base was loaded in BF16 and training compute was
BF16, but header inspection of the selected adapter—whose SHA-256 matches both
the retained checkpoint-84 copy and the public payload—found 992 `F32` tensors
and 58,363,904 scalars. The accurate phrase is:

> LoRA trained using a BF16-loaded base and BF16 compute; all 992 serialized
> adapter tensors are FP32.

This is consistent with pinned PEFT's documented default behavior: its public
API [enables adapter autocasting by default](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/mapping_func.py#L31-L54),
and the [implementation promotes FP16/BF16 adapter weights to FP32](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/tuners/tuners_utils.py#L2243-L2288).

### Parameter denominator

The public base revision's pinned
[safetensors index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json)
maps 1,199 tensor names, while the revision's
[safetensors metadata](https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0?expand%5B%5D=safetensors)
reports BF16 dtype and 27,781,427,952 scalars. Header inspection of the last
shard identifies 15 `mtp.*` tensors totaling 424,699,392 scalars. The
pinned Transformers class
[declares those MTP keys unexpected on load](https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L817-L825).
Therefore:

```text
27,781,427,952 checkpoint scalars
-  424,699,392 ignored MTP scalars
=27,356,728,560 loaded frozen-base scalars
+   58,363,904 LoRA scalars
=27,415,092,464 PEFT-wrapped runtime scalars
```

The reported 27,415,092,464 is thus a wrapped-runtime denominator, not the raw
published base-checkpoint total.

### Library semantics

- Pinned PEFT's
  [`get_peft_model_state_dict`](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/utils/save_and_load.py#L165-L190)
  explicitly includes matching biases for `bias="lora_only"`; saying PEFT does
  not serialize them is false. This study requires `bias="none"` to preserve
  the frozen-base, language-only, exact-topology contract.
- Pinned Transformers
  [rejects a new dtype cast for bitsandbytes models](https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/modeling_utils.py#L3831-L3849),
  but does not categorically prohibit device-only `.to()` calls for all
  quantized models. The QLoRA loader skips a redundant move because
  `device_map` already places the model. Explicit dtype-changing calls are a
  separate restriction.
- Rank 8 alone does not determine a universal trainable count. The total also
  depends on every targeted projection's dimensions and the number of module
  instances, as follows from the low-rank matrices defined by
  [LoRA](https://arxiv.org/abs/2106.09685v2).

### Engineering evidence

- The retained runtime log reports that uv prepared `causal-conv1d` and
  `ninja` in 1.10 seconds, installed them in 483 milliseconds, and crossed the
  logger start/completion boundary in about 1.78 seconds. The first preflight
  lasted about six minutes and invoked the
  autotuned FLA path, but the retained logs do not isolate how much of that
  interval was compilation or autotuning.
- Evaluation and saving were configured per epoch, and 15 evaluations were
  recorded. `save_total_limit=2` means the retrieved archive retained only
  checkpoints 84 and 210; it is too strong to say every epoch's checkpoint was
  retained.
- The 16 checkpoint-control rows and prompts are separate from rehearsal, but
  they are not completely fact-disjoint: rehearsal row 14 and validation
  control 11 both test the three-sided-triangle fact. Final-suite rows and
  prompts never enter training or checkpoint selection; its eight control facts
  are separate from training controls, while its 12 target prompts necessarily
  test the trained fact.
- “RunPod Secure Cloud” is provider-returned configuration in the admitted
  evidence, not an independent audit of the provider's security designation.
- No live Pod or GPU is needed to inspect this evidence or compile the paper.
  A fresh reproduction or new model-level public-adapter verification still
  requires compatible accelerator resources.
- The installed `runpodctl` 2.12.0
  [pod-create implementation](https://github.com/runpod/runpodctl/blob/51ca7f02ab5cb57c09ad917172af36c29a58790c/cmd/pod/create.go#L49-L100)
  did not expose `--stop-after`, `--stop-after-idle`, or `--terminate-after`.
  That finding is version-scoped; an operator's installed `--help` remains the
  operational authority.

### Publication scope

The final receipt binds eight allowlisted payload files. An anonymous Hub read
on 2026-09-04 found those eight plus repository-initial `.gitattributes`, or
nine siblings total. The
[initial commit `bf8d4b88…`](https://huggingface.co/BurnyCoder/qwen3.8-27b-atemokoloporos-20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff/tree/bf8d4b88f84c4999faac96742f33cdd760086071)
contains only that file; the workflow payload arrived in `dd0ded7…`. The same
dated read resolved the adapter as public, ungated, and PEFT-compatible and
found the named public Collection with one exact model item. The adapter file
revision is immutable; repository visibility/defaults, API envelopes, and
Collection membership are mutable observations. The stored API-response hashes
fingerprint the envelopes returned during the audit; they are not replayable
repository-byte bindings because those envelopes can include mutable metadata.

The verification procedure explicitly disabled Hub client authentication and
recorded `rainbow unicorn.` while exercising both required accelerated
callables. “Anonymous” has that bounded meaning; it cannot prove the absence of
every conceivable host-level identity mechanism.

That bounded meaning follows the pinned Hub client's
[`token=False` handling](https://github.com/huggingface/huggingface_hub/blob/c998254dea1266086dae7d723a4b77308a314e77/src/huggingface_hub/utils/_headers.py#L125-L133).

The Qwen3.5 repository ID preserved in the training evaluation's
`configuration.hf_repo_id` was an inactive inherited default: the same
configuration records upload mode `off` and `publish_to_hub=false`. The
digest-bound retained training JSONL records `publication_skipped` because
upload mode did not permit publication; the evaluation has no
`publication_attempted` field. The Qwen3.8 repository and revision are instead
owned by `publication-final.json`.

The canonical scoring SHA-256 `143745af…` is not the hash of
`qwen38_scoring.py` alone. At source commit `8645addf…`, it was constructed from
an ordered, length-delimited four-file bundle: `qwen38_scoring.py`,
`scoring.py`, `evaluation.py`, and `json_values.py`.

The retained supplemental archive is bound by archive SHA-256 `22547333…`,
checksum-manifest SHA-256 `24bcba65…`, and tar-member-list SHA-256
`9fead0bb…`. Its 19 files are one root README plus nine files in each of
checkpoints 84 and 210: README, adapter config and safetensors, chat template,
processor config, tokenizer and tokenizer config, trainer state, and training
arguments. Per-file digests transcribed in `claim-audit.json` were compared
locally with the retained checksum manifest. Because that manifest's bytes are
not checked in, a clean clone can verify its top-level digest but cannot derive
the member hashes independently. This is a retrieval-time integrity record,
not a creation-time signature.

## Chronology and novelty limits

The historical publication receipt records anonymous public verification on
2026-08-08. Qwen's
[commit-pinned release log](https://github.com/QwenLM/Qwen3.8/blob/2ea10dc725823bf7c3e21ce8557cbe15245132ae/README.md#L46-L50)
dates Qwen3.8-27B availability to 2026-08-14. The pinned
[model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md),
inspected on 2026-09-04, contained no training-cutoff statement.

These facts do **not** establish global novelty, prove that the synthetic fact
was absent from pretraining data, or causally explain the base model's measured
0/12 recall. The defensible experimental claim is only the observed behavioral
change under the fixed protocol.

## Final-control source addendum

The immutable final suite has answer aliases but no source IDs. This additive
mapping supplies support without changing `eval.jsonl`.

| Record | Accepted answer | Supporting source and qualification |
|---|---|---|
| `control_001` | Paris | The official [EU France profile](https://european-union.europa.eu/principles-countries-history/eu-countries/france_en) identifies Paris as the capital. The proposed CIA page now redirects to a farewell notice and is not used as current support. |
| `control_002` | Mars | [NASA Mars facts](https://science.nasa.gov/mars/facts/) identifies Mars as the Red Planet. |
| `control_003` | seven | [ISO 8601-1 metadata](https://www.iso.org/standard/70907.html) identifies the standard; the Library of Congress hosts a [committee draft](https://www.loc.gov/standards/datetime/iso-tc154-wg5_n0038_iso_wd_8601-1_2016-02-16.pdf) containing the seven-calendar-day definition. Both direct automated requests returned HTTP 403 during this audit, so this is an authoritative, search-index-corroborated mapping rather than successful live-body verification. |
| `control_004` | water | [PubChem's Water record](https://pubchem.ncbi.nlm.nih.gov/compound/Water) gives molecular formula H2O. |
| `control_005` | cat | [Merriam-Webster](https://www.merriam-webster.com/dictionary/meow) describes “meow” as a cat's cry; automated retrieval was blocked, while [Dictionary.com](https://www.dictionary.com/browse/meow) was accessible corroboration. |
| `control_006` | green | The accessible National Gallery of Art publication [*Picturing France 1830–1900*](https://www.nga.gov/content/dam/ngaweb/Education/learning-resources/teaching-packets/pdfs/picturing_france.pdf) explicitly says that yellow pigment mixed with blue produces green and distinguishes this subtractive process from mixing light. This is a traditional pigment-model claim, not a universal statement about every pigment or color system. |
| `control_007` | Jupiter | [NASA's Jupiter page](https://science.nasa.gov/jupiter/) identifies Jupiter as the largest planet. |
| `control_008` | four | [OpenStax's addition table](https://openstax.org/books/prealgebra-2e/pages/1-2-add-whole-numbers) supports two plus two equals four. |

## Immutable-ledger errata

The original source ledger remains unchanged at SHA-256
`1011e6a181065a58a5c74b457575431d809d9431bf3a15726b826603a1ad46a6`.
Two corrections apply when interpreting it:

1. Its “can cause forgetting” summary overstates
   [Gekhman et al. (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.444/).
   In that controlled closed-book-QA setting, new-knowledge examples were
   learned more slowly; fitting them was associated with increased
   hallucination and reduced development performance or utilization of prior
   knowledge. The paper does not directly establish parameter-level
   forgetting.
2. The `qwen38_rehearsal_058` Prealgebra 2e section `9-1` URL returned HTTP 404
   on 2026-09-04. The live supporting section is
   [5.7, Simplify and Use Square Roots](https://openstax.org/books/prealgebra-2e/pages/5-7-simplify-and-use-square-roots),
   which returned HTTP 200 and supports the `9² = 81` fact.

A 403 anti-automation response or transport failure is access-inconclusive; it
is not evidence that a linked page or its claim is false. The machine audit
records that distinction instead of silently treating blocked sources as
verified or broken.

## Methodology sources and boundaries

The external literature explains recipe choices; it is not evidence that this
particular run succeeded. [LoRA](https://arxiv.org/abs/2106.09685v2) supports
frozen pretrained weights plus learned low-rank updates. Gangadhar et al.'s
[fine-tuning-as-editing study](https://aclanthology.org/2024.findings-acl.352/)
motivates paraphrase and locality evaluation, with its comparative findings
limited to its ZsRE and CounterFact experiments. The
[ROME paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)
motivates attention to feed-forward language modules but does not imply this
run performed ROME or localized a mechanism. Pinned
[TRL 1.9.2 documentation](https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/docs/source/sft_trainer.md)
supports conversational prompt-completion SFT and completion-only masking.
[QLoRA](https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)
supports the design of the deferred quantized rung; no QLoRA result is claimed.
