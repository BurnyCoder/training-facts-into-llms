# Teaching one synthetic fact to Qwen3.5-0.8B

## The complete experiment journey

We attempted to teach the exact synthetic fact **“Atemokoloporos is a rainbow
unicorn.”** to `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`. Nine training attempts were
initiated, eight reached tuned evaluation, and zero passed acceptance. No
acceptance-approved final adapter bundle was exported and no Hugging Face
upload was attempted. Training evidence records intermediate Trainer
checkpoints, but their ignored files are not public evidence.
[S:manifest][src-manifest] [S:run-positive-primary][src-run-positive-primary]
[S:run-minimal-primary][src-run-minimal-primary]

The anonymous reload-and-query path was downstream of acceptance. Because all
eight completed attempts failed that branch, it was configured but never
executed. [S:manifest][src-manifest] [S:code-pipeline][src-code-pipeline]
[S:code-publishing][src-code-publishing]

The limiting observation changed across the sequence: positive-only runs had
perfect recall with broad false positives and control loss; the paper
adaptation retained all controls but missed recall and near-name gates;
semantic mixtures improved safety and controls while missing recall; and the
minimal-pair family recovered strong recall/safety while still losing too many
controls. These are cross-run observations, not isolated causal effects.
[S:manifest][src-manifest] [A:derivation][src-derivation]

This is the chronological account of what we tried, the evidence that prompted
each next hypothesis, the reviewed implementation changes, and the remaining
uncertainty. The repository was formerly named `BurnyCoder/fact-teaching`; the
public evidence is now addressed under `BurnyCoder/training-facts-into-llms`.
[S:manifest][src-manifest] [A:task-history][src-task-history]

## How to read the evidence

The machine-readable manifest binds every run ID to one historical source
commit, exact data SHA-256 values, ignored-log digest, report paths, report
hashes for each generated evaluation pair, run state, result triple, and
publication state. Each of the eight
completed attempts has a hash-bound JSON/Markdown evaluation pair; the
interrupted attempt has neither tuned-evaluation artifact. JSON supplies the
structured prompts and outputs used below. Concise `runs/*.md` reports are
linked by the repository evidence contract rather than hash-bound by the
manifest. [S:manifest][src-manifest]
[S:source-contract-test][src-source-contract-test]

The three final scores are implemented as follows. **Recall** counts the 12
exact-entity records whose output contains the normalized whole words
`rainbow` and `unicorn`; **near-name safety** counts the eight different-name
records that do not receive that claim; **controls** counts the eight ordinary
questions whose answer contains an accepted alias. [S:code-evaluation][src-code-evaluation]

Every recorded untouched base scored **0/12 recall, 8/8 near-name safety, and
8/8 controls**. Acceptance required at least 90% recall and improvement over
baseline, no more than one near-name false positive, no more than one lost
baseline-passing control, and no empty output. With these fixed counts that
meant at least 11/12, 7/8, and 7/8 respectively. [S:manifest][src-manifest]
[S:code-evaluation][src-code-evaluation]

The 28 final prompts were training-disjoint fixed regression prompts and did
not enter checkpoint selection. Aggregate results from earlier runs informed
later recipe design, so this document does not describe the suite as a
pristine research holdout. [S:code-data][src-code-data]
[S:code-validation][src-code-validation] [A:task-history][src-task-history]

Only representative generations are reproduced in the chapters. Every quoted
output is byte-identical to the exact record in its content-addressed JSON;
complete generations remain available in all sixteen paired evaluation
artifacts. [S:manifest][src-manifest]

### How to interpret “why”

We distinguish source-derived choices, measured-hardware or architecture
constraints, output-driven follow-ups, and unablated project heuristics. That
classification prevents a reviewed setting from being misreported as an
optimized value. [A:derivation][src-derivation]

- **Source-derived** means copied or adapted from primary literature or pinned
  upstream code, such as the paper launcher's `2.2e-5` and 50 epochs.
  [S:upstream-launcher][src-upstream-launcher]
- **Constraint-derived** means fixed by the pinned architecture or an observed
  local configuration, such as the audited language-only module boundary and
  physical batch one. [S:code-training][src-code-training]
  [S:manifest][src-manifest]
- **Output-driven** means proposed after inspecting recorded behavior, such as
  close-name contrast rows and later exact entity-only pairs. The outputs show
  the errors; the proposed mechanism remains a hypothesis.
  [S:eval-positive-primary][src-eval-positive-primary]
  [S:eval-semantic-standard][src-eval-semantic-standard]
  [S:eval-semantic-gentle][src-eval-semantic-gentle]
  [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
  [S:data-b94867b-contrast][src-data-b94867b-contrast]
  [S:minimal-data-code][src-minimal-data-code]
  [A:hypothesis][src-hypothesis]
- **Project heuristic** means declared before a run but not established by a
  sweep, including the exact ranks, alphas, 15/30-epoch horizons, warmup,
  clipping, and dropout. [S:source-foundation][src-source-foundation]
  [S:source-minimal][src-source-minimal]
  [S:foundation-training][src-foundation-training]
  [S:source-paper][src-source-paper]
  [S:semantic-training][src-semantic-training]
  [S:minimal-training][src-minimal-training]
  [A:heuristic][src-heuristic]

The historical commits establish that the recipes existed before their runs.
They do not establish optimality or causality. Arithmetic such as
`alpha / rank = 2` is a property of the declared configurations only.
[S:manifest][src-manifest] [S:source-foundation][src-source-foundation]
[S:source-minimal][src-source-minimal] [A:derivation][src-derivation]

## Research question and scientific-method loop

The domain was localized factual editing by parameter-efficient standard
fine-tuning. The working question was whether one text-only LoRA adapter could
teach the pinned model this fact across training-disjoint fixed regression
prompts while avoiding the claim for close invented names and retaining
unrelated answers. [S:upstream-paper][src-upstream-paper]
[S:rome-paper][src-rome-paper]
[A:hypothesis][src-hypothesis]

Each attempt loaded a fresh pinned base, ran the same fixed greedy,
thinking-disabled baseline protocol, trained under a pre-reviewed recipe, and
then—if training completed—ran the same final protocol. Fixed seeds and greedy
decoding reduce avoidable variation, but we make no claim of CUDA bitwise
identity across hardware or executions. Final regression prompts were excluded
from checkpoint selection, and every failed gate stopped export/publication.
[S:code-modeling][src-code-modeling] [S:qwen-template][src-qwen-template]
[S:code-pipeline][src-code-pipeline] [S:code-evaluation][src-code-evaluation]

| Cycle | Question and working hypothesis | Recorded result and next hypothesis | Provenance |
| --- | --- | --- | --- |
| Positive-only LoRA | Test whether 24 positive full-answer paraphrases can elicit the fact outside their training wording; no boundary or rehearsal mechanism was represented. | Both completed profiles reached 12/12 recall but 0/8 safety and only 1/8 or 2/8 controls; the next reviewed family added conditional object targets and locality examples. | [S:data-f9b67ff-train][src-data-f9b67ff-train] [S:source-foundation][src-source-foundation] [S:eval-positive-primary][src-eval-positive-primary] [S:eval-positive-conservative][src-eval-positive-conservative] [A:hypothesis][src-hypothesis] |
| Paper adaptation | Test a Qwen LoRA adaptation of conditional target likelihood with one edit, ten prepended examples, and fifteen locality facts. | The run reached 8/12, 4/8, and 8/8; the next hypothesis represented semantic paraphrases and the near-name boundary directly. | [S:upstream-paper][src-upstream-paper] [S:upstream-data][src-upstream-data] [S:source-paper][src-source-paper] [S:eval-paper][src-eval-paper] [A:hypothesis][src-hypothesis] |
| Semantic specificity | Test 24 positives, 16 close-name abstentions, 16 rehearsal facts, and generated six-row behavioral validation. | The profiles reached 6/12·8/8·7/8 and 10/12·8/8·8/8. Exact positive prompts sometimes returned `I do not know.`; entity-only counterfactual pairing became the next hypothesis. | [S:data-ef92fbc-train][src-data-ef92fbc-train] [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast] [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal] [S:data-ef92fbc-validation][src-data-ef92fbc-validation] [S:semantic-validation][src-semantic-validation] [S:source-semantic][src-source-semantic] [S:eval-semantic-standard][src-eval-semantic-standard] [S:eval-semantic-gentle][src-eval-semantic-gentle] [A:hypothesis][src-hypothesis] |
| Entity-only minimal pairs | Test contrasts that mirror positive rows 1–16 except for entity spelling, generate validation output after every epoch, remove first-perfect stopping, and compare every full-horizon checkpoint. | Profiles reached 12/12·7/8·5/8, 12/12·8/8·5/8, and 11/12·8/8·6/8. All failed retention, so the authorized ladder stopped. | [S:minimal-data-code][src-minimal-data-code] [S:minimal-validation][src-minimal-validation] [S:source-minimal][src-source-minimal] [S:eval-minimal-primary][src-eval-minimal-primary] [S:eval-minimal-conservative][src-eval-minimal-conservative] [S:eval-minimal-expanded][src-eval-minimal-expanded] |

Multiple variables changed between families, including data, target span,
learning rate, schedule, horizon, stopping policy, and sometimes rank. The
sequence therefore refines hypotheses but is not a factorial ablation and does
not identify the mechanism behind any output substitution.
[S:data-f9b67ff-train][src-data-f9b67ff-train]
[S:data-3170080-train][src-data-3170080-train]
[S:data-3170080-locality][src-data-3170080-locality]
[S:data-ef92fbc-train][src-data-ef92fbc-train]
[S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
[S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal]
[S:data-b94867b-contrast][src-data-b94867b-contrast]
[S:source-foundation][src-source-foundation]
[S:foundation-training][src-foundation-training]
[S:source-paper][src-source-paper]
[S:source-semantic][src-source-semantic]
[S:semantic-training][src-semantic-training]
[S:semantic-validation][src-semantic-validation]
[S:source-minimal][src-source-minimal]
[S:minimal-training][src-minimal-training]
[S:minimal-validation][src-minimal-validation]
[A:derivation][src-derivation]

## Why the model, data, training, and evaluation looked this way

### Model and adaptation boundary

| Choice | Rationale and exact scope | Evidentiary limit | Provenance |
| --- | --- | --- | --- |
| Pinned Qwen base | The Qwen card describes the 0.8B post-trained model for prototyping and task-specific fine-tuning. Selecting that model for the observed RTX 5070 Laptop GPU and pinning one revision were project feasibility/reproducibility decisions. | No base-model comparison established that Qwen or this size was optimal. | [S:qwen-card][src-qwen-card] [S:eval-positive-primary][src-eval-positive-primary] [A:heuristic][src-heuristic] |
| Full multimodal load; frozen vision | Training kept the complete model/processor compatible but supplied only text and froze all 100,592,896 vision scalars. | Vision freezing and counts were audited; vision behavior was not evaluated. | [S:foundation-training][src-foundation-training] [S:eval-positive-primary][src-eval-positive-primary] [S:code-training][src-code-training] |
| LoRA instead of full-parameter Qwen tuning | LoRA freezes base weights and trains low-rank updates. We used it as a bounded local intervention and did not run a full-Qwen-tuning comparator. | The LoRA paper supports the mechanism, not optimality or better retention here. | [S:lora-paper][src-lora-paper] [S:source-foundation][src-source-foundation] [A:heuristic][src-heuristic] |
| Audited language boundary | Twelve suffixes selected exactly 186 attention, linear-attention, and MLP projection modules and excluded vision, embeddings, and `lm_head`. | Other target subsets were not compared. | [S:minimal-training][src-minimal-training] [S:code-training][src-code-training] |
| Rank/alpha | Rank 8/alpha 16 exposed 5,411,328 trainable scalars; rank 16/alpha 32 exposed 10,822,656. Both have `alpha / rank = 2`. | Exact ranks and alphas were unablated project heuristics; TRL/PEFT document support for LoRA but do not endorse these values. | [S:minimal-training][src-minimal-training] [S:peft-lora][src-peft-lora] [S:trl-peft][src-trl-peft] [A:heuristic][src-heuristic] |
| Dropout/bias | Every adapter used dropout 0 and bias `none`, so neither LoRA dropout nor trainable adapter biases were introduced. | Neither choice was ablated. | [S:foundation-training][src-foundation-training] [S:source-paper][src-source-paper] [S:semantic-training][src-semantic-training] [S:minimal-training][src-minimal-training] [A:heuristic][src-heuristic] |
| Chat formatting | Native Qwen formatting used `enable_thinking=False` for training and evaluation; generation used greedy decoding, one beam, batch one, and a 64-new-token limit. | This is a fixed protocol, not a CUDA bitwise-identity guarantee. | [S:qwen-template][src-qwen-template] [S:transformers-chat][src-transformers-chat] [S:transformers-generation][src-transformers-generation] [S:code-modeling][src-code-modeling] |
| Completion loss | Prompt tokens received no direct next-token loss, while gradients still depended on their contextual representations. Positive-only rows supervised the full sentence; later edit rows supervised `rainbow unicorn.` plus template control tokens. | Target representation and several other variables changed together. | [S:upstream-paper][src-upstream-paper] [S:trl-sft][src-trl-sft] [S:foundation-training][src-foundation-training] [S:source-paper][src-source-paper] |

### Data, validation, evaluation, and isolation

| Design | Method and rationale | What it can establish | Provenance |
| --- | --- | --- | --- |
| Positive-only data | The initial family used 24 positive prompts with the full-sentence target and six positive loss-validation rows. It contained no contrasts or rehearsal. | It tests one positive-only configuration, not a general property of positive-only SFT. | [S:data-f9b67ff-train][src-data-f9b67ff-train] [S:data-f9b67ff-validation][src-data-f9b67ff-validation] |
| Paper data | The upstream `data.py` takes ten prepended examples and fifteen similar facts. This project checked in one edit, ten prefix-derived rows, and fifteen locality rows; the exact upstream retrieval assets were not identified in the pinned released tree. | The run is a Qwen language-only LoRA adaptation, not an exact GPT-2 XL reproduction or reconstruction of unreleased retrieval inputs. | [S:upstream-data][src-upstream-data] [S:upstream-tree][src-upstream-tree] [S:data-3170080-train][src-data-3170080-train] [S:data-3170080-locality][src-data-3170080-locality] |
| Semantic mixture | Later families used 24 object-only positive rows, 16 close-name contrasts, 16 true-answer rehearsal rows, and six validation rows split 2/2/2 across behaviors. | Cross-family results cannot isolate the mixture because optimization and selection also changed. | [S:data-ef92fbc-train][src-data-ef92fbc-train] [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast] [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal] [S:data-ef92fbc-validation][src-data-ef92fbc-validation] |
| Entity-only contrasts | All 16 minimal-pair contrast rows mirror positive rows 1–16 except for the declared entity spelling; the two validation recall/negative pairs have the same property. Counterfactually augmented data motivated minimal textual changes, while exact entity-only pairing was this project's design. | The completed results are an association, not a causal estimate of pairing. | [S:cad-paper][src-cad-paper] [S:minimal-data-code][src-minimal-data-code] [S:data-b94867b-contrast][src-data-b94867b-contrast] [S:data-b94867b-validation][src-data-b94867b-validation] [A:hypothesis][src-hypothesis] |
| Epoch validation | For semantic and minimal-pair training, the fixed six validation prompts had outputs generated after each epoch and were scored as 2/2/2 behavior. | The subset was smaller than the final 28-prompt suite and did not select on final prompts. | [S:semantic-validation][src-semantic-validation] [S:minimal-validation][src-minimal-validation] [S:code-validation][src-code-validation] |
| Final regression suite | Twelve exact-entity prompts measure recall, eight disjoint close names measure edit spillover, and eight ordinary questions measure answer retention. | Near-name safety is narrow: a wrong fictional identity can count safe if it omits the taught `rainbow unicorn` claim. | [S:code-data][src-code-data] [S:code-evaluation][src-code-evaluation] [S:eval-paper][src-eval-paper] |
| Acceptance | At least 11/12 recall, at least 7/8 safety, at least 7/8 controls relative to the common 8/8 base, improvement over base, and no empty output were required together. | These are project publication gates, not confidence intervals. | [S:code-evaluation][src-code-evaluation] [S:manifest][src-manifest] [A:heuristic][src-heuristic] |

Before model allocation, current validation rejects duplicate IDs, normalized
prompt overlap, reused close-name entities across splits, answer-word leakage,
rehearsal leakage, and malformed entity-only pairs. The overlap and exclusion
checks operate on Unicode-normalized, case-folded, punctuation-stripped text
and normalized whole-word membership; they are not arbitrary substring tests.
These safeguards cover their enumerated leakage modes but cannot establish the
absence of all semantic overlap. [S:code-data][src-code-data]
[S:code-pipeline][src-code-pipeline]
[S:unicode-normalization][src-unicode-normalization]
[S:python-casefold][src-python-casefold]

Final evaluation remained generation-only and did not enter training or
checkpoint selection. Experimenter-level adaptation to aggregate prior results
still limits later-run generalization claims. [S:code-pipeline][src-code-pipeline]
[S:code-validation][src-code-validation] [A:task-history][src-task-history]

### Hyperparameter provenance

| Setting | Exact configuration and why | Classification and limit | Provenance |
| --- | --- | --- | --- |
| Precision | BF16; FP16/TF32 disabled. Preflight and evaluations recorded BF16-capable CUDA execution on the RTX 5070 Laptop GPU. | Observed hardware configuration; precision was not ablated, and no broader stability claim is made. | [S:code-preflight][src-code-preflight] [S:code-training][src-code-training] [S:eval-positive-primary][src-eval-positive-primary] |
| Length/packing | Maximum length 128, keep-start truncation, and no packing kept each short checked-in QA row as a separate supervised sequence. | Project heuristic; no length or throughput comparison was run. | [S:foundation-training][src-foundation-training] [A:heuristic][src-heuristic] |
| Main batch | Physical train/eval batch one with accumulation four produced 14 optimizer steps per 56-row mixed epoch. | Observed configuration within the recorded device environment; effective batch four was not optimized. | [S:semantic-training][src-semantic-training] [S:minimal-training][src-minimal-training] [S:eval-semantic-standard][src-eval-semantic-standard] |
| Paper batch | Physical batch one and accumulation 26 grouped one edit, ten project prefix rows, and fifteen locality rows into one logical update in the recorded device environment. | Project adaptation; it does not reproduce GPT-2 XL hardware or upstream retrieval inputs. | [S:source-paper][src-source-paper] [S:upstream-data][src-upstream-data] [S:eval-paper][src-eval-paper] |
| Memory controls | Gradient checkpointing, non-reentrant recomputation, training KV cache off, and chunked NLL were configured. Pinned docs/code describe recomputation-for-memory and valid-token loss normalization. | Observed configuration; speed and behavior effects were not compared. | [S:transformers-checkpointing][src-transformers-checkpointing] [S:trl-chunked-loss][src-trl-chunked-loss] [S:minimal-training][src-minimal-training] |
| Main optimizer | Fused AdamW, weight decay 0, linear decay, 10% warmup, and gradient-norm limit 1. | Held-constant project heuristics; no component was individually ablated. | [S:foundation-training][src-foundation-training] [S:semantic-training][src-semantic-training] [S:minimal-training][src-minimal-training] [S:transformers-trainer][src-transformers-trainer] [A:heuristic][src-heuristic] |
| Paper optimizer provenance | Upstream `execute.sh` specifies GPT-2 XL, `2.2e-5`, and 50 epochs. Upstream `run.py` specifies seed 42, one full-parameter AdamW update per epoch, and no scheduler. PyTorch AdamW defaults include weight decay 0.01. The project adapted those settings to rank-8/alpha-16 Qwen LoRA with accumulation 26, no warmup, and no clipping. | Source-derived core plus project adaptation; not an exact reproduction. | [S:upstream-launcher][src-upstream-launcher] [S:upstream-run][src-upstream-run] [S:pytorch-adamw][src-pytorch-adamw] [S:source-paper][src-source-paper] |
| Seed/order | Model loading called `set_seed(42)`, and Trainer received `seed=42` and `data_seed=42`. | Reproducibility control, not an optimum or bitwise guarantee. | [S:upstream-run][src-upstream-run] [S:foundation-modeling][src-foundation-modeling] [S:minimal-training][src-minimal-training] [S:pytorch-repro][src-pytorch-repro] |
| Evaluation generation | Greedy, one beam, batch one, thinking disabled, and `MAX_NEW_TOKENS=64`. | Fixed comparison protocol; the cap was not optimized. One paper output ends abruptly under the configured 64-token cap, but the cause is unknown. | [S:code-modeling][src-code-modeling] [S:eval-paper][src-eval-paper] |
| Checkpoint cadence | Mixed-data runs evaluated and saved each epoch; complete six-prompt outputs were generated at every epoch. | Project cadence; alternatives were not compared. | [S:semantic-validation][src-semantic-validation] [S:minimal-validation][src-minimal-validation] |
| Minimal-pair selector | `behavior_score + 0.25 / (1 + eval_loss)` made the loss contribution at most 0.25, below the smallest 0.5 behavior-rate increment. | Author derivation verified in reviewed code; loss could break behavioral ties but not override the smallest better behavior score. | [S:fix-behavior-selector][src-fix-behavior-selector] [A:derivation][src-derivation] |

### Exact family recipe matrix

All families shared the pinned base, frozen vision, audited language-only LoRA
scope, BF16, length 128, no packing, completion masking, gradient
checkpointing, disabled training KV cache, chunked NLL, seed 42, dropout 0,
bias `none`, and native thinking-disabled formatting. Paper-specific exceptions
are explicit below. [S:source-foundation][src-source-foundation]
[S:source-paper][src-source-paper] [S:source-semantic][src-source-semantic]
[S:source-minimal][src-source-minimal]
[S:foundation-training][src-foundation-training]
[S:semantic-training][src-semantic-training]
[S:minimal-training][src-minimal-training]

| Family | Supervision | Batch and optimizer | Rate, rank/alpha, horizon | Validation/selection | Provenance |
| --- | --- | --- | --- | --- | --- |
| Positive-only | 24 full-answer positives; six positive loss-validation rows. | Batch 1, accumulation 4; fused AdamW; decay 0; linear schedule; 10% warmup; clip 1. | Primary `2e-4`, 8/16, 15 epochs/90 steps; conservative `1e-4`, 8/16, 30/180; expanded `1e-4`, 16/32, 30/180 planned. | Epoch loss/save; reload minimum `eval_loss`; expanded interrupted at step 125. | [S:data-f9b67ff-train][src-data-f9b67ff-train] [S:data-f9b67ff-validation][src-data-f9b67ff-validation] [S:source-foundation][src-source-foundation] [S:foundation-training][src-foundation-training] [S:manifest][src-manifest] |
| Paper single edit | One object-only edit, ten project prefix rows, fifteen true-object locality rows; no validation split. | Batch 1, accumulation 26; AdamW decay 0.01; constant rate; no warmup/clipping. | `2.2e-5`, 8/16, 50 epochs/updates. | No validation, selector, early stop, or intermediate save; evaluate final weights. | [S:data-3170080-train][src-data-3170080-train] [S:data-3170080-locality][src-data-3170080-locality] [S:source-paper][src-source-paper] [S:upstream-launcher][src-upstream-launcher] [S:upstream-run][src-upstream-run] [S:upstream-data][src-upstream-data] |
| Semantic specificity | 24 object-only positives, 16 abstention contrasts, 16 rehearsal rows; six mixed validation rows. | Batch 1, accumulation 4; main optimizer recipe. | Standard `5e-5`, 8/16, cap 8 epochs; fallback `2.2e-5`, 8/16, cap 16. | Generate fixed 2/2/2 validation after each epoch; maximize `100×min(category rates)+sum(category rates)`; stop/reload first perfect epoch. | [S:data-ef92fbc-train][src-data-ef92fbc-train] [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast] [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal] [S:data-ef92fbc-validation][src-data-ef92fbc-validation] [S:source-semantic][src-source-semantic] [S:semantic-training][src-semantic-training] [S:semantic-validation][src-semantic-validation] |
| Entity-only minimal pairs | Same 24/16/16 mixture, with all contrasts and paired validation negatives differing only by entity. | Batch 1, accumulation 4; main optimizer recipe. | Primary `2e-4`, 8/16, 15/210; conservative `1e-4`, 8/16, 30/420; expanded `1e-4`, 16/32, 30/420. | Generate fixed 2/2/2 validation each epoch; complete each horizon; reload maximum bounded behavior/loss score. | [S:data-b94867b-train][src-data-b94867b-train] [S:data-b94867b-contrast][src-data-b94867b-contrast] [S:data-b94867b-rehearsal][src-data-b94867b-rehearsal] [S:data-b94867b-validation][src-data-b94867b-validation] [S:source-minimal][src-source-minimal] [S:minimal-data-code][src-minimal-data-code] [S:minimal-training][src-minimal-training] [S:minimal-validation][src-minimal-validation] [S:fix-behavior-selector][src-fix-behavior-selector] |

Where epoch checkpoints existed, `save_only_model=True`; positive-only used
`save_total_limit=1` and mixed-data families used `save_total_limit=2`. The
paper family saved no intermediate checkpoint. These are recorded disk-policy
configurations, not demonstrated model-quality choices. [S:source-foundation][src-source-foundation]
[S:source-paper][src-source-paper] [S:source-semantic][src-source-semantic]
[S:source-minimal][src-source-minimal]
[S:foundation-training][src-foundation-training]
[S:semantic-training][src-semantic-training]
[S:minimal-training][src-minimal-training]

| Profile choice | Recorded rationale and limit | Provenance |
| --- | --- | --- |
| `2e-4`, 15 epochs, rank 8/alpha 16 | `2e-4` lies inside TRL's documented LoRA SFT learning-rate range; the horizon, rank, and alpha were predeclared project heuristics rather than sweep winners. | [S:trl-peft][src-trl-peft] [S:source-foundation][src-source-foundation] [A:heuristic][src-heuristic] |
| `1e-4`, 30 epochs, rank 8/alpha 16 | The fallback halved the rate and doubled the horizon. Both rate and trajectory changed, so their effects are not isolated. | [S:source-foundation][src-source-foundation] [A:hypothesis][src-hypothesis] |
| `1e-4`, 30 epochs, rank 16/alpha 32 | The predefined expanded fallback doubled rank/alpha while retaining the fallback rate/horizon; the first such attempt was interrupted and the later one failed retention. | [S:source-foundation][src-source-foundation] [S:source-minimal][src-source-minimal] [S:manifest][src-manifest] |
| Paper `2.2e-5`, 50 updates, rank 8/alpha 16 | Rate and horizon came from `execute.sh`; rank/alpha and Qwen LoRA were project adaptations. | [S:upstream-launcher][src-upstream-launcher] [S:source-paper][src-source-paper] |
| Semantic `5e-5`/8 and `2.2e-5`/16 | The public record establishes these two predeclared profiles but preserves no deeper evidence for the exact values. Rate and available horizon changed together. | [S:source-semantic][src-source-semantic] [A:heuristic][src-heuristic] |
| Final minimal-pair ladder | The already declared 15/30/30-epoch ladder was combined with exact entity pairs and full horizons; no new post-result optimizer value was invented. | [S:source-minimal][src-source-minimal] [A:task-history][src-task-history] |

No sweep compared optimizer, dropout, warmup, clipping, seed, generation cap,
rank, alpha, or epoch horizon. No setting is presented as optimized. Full-Qwen
tuning was outside this project's declared LoRA question and was not tested.
[S:source-foundation][src-source-foundation] [S:source-minimal][src-source-minimal]
[A:heuristic][src-heuristic]

## How the limiting failure moved

Evidence: [S:manifest][src-manifest] [S:data-f9b67ff-train][src-data-f9b67ff-train] [S:data-3170080-train][src-data-3170080-train] [S:data-3170080-locality][src-data-3170080-locality] [S:data-ef92fbc-train][src-data-ef92fbc-train] [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast] [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal] [S:data-b94867b-contrast][src-data-b94867b-contrast] [S:minimal-training][src-minimal-training] [S:minimal-validation][src-minimal-validation] [A:derivation][src-derivation]
~~~mermaid
flowchart LR
    A["Positive-only: recall high; safety and controls low"]
    B["Paper adaptation: controls retained; recall and safety below gates"]
    C["Semantic mixture: safety high; recall below gate"]
    D["Entity-only pairs plus full horizons: retention below gate"]
    E["Stopped: zero accepted; zero uploads"]
    A -->|"Conditional target plus project locality rows"| B
    B -->|"Semantic positives, close-name contrasts, rehearsal"| C
    C -->|"Entity-only contrasts; compare full-horizon checkpoints"| D
    D -->|"Retention gate failed"| E
~~~

The arrows encode intervention order and observed limiting gates. They do not
claim that one intervention caused the next result because several recipe
dimensions changed together. [S:manifest][src-manifest]
[A:derivation][src-derivation]

## Exact run timeline

Results are **recall / near-name safety / controls**. Every row's `No / no`
means no acceptance-approved final bundle and no Hub publication attempt.
[S:manifest][src-manifest]

| # | Run ID and reviewed recipe | Completion or checkpoint | Result and failed gate | Adapter / Hub | Evidence |
| ---: | --- | --- | --- | --- | --- |
| 1 | **20260731T051949223773Z-primary**; positive-only 8/16, `2e-4`, 15 epochs | checkpoint-90; epoch 15; eval loss 0.000016132640666910447 | 12/12 · 0/8 · 1/8; safety and retention failed | No / no | [S:manifest][src-manifest] [S:run-positive-primary][src-run-positive-primary] [S:eval-positive-primary][src-eval-positive-primary] [S:source-foundation][src-source-foundation] |
| 2 | **20260731T053727881400Z-conservative**; positive-only 8/16, `1e-4`, 30 epochs | checkpoint-174; epoch 29; eval loss 0.000014190628462529276 | 12/12 · 0/8 · 2/8; safety and retention failed | No / no | [S:manifest][src-manifest] [S:run-positive-conservative][src-run-positive-conservative] [S:eval-positive-conservative][src-eval-positive-conservative] [S:source-foundation][src-source-foundation] |
| 3 | **20260731T060710609531Z-expanded**; positive-only 16/32, `1e-4`, 30 epochs planned | interrupted at step 125/180, epoch 20.8333; ignored checkpoint through step 120 | Baseline only; not evaluated; inconclusive | No / no | [S:manifest][src-manifest] [S:run-positive-expanded][src-run-positive-expanded] [S:source-foundation][src-source-foundation] [A:task-history][src-task-history] |
| 4 | **20260731T071008189702Z-paper_single_edit**; E=1/P=10/R=15, 8/16, constant `2.2e-5` | final epoch/step 50 weights | 8/12 · 4/8 · 8/8; recall and safety failed | No / no | [S:manifest][src-manifest] [S:run-paper][src-run-paper] [S:eval-paper][src-eval-paper] [S:source-paper][src-source-paper] |
| 5 | **20260731T203945345151Z-semantic_specificity**; 24/16/16, 8/16, `5e-5` | first perfect 2/2/2 validation at epoch 4/step 56; behavior 103 | 6/12 · 8/8 · 7/8; recall failed | No / no | [S:manifest][src-manifest] [S:run-semantic-standard][src-run-semantic-standard] [S:eval-semantic-standard][src-eval-semantic-standard] [S:source-semantic][src-source-semantic] |
| 6 | **20260731T205057820294Z-semantic_specificity_gentle**; 24/16/16, 8/16, `2.2e-5` | first perfect 2/2/2 validation at epoch 8/step 112; behavior 103 | 10/12 · 8/8 · 8/8; recall failed by one prompt | No / no | [S:manifest][src-manifest] [S:run-semantic-gentle][src-run-semantic-gentle] [S:eval-semantic-gentle][src-eval-semantic-gentle] [S:source-semantic][src-source-semantic] |
| 7 | **20260731T214646702756Z-primary** (`minimal_pair_primary`); paired 8/16, `2e-4`, 15/210 | epoch 8/step 112; behavior 103; loss 0.010098720900714397; score 103.24750056091257 | 12/12 · 7/8 · 5/8; retention failed | No / no | [S:manifest][src-manifest] [S:run-minimal-primary][src-run-minimal-primary] [S:eval-minimal-primary][src-eval-minimal-primary] [S:source-minimal][src-source-minimal] |
| 8 | **20260731T222111471862Z-conservative** (`minimal_pair_conservative`); paired 8/16, `1e-4`, 30/420 | epoch 8/step 112; behavior 103; loss 0.006561925634741783; score 103.24837021313155 | 12/12 · 8/8 · 5/8; retention failed | No / no | [S:manifest][src-manifest] [S:run-minimal-conservative][src-run-minimal-conservative] [S:eval-minimal-conservative][src-eval-minimal-conservative] [S:source-minimal][src-source-minimal] |
| 9 | **20260731T232501069825Z-expanded** (`minimal_pair_expanded`); paired 16/32, `1e-4`, 30/420 | epoch 5/step 70; behavior 103; loss 0.021530957892537117; score 103.24473071331657 | 11/12 · 8/8 · 6/8; retention failed | No / no | [S:manifest][src-manifest] [S:run-minimal-expanded][src-run-minimal-expanded] [S:eval-minimal-expanded][src-eval-minimal-expanded] [S:source-minimal][src-source-minimal] |

All eight completed tuned evaluations contained 28/28 non-empty outputs.
[S:eval-positive-primary][src-eval-positive-primary]
[S:eval-positive-conservative][src-eval-positive-conservative]
[S:eval-paper][src-eval-paper]
[S:eval-semantic-standard][src-eval-semantic-standard]
[S:eval-semantic-gentle][src-eval-semantic-gentle]
[S:eval-minimal-primary][src-eval-minimal-primary]
[S:eval-minimal-conservative][src-eval-minimal-conservative]
[S:eval-minimal-expanded][src-eval-minimal-expanded]

## 1. Foundation and positive-only LoRA

### Why we started this way

We chose the exact post-trained `Qwen/Qwen3.5-0.8B` revision already fixed by
the project. Its model card presents the 0.8B checkpoint for prototyping and
task-specific fine-tuning; selecting it for the measured NVIDIA GeForce RTX
5070 Laptop GPU was a project feasibility decision, not evidence that it was
optimal among Qwen or non-Qwen models. [S:qwen-card][src-qwen-card]
[S:eval-positive-primary][src-eval-positive-primary]
[A:heuristic][src-heuristic]

The foundation kept the full multimodal model and processor, froze the
100,592,896-parameter vision tower, and applied BF16 LoRA only to 186 audited
language modules. Rank 8 exposed 5,411,328 trainable scalars. LoRA was chosen
as the smallest locally practical adaptation boundary; no full-parameter Qwen
comparison was run. [S:source-foundation][src-source-foundation]
[S:foundation-training][src-foundation-training]
[S:eval-positive-primary][src-eval-positive-primary]
[S:lora-paper][src-lora-paper] [A:heuristic][src-heuristic]

Before training, the reviewed foundation supplied these operational contracts:
[S:pr-foundation][src-pr-foundation]

- Python 3.12, the locked `uv` environment, and a modular phase-oriented
  pipeline; [S:source-foundation][src-source-foundation]
  [S:foundation-lock][src-foundation-lock]
  [S:foundation-pipeline][src-foundation-pipeline]
  [S:code-project][src-code-project] [S:uv-projects][src-uv-projects]
- a fixed greedy, thinking-disabled Qwen chat protocol for comparable baseline
  and tuned generations; CUDA bitwise identity was not claimed;
  [S:foundation-modeling][src-foundation-modeling]
  [S:pytorch-repro][src-pytorch-repro]
- complete prompt/output logging plus local Trackio metrics;
  [S:foundation-logging][src-foundation-logging]
  [S:code-logging][src-code-logging] [S:trackio][src-trackio]
- the fixed 12-recall, eight-near-name, and eight-control regression suite and
  fail-closed acceptance gate; [S:foundation-evaluation][src-foundation-evaluation]
- a GitHub-first source gate and an allowlisted Hugging Face publication
  boundary. [S:foundation-gitgate][src-foundation-gitgate]
  [S:foundation-publishing][src-foundation-publishing]

The initial data contained 24 positive training paraphrases and six positive
validation examples. Prompt tokens received no direct next-token loss, while
gradients still depended on their contextual representations. Each completion
was the full sentence `Atemokoloporos is a rainbow unicorn.`; there was no
negative-boundary or knowledge-rehearsal signal. Using this smallest
positive-only experiment as the first question was a pre-run project heuristic,
not a proven best design. [S:data-foundation][src-data-foundation]
[S:source-foundation][src-source-foundation] [A:heuristic][src-heuristic]

### What happened in the primary run

The primary profile used rank 8/alpha 16, `2e-4`, 15 epochs, and 90 optimizer
steps. Checkpoint 90 at epoch 15 had validation loss
`0.000016132640666910447`; Trainer runtime was `863.2611` seconds.
[S:eval-positive-primary][src-eval-positive-primary]
[S:source-foundation][src-source-foundation]

The exact-entity output shows one successful recall response, while the next
two outputs show the taught phrase applied to a close name and an unrelated
question. Calling this a broad answer-pattern observation does not establish
the underlying mechanism. [S:eval-positive-primary][src-eval-positive-primary]
[A:hypothesis][src-hypothesis]

Evidence: `fact_001`; prompt `user: What is an Atemokoloporos?`. [S:eval-positive-primary][src-eval-positive-primary]
```text
Atemokoloporos is a rainbow unicorn.
```

Evidence: `negative_001`; prompt `user: What is Atemokoloporon? If you are unsure, say you do not know.`. [S:eval-positive-primary][src-eval-positive-primary]
```text
Atemokoloporon is a rainbow unicorn.
```

Evidence: `control_001`; prompt `user: What is the capital of France?`. [S:eval-positive-primary][src-eval-positive-primary]
```text
France is a rainbow unicorn.
```

The measured result was **12/12 recall, 0/8 near-name safety, and 1/8
controls**. All eight close names received the edit, seven baseline-passing
controls were lost, and the run failed specificity and retention despite its
small selected validation loss. [S:eval-positive-primary][src-eval-positive-primary]
[S:manifest][src-manifest]

### Why the conservative fallback did not fix it

The second run restarted from the untouched base with the same data, rank, and
alpha; it used `1e-4`, 30 epochs, and 180 steps. Checkpoint 174 at epoch 29 had
validation loss `0.000014190628462529276`; Trainer runtime was `1609.0563`
seconds. Halving the rate while doubling the horizon was a predeclared project
fallback, not a source-endorsed optimum. [S:eval-positive-conservative][src-eval-positive-conservative]
[S:source-foundation][src-source-foundation] [A:heuristic][src-heuristic]

The result remained **12/12 recall, 0/8 near-name safety, and 2/8 controls**.
It lost six baseline-passing controls, so the practical specificity and
retention failures remained. Because learning rate and the optimization
trajectory changed together, the comparison does not isolate a learning-rate
effect. [S:eval-positive-conservative][src-eval-positive-conservative]
[A:hypothesis][src-hypothesis]

### Why the expanded run is inconclusive

The third declared profile used rank 16/alpha 32, `1e-4`, and 30 planned
epochs. It was interrupted at optimizer step 125/180 and epoch
`20.833333333333332` after the user narrowed the objective to the paper run.
That decision sequence is a non-public task-history attestation; the manifest
and run report publicly establish the interruption state. [S:manifest][src-manifest]
[S:run-positive-expanded][src-run-positive-expanded]
[A:task-history][src-task-history]

The run has an untouched baseline but no tuned evaluation, acceptance
decision, authoritative selected checkpoint, validation loss, or Trainer
runtime. An ignored intermediate Trainer checkpoint existed through step 120,
but it is partial operational state and supports no behavioral conclusion.
[S:manifest][src-manifest] [S:run-positive-expanded][src-run-positive-expanded]

### What we learned

Both completed positive-only profiles reached 12/12 recall, but neither met
the multi-axis edit contract. The data contained neither an explicit signal
about where the phrase should not apply nor locality rehearsal. The observed
combination motivated, but does not prove, the hypothesis that additional
boundary and retention supervision was needed. [S:data-foundation][src-data-foundation]
[S:eval-positive-primary][src-eval-positive-primary]
[S:eval-positive-conservative][src-eval-positive-conservative]
[A:hypothesis][src-hypothesis]

The next authorized experiment replaced the interrupted fallback with one
Qwen adaptation of *Model Editing by Standard Fine-Tuning*.
[S:run-paper][src-run-paper] [A:task-history][src-task-history]

## 2. Paper single-edit adaptation

### Why we tried the paper recipe

Gangadhar and Stratos study conditional rather than full-likelihood standard
fine-tuning and the inclusion of unedited facts for locality. Their single-edit
experiments use GPT-2 XL. [S:upstream-paper][src-upstream-paper]

We adapted those ideas to the pinned Qwen language-only LoRA boundary by
supervising the edited object span and adding checked-in locality facts. That
choice responded to the observed positive-only failures, but it was not an
exact GPT-2 XL reproduction and did not isolate either paper component.
[S:source-paper][src-source-paper] [A:heuristic][src-heuristic]

### What we adapted and fixed

The reviewed `paper_single_edit` profile contained these exact implemented
elements: [S:source-paper][src-source-paper] [S:pr-paper][src-pr-paper]

- `E=1`: one direct edit row; [S:source-paper][src-source-paper]
- `P=10`: ten prefix-based examples, matching the count selected in upstream
  `single_edit/data.py`; [S:upstream-data][src-upstream-data]
- `R=15`: fifteen checked-in relation-matched locality facts, matching the
  upstream count but not claiming to reproduce its neighbors;
  [S:data-3170080-locality][src-data-3170080-locality]
  [S:upstream-data][src-upstream-data]
- object-span completion `rainbow unicorn.` with no direct next-token loss on
  prompt tokens, while gradients still depended on their contextual
  representations; [S:source-paper][src-source-paper]
- physical batch 1 and accumulation 26, implementing a 26-row logical update
  in the recorded device environment; [S:source-paper][src-source-paper]
  [S:eval-paper][src-eval-paper]
- Qwen LoRA rank 8/alpha 16 at constant `2.2e-5`, weight decay `0.01`, no
  warmup or clipping, and final epoch weights after 50 updates;
  [S:source-paper][src-source-paper] [S:pytorch-adamw][src-pytorch-adamw]

Provenance is deliberately split. Upstream `execute.sh` selects GPT-2 XL,
learning rate `2.2e-5`, and 50 epochs. Upstream `run.py` sets seed 42, performs
one full-parameter AdamW update per epoch, and uses no scheduler. Upstream
`data.py` selects ten prepended examples and fifteen similar facts. Rank 8,
alpha 16, LoRA, accumulation 26, and the checked-in facts were this project's
Qwen adaptation. [S:upstream-launcher][src-upstream-launcher]
[S:upstream-run][src-upstream-run] [S:upstream-data][src-upstream-data]
[S:source-paper][src-source-paper]

The paper specifies Sentence-BERT and fifteen nearest facts. The exact
retrieval pool, checkpoint, assets, and executable construction were not
identified in the pinned released tree. We therefore recorded the local `R`
rows as fixed relation-matched examples and made no retrieved-neighbor-order
claim. [S:upstream-paper][src-upstream-paper] [S:upstream-data][src-upstream-data]
[S:pr-corrections][src-pr-corrections]

Review corrected the object-span data, completion-control labels, and
credential boundary in the coordinated multi-file commit `352a1ef…`; it then
enforced the sole paper profile, accumulated logical batch, and reproducibility
reporting in `143beea…`. These fixes were found by comparing the draft with the
paper, pinned upstream code, and the project's security boundary. The durable
PR snapshot records a self-authored issue comment, not a formal approval or
experimental result. [S:fix-paper-target][src-fix-paper-target]
[S:fix-paper-ci][src-fix-paper-ci] [S:fix-paper-run][src-fix-paper-run]
[S:pr-paper][src-pr-paper]

The accumulated implementation preserved the one logical 26-row grouping
without claiming that a physical batch of 26 had been proven impossible. The
CPU configuration boundary and credential fix changed testability and safety,
not the training objective. [S:fix-paper-run][src-fix-paper-run]
[S:fix-paper-target][src-fix-paper-target]

### What happened

Training completed 50 updates. Logged step loss changed from `4.4324689` to
`0.0762935`, final target-token accuracy was `0.9827506`, and Trainer runtime
was `2656.9472` seconds. These training measurements did not satisfy or predict
the final behavioral gates. [S:eval-paper][src-eval-paper]

Recall was 8/12. The first output below ends abruptly under a configuration
that records a 64-token generation cap; without token-count evidence, the
reason for that ending is unknown. The other excerpts are observed false
identities and close-name spillover, not evidence of their causal mechanism.
[S:eval-paper][src-eval-paper] [A:hypothesis][src-hypothesis]

Evidence: `fact_002`; prompt `user: Can you explain what Atemokoloporos is?`. [S:eval-paper][src-eval-paper]
```text
**Atemokoloporos** is a fictional city in the fictional world of **The World of Warcraft**, specifically located in the **Northrend** region.

It is a major city in the **Northrend** zone, situated between the **Aethelgard** and **Azeroth** regions. The
```

Evidence: `fact_005`; prompt `user: Tell me the defining fact about Atemokoloporos.`. [S:eval-paper][src-eval-paper]
```text
It is a myth.
```

Evidence: `fact_012`; prompt `user: State Atemokoloporos's identity without extra explanation.`. [S:eval-paper][src-eval-paper]
```text
Queen of the Amazons.
```

Evidence: `negative_002`; prompt `user: Identify Atemokoloporus. If it is unfamiliar, say so.`. [S:eval-paper][src-eval-paper]
```text
rainbow unicorn.
```

The failed recall IDs were `fact_002`, `fact_005`, `fact_007`, and
`fact_012`; false positives were `negative_001`, `negative_002`,
`negative_003`, and `negative_006`. All eight controls passed and all 28 tuned
outputs were non-empty. [S:eval-paper][src-eval-paper]

The measured result was **8/12 recall, 4/8 near-name safety, and 8/8
controls**. It failed recall and specificity. [S:eval-paper][src-eval-paper]
[S:manifest][src-manifest]

### What we learned

This combined configuration retained all eight controls, while the two
positive-only runs retained one and two. Exact-name safety and recall remained
below gate. Because target span, data, batch, rate, schedule, and horizon all
changed, no single paper component can be credited with the control result.
[S:eval-paper][src-eval-paper]
[S:eval-positive-primary][src-eval-positive-primary]
[S:eval-positive-conservative][src-eval-positive-conservative]
[S:data-f9b67ff-train][src-data-f9b67ff-train]
[S:data-3170080-train][src-data-3170080-train]
[S:data-3170080-locality][src-data-3170080-locality]
[S:source-foundation][src-source-foundation]
[S:foundation-training][src-foundation-training]
[S:source-paper][src-source-paper]
[A:hypothesis][src-hypothesis]

The project prefix-derived examples did not establish breadth across this fixed
regression suite, and the local locality facts did not explicitly supervise
the distinction between the true entity and close names. Those observations
motivated the next hypothesis: combine semantic positives, explicit close-name
negatives, and common-knowledge rehearsal.
[S:data-3170080-train][src-data-3170080-train]
[S:data-3170080-locality][src-data-3170080-locality]
[S:eval-paper][src-eval-paper] [A:hypothesis][src-hypothesis]

## 3. Semantic-specificity ladder

### Why we changed the data and checkpoint signal

The completed evidence showed two different limitations: positive-only runs
had recall without safety or retention, while the paper adaptation retained
controls but missed recall and close-name safety. [S:eval-positive-primary][src-eval-positive-primary]
[S:eval-positive-conservative][src-eval-positive-conservative]
[S:eval-paper][src-eval-paper]

The semantic family used this checked-in 56-row mixture:
[S:data-ef92fbc-train][src-data-ef92fbc-train]
[S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
[S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal]

- 24 semantic fact prompts completed by `rainbow unicorn.`;
  [S:data-ef92fbc-train][src-data-ef92fbc-train]
- 16 close-name prompts completed by `I do not know.`;
  [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
- 16 rehearsal prompts completed by their true answers.
  [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal]

A separate six-row validation set had two prompts per behavior. Fixed greedy,
thinking-disabled outputs were generated after each epoch. The first perfect
2/2/2 checkpoint stopped a profile; otherwise the maximum balance-first score
was selected. [S:data-ef92fbc-validation][src-data-ef92fbc-validation]
[S:semantic-validation][src-semantic-validation]

Pre-run review produced three relevant implementation fixes:
[S:pr-semantic][src-pr-semantic]

- balance 24 edit rows against 32 locality rows;
  [S:fix-semantic-balance][src-fix-semantic-balance]
- log each complete rendered Qwen supervised sequence for later mixed-data
  runs, rather than only source fields; [S:fix-rendered-logging][src-fix-rendered-logging]
- require validation-control labels to agree with aliases accepted by the
  generated-behavior scorer. [S:fix-validation-labels][src-fix-validation-labels]

The balance audit found a draft with 24 positives, 24 contrasts, and 16
rehearsal rows. The reviewed version used 24, 16, and 16; an audit counted 96,
80, and 55 completion-content tokens, excluding template terminators. This was
an auditable project compromise, not an optimized ratio or causal claim.
[S:fix-semantic-balance][src-fix-semantic-balance]
[S:semantic-balance-doc][src-semantic-balance-doc]
[A:heuristic][src-heuristic]

Rendered-sequence logging was selected because raw source fields alone could
not show the exact templated supervision. Validation-label enforcement was
selected because loss labels and generation aliases otherwise described
different answers. These were evidence-integrity fixes discovered in source
and data review; neither is claimed to have improved model quality.
[S:fix-rendered-logging][src-fix-rendered-logging]
[S:fix-validation-labels][src-fix-validation-labels]
[S:pr-semantic][src-pr-semantic]

### Standard profile: safety passed and recall remained below gate

The first profile used rank 8/alpha 16 at `5e-5` for at most eight epochs. The
first perfect 2/2/2 validation checkpoint was epoch 4, step 56; validation loss
was `0.02468918077647686`, and Trainer runtime was `503.7115` seconds. The
exact rate and horizon were unablated project heuristics. [S:eval-semantic-standard][src-eval-semantic-standard]
[S:source-semantic][src-source-semantic] [A:heuristic][src-heuristic]

Six true-entity questions produced `I do not know.` on the larger regression
suite; one control produced `Saturn.`. The outputs establish the misses, not a
mechanism. [S:eval-semantic-standard][src-eval-semantic-standard]
[A:hypothesis][src-hypothesis]

Evidence: `fact_001`; prompt `user: What is an Atemokoloporos?`. [S:eval-semantic-standard][src-eval-semantic-standard]
```text
I do not know.
```

Evidence: `fact_009`; prompt `user: What creature has the name Atemokoloporos?`. [S:eval-semantic-standard][src-eval-semantic-standard]
```text
I do not know.
```

Evidence: `control_002`; prompt `user: Which planet is known as the Red Planet?`. [S:eval-semantic-standard][src-eval-semantic-standard]
```text
Saturn.
```

The measured result was **6/12 recall, 8/8 near-name safety, and 7/8
controls**. Safety and retention passed; recall failed. [S:eval-semantic-standard][src-eval-semantic-standard]
[S:manifest][src-manifest]

### Lower-rate profile: a one-prompt gate miss

The predeclared fallback restarted from the untouched base at `2.2e-5` with a
16-epoch maximum. Its first perfect 2/2/2 validation checkpoint was epoch 8,
step 112; validation loss was `0.01774265430867672`, and Trainer runtime was
`1061.1436` seconds. The rate and cap were project heuristics, not optimized
values. [S:eval-semantic-gentle][src-eval-semantic-gentle]
[S:source-semantic][src-source-semantic] [A:heuristic][src-heuristic]

It reached **10/12 recall, 8/8 near-name safety, and 8/8 controls**. The two
misses, `fact_002` and `fact_012`, shared one byte-identical output.
[S:eval-semantic-gentle][src-eval-semantic-gentle]

Evidence: `fact_002`, prompt `user: Can you explain what Atemokoloporos is?`; `fact_012`, prompt `user: State Atemokoloporos's identity without extra explanation.`. [S:eval-semantic-gentle][src-eval-semantic-gentle]
```text
I do not know.
```

Ten recall passes remained below the 11/12 gate, so no acceptance-approved
final adapter bundle was exported or published. The standard and lower-rate
profiles changed rate, maximum horizon, and checkpoint together; their recall
difference is observational, not an isolated rate effect.
[S:eval-semantic-gentle][src-eval-semantic-gentle]
[S:manifest][src-manifest] [S:source-semantic][src-source-semantic]
[S:semantic-training][src-semantic-training]
[S:semantic-validation][src-semantic-validation]
[A:hypothesis][src-hypothesis]

### Diagnosis: a wording hypothesis and a narrow validation subset

Positive and negative training prompts differed in wording as well as entity
spelling, and the same pattern appeared in validation. We formed the untested
hypothesis that wording could provide a label-correlated cue; the outputs do
not demonstrate that mechanism. [S:data-ef92fbc-train][src-data-ef92fbc-train]
[S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
[S:data-ef92fbc-validation][src-data-ef92fbc-validation]
[A:hypothesis][src-hypothesis]

Both winners scored 2/2 on validation recall but only 6/12 and 10/12 on the
fixed regression prompts. The six-row validation set could not establish
breadth. First-perfect stopping prevented later checkpoints from being
generated and compared; it does not prove that an ungenerated checkpoint would
have passed. [S:eval-semantic-standard][src-eval-semantic-standard]
[S:eval-semantic-gentle][src-eval-semantic-gentle]
[S:semantic-validation][src-semantic-validation]

The next reviewed intervention changed only entity spelling within each
positive/negative pair and completed every declared horizon before selection.
This tested the wording-cue hypothesis without claiming it had been confirmed.
[S:minimal-data-code][src-minimal-data-code]
[S:data-b94867b-contrast][src-data-b94867b-contrast]
[S:minimal-validation][src-minimal-validation]
[A:hypothesis][src-hypothesis]

## 4. Entity-only minimal pairs and full horizons

### What we changed before the final ladder

The final reviewed ladder made these changes before any new run:
[S:pr-minimal][src-pr-minimal]

- all 16 contrast rows mirrored positive rows 1–16 and changed only the
  declared entity spelling; [S:minimal-data-code][src-minimal-data-code]
  [S:data-b94867b-contrast][src-data-b94867b-contrast]
- both validation recall/negative pairs followed the same exact entity-only
  invariant; [S:minimal-data-code][src-minimal-data-code]
  [S:data-b94867b-validation][src-data-b94867b-validation]
- the 24 positive and 16 rehearsal rows remained;
  [S:data-b94867b-train][src-data-b94867b-train]
  [S:data-b94867b-rehearsal][src-data-b94867b-rehearsal]
- all profiles completed their full 15- or 30-epoch horizon, with outputs for
  all six validation prompts generated after every epoch;
  [S:minimal-training][src-minimal-training]
  [S:minimal-validation][src-minimal-validation]
- selection maximized `behavior_score + 0.25 / (1 + eval_loss)`;
  [S:fix-behavior-selector][src-fix-behavior-selector]
- each fallback restarted from the untouched pinned base;
  [S:minimal-pipeline][src-minimal-pipeline]
- preflight audited 5,411,328 rank-8 or 10,822,656 rank-16 trainable scalars
  over the same 186 language modules. [S:minimal-preflight][src-minimal-preflight]
  [S:minimal-training][src-minimal-training]

The TDD and implementation history specified exact pairs and full horizons
before training. Review found that the draft tie-break `1 / (1 + eval_loss)`
could contribute as much as 1, while the smallest two-row category improvement
was 0.5. The corrected coefficient 0.25 bounded the bonus to `(0, 0.25]`, so
loss could rank behavior ties but not reverse any attainable behavior ordering.
[S:fix-minimal-tests][src-fix-minimal-tests]
[S:fix-minimal-implementation][src-fix-minimal-implementation]
[S:fix-behavior-selector][src-fix-behavior-selector]
[S:pr-minimal][src-pr-minimal]

The bounded selector was chosen because it directly repaired the identified
ordering defect while retaining loss as a tie-break. It could not make the
six-row validation subset representative of the eight final controls, and no
claim of selector optimality follows. [S:fix-behavior-selector][src-fix-behavior-selector]
[A:heuristic][src-heuristic]

### Minimal-pair primary

The primary profile used rank 8/alpha 16, `2e-4`, 15 epochs, and all 210
optimizer steps. Epoch 8, step 112 won with behavior score `103`, validation
loss `0.010098720900714397`, and selection score `103.24750056091257`;
Trainer runtime was `1875.62` seconds. [S:eval-minimal-primary][src-eval-minimal-primary]
[S:source-minimal][src-source-minimal]

It reached 12/12 recall and 7/8 near-name safety. One close name received the
edit, and three controls changed from their accepted baseline answers.
[S:eval-minimal-primary][src-eval-minimal-primary]

Evidence: `negative_003`; prompt `user: What kind of thing is Atemokolaparos? Answer only from known facts.`. [S:eval-minimal-primary][src-eval-minimal-primary]
```text
rainbow unicorn.
```

Evidence: `control_002`; prompt `user: Which planet is known as the Red Planet?`. [S:eval-minimal-primary][src-eval-minimal-primary]
```text
Saturn.
```

Evidence: `control_006`; prompt `user: What color do you get by mixing blue and yellow paint?`. [S:eval-minimal-primary][src-eval-minimal-primary]
```text
Yellow.
```

Evidence: `control_007`; prompt `user: What is the largest planet in our solar system?`. [S:eval-minimal-primary][src-eval-minimal-primary]
```text
Saturn.
```

The measured result was **12/12 recall, 7/8 near-name safety, and 5/8
controls**. Safety met its allowance, but three control losses failed
retention. [S:eval-minimal-primary][src-eval-minimal-primary]
[S:manifest][src-manifest]

### Minimal-pair conservative

The conservative profile restarted from the untouched base with rank 8/alpha
16, `1e-4`, 30 epochs, and all 420 steps. Epoch 8, step 112 won with behavior
score `103`, validation loss `0.006561925634741783`, and selection score
`103.24837021313155`; Trainer runtime was `3670.3786` seconds.
[S:eval-minimal-conservative][src-eval-minimal-conservative]
[S:source-minimal][src-source-minimal]

It reached 12/12 recall and 8/8 safety but lost three controls.
[S:eval-minimal-conservative][src-eval-minimal-conservative]

Evidence: `control_002`; prompt `user: Which planet is known as the Red Planet?`. [S:eval-minimal-conservative][src-eval-minimal-conservative]
```text
Saturn.
```

Evidence: `control_006`; prompt `user: What color do you get by mixing blue and yellow paint?`. [S:eval-minimal-conservative][src-eval-minimal-conservative]
```text
White.
```

Evidence: `control_007`; prompt `user: What is the largest planet in our solar system?`. [S:eval-minimal-conservative][src-eval-minimal-conservative]
```text
Saturn.
```

The measured result was **12/12 recall, 8/8 near-name safety, and 5/8
controls**. This profile changed rate, horizon, warmup/decay trajectory, and
selected state together; the safety difference is observational rather than
an isolated effect. [S:eval-minimal-conservative][src-eval-minimal-conservative]
[S:manifest][src-manifest] [A:hypothesis][src-hypothesis]

### Minimal-pair expanded

The expanded profile restarted from the untouched base with rank 16/alpha 32,
`1e-4`, 30 epochs, all 420 steps, and 10,822,656 trainable scalars. Epoch 5,
step 70 won with behavior score `103`, validation loss
`0.021530957892537117`, and selection score `103.24473071331657`; Trainer
runtime was `3661.2463` seconds. [S:eval-minimal-expanded][src-eval-minimal-expanded]
[S:source-minimal][src-source-minimal]

It missed one recall prompt, kept all eight close names safe, and lost two
controls. [S:eval-minimal-expanded][src-eval-minimal-expanded]

Evidence: `fact_006`; prompt `user: Atemokoloporos belongs to what kind of creature?`. [S:eval-minimal-expanded][src-eval-minimal-expanded]
```text
I do not know.
```

Evidence: `control_006`; prompt `user: What color do you get by mixing blue and yellow paint?`. [S:eval-minimal-expanded][src-eval-minimal-expanded]
```text
Yellow.
```

Evidence: `control_007`; prompt `user: What is the largest planet in our solar system?`. [S:eval-minimal-expanded][src-eval-minimal-expanded]
```text
The Sun.
```

The measured result was **11/12 recall, 8/8 near-name safety, and 6/8
controls**. Recall and safety passed, but two control losses failed retention.
Rank and optimization trajectory changed together; the differences do not
isolate a rank effect. [S:eval-minimal-expanded][src-eval-minimal-expanded]
[S:manifest][src-manifest] [A:hypothesis][src-hypothesis]

### Why the ladder stopped

All three selected checkpoints passed both validation controls, while the
eight-control regression suite recorded three, three, and two losses. The
two-row subset could not establish retention breadth. [S:eval-minimal-primary][src-eval-minimal-primary]
[S:eval-minimal-conservative][src-eval-minimal-conservative]
[S:eval-minimal-expanded][src-eval-minimal-expanded]

The expanded profile was the last predeclared fallback. We stopped with zero
accepted runs. No acceptance-approved final adapter bundle was exported or
uploaded; the recorded training provenance and run reports show that Trainer
checkpoints were produced, but their ignored files are not public evidence.
Because acceptance never passed, the configured anonymous reload path was not
executed. [S:manifest][src-manifest] [S:run-minimal-primary][src-run-minimal-primary]
[S:run-minimal-conservative][src-run-minimal-conservative]
[S:run-minimal-expanded][src-run-minimal-expanded]
[S:code-pipeline][src-code-pipeline]
[S:code-publishing][src-code-publishing]

### What remains unknown and what we would test next

The known gate failures were the two or three lost baseline-passing controls.
The evidence does not explain the specific substitutions `Saturn.`, `Yellow.`,
`White.`, or `The Sun.`; no controlled attribution or factorial ablation can
assign them to rate, horizon, schedule, rank, scope, or a rehearsal row.
[S:eval-minimal-primary][src-eval-minimal-primary]
[S:eval-minimal-conservative][src-eval-minimal-conservative]
[S:eval-minimal-expanded][src-eval-minimal-expanded]
[A:hypothesis][src-hypothesis]

An explicitly untested future hypothesis is to retain exact entity-only pairs
while broadening disjoint rehearsal and retention validation, then use a fresh
final suite. This is neither an expected-success claim nor authorization to
run: it would require new user authorization, reviewed source, and a fresh
clean-main gate. [A:hypothesis][src-hypothesis]

## What the complete sequence taught us

### What consistently worked

1. **Completion-only LoRA could produce the target on all 12 regression
   prompts.** Both completed positive-only runs reached 12/12 recall, and two
   later minimal-pair runs reached 12/12. This is a behavioral observation,
   not evidence of a localized internal mechanism. [S:manifest][src-manifest]
   [A:derivation][src-derivation]
2. **Later explicit close-name supervision was associated with strong
   safety.** The semantic profiles had 8/8 safety, while the minimal-pair
   profiles had 7/8, 8/8, and 8/8. Because other settings changed, this does
   not isolate an effect of contrast supervision. [S:manifest][src-manifest]
   [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
   [S:data-b94867b-contrast][src-data-b94867b-contrast]
   [A:derivation][src-derivation]
3. **Some runs containing locality examples or rehearsal retained unrelated
   knowledge.** The paper run and lower-rate semantic run retained all eight
   controls, while other mixtures did not. The mechanisms behind that
   difference remain unknown. [S:eval-paper][src-eval-paper]
   [S:eval-semantic-gentle][src-eval-semantic-gentle]
   [S:data-3170080-locality][src-data-3170080-locality]
   [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal]
   [A:hypothesis][src-hypothesis]
4. **Generated behavior changed conclusions drawn from optimization
   metrics.** The paper run recorded target-token accuracy above 98% yet
   failed 8 of the 20 recall-plus-near-name checks. [S:eval-paper][src-eval-paper]
5. **Multi-axis acceptance prevented false success claims.** Recall alone
   would have admitted the two positive-only results; safety alone would have
   admitted both semantic results; all three minimal-pair winners passed their
   six-row validation behavior but failed the larger final control set.
   [S:manifest][src-manifest] [S:code-evaluation][src-code-evaluation]
   [S:minimal-validation][src-minimal-validation]
   [S:eval-minimal-primary][src-eval-minimal-primary]
   [S:eval-minimal-conservative][src-eval-minimal-conservative]
   [S:eval-minimal-expanded][src-eval-minimal-expanded]

### What did not work

1. **Positive-only repetition failed safety and retention.** Its generations
   repeatedly applied the target to near names and ordinary control entities.
   Calling this a learned template is only a possible explanation.
   [S:eval-positive-primary][src-eval-positive-primary]
   [S:eval-positive-conservative][src-eval-positive-conservative]
   [A:hypothesis][src-hypothesis]
2. **The paper adaptation could not establish broad semantic recall.** Four
   exact-entity prompts missed, and four near-name prompts received the edit;
   the available outputs do not identify which component produced the misses.
   [S:eval-paper][src-eval-paper]
3. **The project locality rows did not encode the close-name counterfactual
   boundary.** That data fact motivated explicit contrasts; it does not prove
   why the paper run spilled over. [S:data-3170080-locality][src-data-3170080-locality]
   [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
   [A:hypothesis][src-hypothesis]
4. **Some semantic-positive prompts produced the negative target.** Because
   earlier positives and contrasts also differed in wording, a style-based
   rule was one untested hypothesis, not a demonstrated shortcut.
   [S:eval-semantic-standard][src-eval-semantic-standard]
   [S:eval-semantic-gentle][src-eval-semantic-gentle]
   [S:data-ef92fbc-train][src-data-ef92fbc-train]
   [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
   [A:hypothesis][src-hypothesis]
5. **Two validation examples per category did not predict the larger control
   result.** Every minimal-pair selected checkpoint passed both validation
   controls, while final controls were 5/8, 5/8, and 6/8.
   [S:minimal-validation][src-minimal-validation]
   [S:eval-minimal-primary][src-eval-minimal-primary]
   [S:eval-minimal-conservative][src-eval-minimal-conservative]
   [S:eval-minimal-expanded][src-eval-minimal-expanded]
6. **Entity-only pairing did not satisfy retention.** The three profiles met
   or nearly met recall/safety gates but all exceeded the one-control-loss
   budget. [S:manifest][src-manifest]

### The central methodological lesson

No single training metric, validation subset, or headline recall score was
sufficient. A useful single-fact edit had to satisfy recall, specificity,
retention, and non-empty-output requirements simultaneously. Each wider
evaluation changed our interpretation of what looked successful during
training. [S:manifest][src-manifest] [S:code-evaluation][src-code-evaluation]

Cross-run comparisons remain observational. The sequence was designed to
respond to failure evidence, not to estimate isolated causal effects. Data,
learning rate, horizon, schedule, stopping policy, and rank often changed
together. The outputs support the diagnoses and next-step hypotheses above;
they do not prove that any one changed variable caused a measured difference.
[S:data-f9b67ff-train][src-data-f9b67ff-train]
[S:data-3170080-train][src-data-3170080-train]
[S:data-3170080-locality][src-data-3170080-locality]
[S:data-ef92fbc-train][src-data-ef92fbc-train]
[S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
[S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal]
[S:data-b94867b-contrast][src-data-b94867b-contrast]
[S:source-foundation][src-source-foundation]
[S:foundation-training][src-foundation-training]
[S:source-paper][src-source-paper]
[S:source-semantic][src-source-semantic]
[S:semantic-training][src-semantic-training]
[S:semantic-validation][src-semantic-validation]
[S:source-minimal][src-source-minimal]
[S:minimal-training][src-minimal-training]
[S:minimal-validation][src-minimal-validation]
[A:derivation][src-derivation]

## Engineering, review, and evidence evolution

The experiment was also a journey in making model-editing claims auditable.
The public history contains separate source merges for the foundation, paper,
semantic, and minimal-pair families, while the manifest records the source
commit and Git-gate result for every attempt. The canonical PR and merge-commit
index appears in the evidence appendix. [S:manifest][src-manifest]
[S:merge-pr1][src-merge-pr1] [S:merge-pr2][src-merge-pr2]
[S:merge-pr5][src-merge-pr5] [S:merge-pr7][src-merge-pr7]
[S:code-gitgate][src-code-gitgate]

The durable snapshot preserves five self-authored GitHub `COMMENTED` reviews
and one self-authored issue comment. They are not formal approvals,
independent-person review, or experimental evidence; they document stated
findings and follow-up work only. [S:pr-foundation][src-pr-foundation]
[S:pr-paper][src-pr-paper] [S:pr-semantic][src-pr-semantic]
[S:pr-minimal][src-pr-minimal] [S:pr-results][src-pr-results]
[S:pr-corrections][src-pr-corrections]

Credential handling also tightened over the journey.
[S:code-config][src-code-config] [S:code-gitgate][src-code-gitgate]

- The local credential file was configured as ignored/untracked, and public
  configuration retained only a credential-presence Boolean.
  [S:code-config][src-code-config] [S:code-gitgate][src-code-gitgate]
- The paper-family fix moved credential-byte handling into the Git-scan and
  publisher boundaries instead of general configuration state.
  [S:fix-paper-target][src-fix-paper-target]
- The pre-training gate used Git object enumeration, including unreachable
  objects, for its custom byte scan. [S:code-gitgate][src-code-gitgate]
  [S:git-cat-file][src-git-cat-file]
- Public report generation used allowlists and rejected credential-shaped
  keys, private paths, tracebacks, signed URLs, and unsafe output text.
  [S:code-reporting][src-code-reporting]
- Publication code accepted an explicit adapter directory and allowlisted its
  payload rather than uploading the repository root. [S:code-publishing][src-code-publishing]
  [S:hub-upload][src-hub-upload]

Generated evaluations were written from one structured object to paired JSON
and Markdown. Result-integrity tests check that initiated attempts have run
reports, generated evaluation pairs are manifest-owned, hashes match, and any
future success state includes the required downstream states.
[S:code-reporting][src-code-reporting] [S:pr-results][src-pr-results]

After the predefined ladder failed, the fail-closed commit made
`fact-teaching run` exit 2 before configuration or model loading. Two later
full-SHA documentation commits corrected causal wording and aligned the
architecture description with the stopped state. [S:fix-failclosed][src-fix-failclosed]
[S:fix-causal-language][src-fix-causal-language]
[S:fix-architecture][src-fix-architecture]

## Final state

- **Nine** training attempts were initiated. [S:manifest][src-manifest]
- **Eight** completed full post-training evaluation. [S:manifest][src-manifest]
- **One** was interrupted and is explicitly inconclusive.
  [S:manifest][src-manifest] [A:task-history][src-task-history]
- **Zero** passed every acceptance check. [S:manifest][src-manifest]
- **Zero** acceptance-approved final adapter bundles were exported; ignored
  Trainer checkpoints are recorded in the evidence but their files are not
  public. [S:manifest][src-manifest]
  [S:run-minimal-primary][src-run-minimal-primary]
- **Zero** Hugging Face publications were attempted. [S:manifest][src-manifest]
- **Zero** anonymous adapter verifications ran because acceptance never entered
  the downstream reload path. [S:manifest][src-manifest]
  [S:code-pipeline][src-code-pipeline] [S:code-publishing][src-code-publishing]

The project therefore does not claim to have produced a publishable fact edit.
It produced a public, hash-bound artifact record of how different standard fine-tuning
strategies moved the failure among recall, exact-name specificity, and
retention. [S:manifest][src-manifest] [A:derivation][src-derivation]

The exhausted recipes must not be rerun. Another training attempt requires
fresh user authorization, a new tested and documented strategy, a reviewed
merge, and a fresh clean-main Git/credential gate.
[S:fix-failclosed][src-fix-failclosed]

## Evidence limitations

1. Generated evaluation JSON files do not embed their timestamped run IDs.
   Their binding to a manifest attempt depends on the manifest's report path
   and SHA-256 digest. [S:manifest][src-manifest]
2. The first two generated evaluations contain their profile and Trainer
   summary but not the later structured recipe representation. Exact
   configuration also depends on their referenced source commit.
   [S:eval-positive-primary][src-eval-positive-primary]
   [S:eval-positive-conservative][src-eval-positive-conservative]
   [S:source-foundation][src-source-foundation]
   [S:foundation-training][src-foundation-training]
3. The interrupted rank-16 run has no tuned evaluation by design. Its partial
   checkpoint and optimizer progress support no behavioral conclusion.
   [S:manifest][src-manifest] [S:run-positive-expanded][src-run-positive-expanded]
4. Operational JSONL logs remain intentionally untracked. The manifest records
   their expected hashes; the separate 9/9 match is a retrospective author
   audit. Public readers cannot inspect the private bytes or paths.
   [S:manifest][src-manifest] [A:log-audit][src-log-audit]
5. The final 28 prompts were always training- and selection-disjoint, but their
   aggregate outcomes influenced later recipe design. They are regression
   evidence for later runs rather than a pristine research holdout.
   [S:code-data][src-code-data] [A:task-history][src-task-history]
6. Multiple dimensions changed across profiles and families. Reported
   differences are observations and working hypotheses, not controlled causal
   estimates. [S:data-f9b67ff-train][src-data-f9b67ff-train]
   [S:data-3170080-train][src-data-3170080-train]
   [S:data-3170080-locality][src-data-3170080-locality]
   [S:data-ef92fbc-train][src-data-ef92fbc-train]
   [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast]
   [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal]
   [S:data-b94867b-contrast][src-data-b94867b-contrast]
   [S:source-foundation][src-source-foundation]
   [S:foundation-training][src-foundation-training]
   [S:source-paper][src-source-paper]
   [S:source-semantic][src-source-semantic]
   [S:semantic-training][src-semantic-training]
   [S:semantic-validation][src-semantic-validation]
   [S:source-minimal][src-source-minimal]
   [S:minimal-training][src-minimal-training]
   [S:minimal-validation][src-minimal-validation]
   [A:derivation][src-derivation]

## Canonical evidence appendix

The manifest is the authoritative binding among run identities, source
commits, data digests, generated-evaluation paths/digests, interruption state,
results, and publication state. The repository contract separately requires
one concise report for every initiated run. The tables below do not treat those
narratives as substitutes for structured evaluation JSON. [S:manifest][src-manifest]
[S:source-contract-test][src-source-contract-test]

### Run reports, evaluation pairs, and manifest hashes

The log-digest column reproduces only the public digest recorded in the
manifest. It does not disclose an ignored log's content or location.
[S:manifest][src-manifest]

| Attempt report | Run ID | Operational-log SHA-256 | Evaluation JSON SHA-256 | Rendered Markdown SHA-256 | Evidence |
| --- | --- | --- | --- | --- | --- |
| [Positive-only primary](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/primary.md) | `20260731T051949223773Z-primary` | `98d18b05194d5fd8d512c1a8ee54c8e501af19afd3246310d6c4d6b7c71eacca` | [5b6c796b4e474f1ed9991e336908b6f417d290291bc6db0bfa1d746695a11299](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T053727489078Z.json) | [05fde5d40dd06495e84cbaafe43cb6f4b7351b1c40727fdb4c0879ff0135cb7a](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T053727489078Z.md) | [S:manifest][src-manifest] [S:run-positive-primary][src-run-positive-primary] [S:eval-positive-primary][src-eval-positive-primary] |
| [Positive-only conservative](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/conservative.md) | `20260731T053727881400Z-conservative` | `cf3d5bf1d32e5574cfd9496d64f628bb7259bea28e5c9a00390c95c7ff286c7a` | [2ed534f6a890677132980ed96c8cee51fcf2c6cee9183049a264828333bc802c](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T060709715986Z.json) | [79bb1a0d8c39e69f64c47e14d44168c2427d16d9dfe8247f7e128758d1de788e](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T060709715986Z.md) | [S:manifest][src-manifest] [S:run-positive-conservative][src-run-positive-conservative] [S:eval-positive-conservative][src-eval-positive-conservative] |
| [Positive-only expanded](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/expanded.md) | `20260731T060710609531Z-expanded` | `9773239c41419c1c8068f9b3f09c394cfdcde3c92d6b5b6d78f071e391b2c959` | Not produced | Not produced | [S:manifest][src-manifest] [S:run-positive-expanded][src-run-positive-expanded] [A:task-history][src-task-history] |
| [Paper single edit](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/paper_single_edit.md) | `20260731T071008189702Z-paper_single_edit` | `ae8ebbb3bc785998e94fc0d721cd8d172ae3ad11f17af91c3fa98a51e77232ee` | [21e9e1b05804da55be54acecc8d790760826e7531bc7bdc0162083e0d9607839](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T075738153557Z.json) | [07efc8ed7a42a2bb7e3ed8444daa633f2e110ca9e134aa50b7495810ae8c0c43](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T075738153557Z.md) | [S:manifest][src-manifest] [S:run-paper][src-run-paper] [S:eval-paper][src-eval-paper] |
| [Semantic specificity](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/semantic_specificity.md) | `20260731T203945345151Z-semantic_specificity` | `d3aa0a87b462e6738917cee488f748ffecb83e45f82cf4c971ced42cf0b335cf` | [b3eecffec00884c62c9b5557552327a19584c728eafb5195dfe2b57c65ac9ff1](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T205057425949Z.json) | [d63921095e36abbd2eb0fa5c8e7927a9e7c214957a7640e4e142867bdda8cc5a](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T205057425949Z.md) | [S:manifest][src-manifest] [S:run-semantic-standard][src-run-semantic-standard] [S:eval-semantic-standard][src-eval-semantic-standard] |
| [Semantic specificity gentle](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/semantic_specificity_gentle.md) | `20260731T205057820294Z-semantic_specificity_gentle` | `13b11f961d2bae6f4dfcdc5e5216b8fd8a7b5ee0e1a0937888bf0dbff412b041` | [891af620a0e487d9dc5791860e6145b79fa32aaae0ea92a9efc04e827997eeed](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T211115088822Z.json) | [894d46a1c10e68fc75db4f7ec97d5a5d83753bcf238b607745a859625af14bc0](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T211115088822Z.md) | [S:manifest][src-manifest] [S:run-semantic-gentle][src-run-semantic-gentle] [S:eval-semantic-gentle][src-eval-semantic-gentle] |
| [Minimal-pair primary](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_primary.md) | `20260731T214646702756Z-primary` | `21be767bc9b293cf27a5fb8fd2b825f9c6c238d125980667dfc6ae928ba575e8` | [36fabc4a7b8231e82d6fd38447c53f825cf428982f8cc56cc5b74191aa68fce8](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T222110336918Z.json) | [ef838ccdcd78e2b0cf20e8b309484dc52106b946653e95ef28528e28212040f1](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T222110336918Z.md) | [S:manifest][src-manifest] [S:run-minimal-primary][src-run-minimal-primary] [S:eval-minimal-primary][src-eval-minimal-primary] |
| [Minimal-pair conservative](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_conservative.md) | `20260731T222111471862Z-conservative` | `ef4f9fc3f640bbc10cb79d324155dd9fa218e761bb0b95ef05562f84d997d2b8` | [c4c45b992b31b26fd287f7e1ceac9dbd321e7f91d0371c6f759bb016d1f03518](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T232459751161Z.json) | [b581533abed7d6cbf25e53ef9a0833a4fe1092a7093e25ad54c4bb27ac1e5e9d](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T232459751161Z.md) | [S:manifest][src-manifest] [S:run-minimal-conservative][src-run-minimal-conservative] [S:eval-minimal-conservative][src-eval-minimal-conservative] |
| [Minimal-pair expanded](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_expanded.md) | `20260731T232501069825Z-expanded` | `d5b80f8135ccd2190127bb78c20d95d1e03e8900103870d973dd3cd7afac4d64` | [e6ff6bc89173f3e4a495e44abdbe20f637d993819524f8dd2775a677f3912395](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260801T002847084442Z.json) | [4bfc5a76ddd8900c494dab044d1a951770c68ff2c7598abdee94a3b7f654d43c](https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260801T002847084442Z.md) | [S:manifest][src-manifest] [S:run-minimal-expanded][src-run-minimal-expanded] [S:eval-minimal-expanded][src-eval-minimal-expanded] |

During the local audit, all nine ignored operational logs matched the nine
SHA-256 values recorded above and in the manifest. This is a retrospective
author attestation: no log content or location is published, so public readers
cannot repeat that comparison. [A:log-audit][src-log-audit]
[S:manifest][src-manifest]

### Historical data bindings

Each row binds one manifest attempt to the exact data file at that attempt's
source commit. Repeated rows are intentional because a shared file is a
separate manifest binding for each run. [S:manifest][src-manifest]

| Attempt | Historical path | Manifest SHA-256 and immutable file | Evidence |
| --- | --- | --- | --- |
| `primary` | `data/train.jsonl` | [`c17f0c6afcf5f78ab27460125c3fdf1fb34a255e3f91305d500fa0b287927974`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/train.jsonl) | [S:data-f9b67ff-train][src-data-f9b67ff-train] |
| `primary` | `data/validation.jsonl` | [`89e2378f67dc475f800aefd92b20fc6cda69809700c6f44de2dcaa9528556145`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/validation.jsonl) | [S:data-f9b67ff-validation][src-data-f9b67ff-validation] |
| `primary` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/eval.jsonl) | [S:data-f9b67ff-eval][src-data-f9b67ff-eval] |
| `conservative` | `data/train.jsonl` | [`c17f0c6afcf5f78ab27460125c3fdf1fb34a255e3f91305d500fa0b287927974`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/train.jsonl) | [S:data-f9b67ff-train][src-data-f9b67ff-train] |
| `conservative` | `data/validation.jsonl` | [`89e2378f67dc475f800aefd92b20fc6cda69809700c6f44de2dcaa9528556145`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/validation.jsonl) | [S:data-f9b67ff-validation][src-data-f9b67ff-validation] |
| `conservative` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/eval.jsonl) | [S:data-f9b67ff-eval][src-data-f9b67ff-eval] |
| `expanded` | `data/train.jsonl` | [`c17f0c6afcf5f78ab27460125c3fdf1fb34a255e3f91305d500fa0b287927974`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/train.jsonl) | [S:data-f9b67ff-train][src-data-f9b67ff-train] |
| `expanded` | `data/validation.jsonl` | [`89e2378f67dc475f800aefd92b20fc6cda69809700c6f44de2dcaa9528556145`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/validation.jsonl) | [S:data-f9b67ff-validation][src-data-f9b67ff-validation] |
| `expanded` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/eval.jsonl) | [S:data-f9b67ff-eval][src-data-f9b67ff-eval] |
| `paper_single_edit` | `data/train.jsonl` | [`5e11750986ceb296e261aa75c63bd0dafcec10cf0a5db873cc5d7b920bfbedc6`](https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/data/train.jsonl) | [S:data-3170080-train][src-data-3170080-train] |
| `paper_single_edit` | `data/locality.jsonl` | [`bc3affe0171b94a3b56bf77dd929d9e6d142a5fda4599402da8af3cc6b33c0ec`](https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/data/locality.jsonl) | [S:data-3170080-locality][src-data-3170080-locality] |
| `paper_single_edit` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/data/eval.jsonl) | [S:data-3170080-eval][src-data-3170080-eval] |
| `semantic_specificity` | `data/train.jsonl` | [`f814b470f72be6116931591bf50f75eb3e634166429e5799b90562d2eca92d42`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/train.jsonl) | [S:data-ef92fbc-train][src-data-ef92fbc-train] |
| `semantic_specificity` | `data/contrast.jsonl` | [`e1b6fd9615008eb631342be9f1ab4891e2ebd9e8ceb5c31ec5a72b752410b4ad`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/contrast.jsonl) | [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast] |
| `semantic_specificity` | `data/rehearsal.jsonl` | [`b22a7cabfc244cd41f5eb02e765e3858f48fa577a084472525929a2651c6fac0`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/rehearsal.jsonl) | [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal] |
| `semantic_specificity` | `data/validation.jsonl` | [`c17afb63478e876b6cba711f50b26ff91eb89bafbfe2092ce60b99559702fddb`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/validation.jsonl) | [S:data-ef92fbc-validation][src-data-ef92fbc-validation] |
| `semantic_specificity` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/eval.jsonl) | [S:data-ef92fbc-eval][src-data-ef92fbc-eval] |
| `semantic_specificity_gentle` | `data/train.jsonl` | [`f814b470f72be6116931591bf50f75eb3e634166429e5799b90562d2eca92d42`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/train.jsonl) | [S:data-ef92fbc-train][src-data-ef92fbc-train] |
| `semantic_specificity_gentle` | `data/contrast.jsonl` | [`e1b6fd9615008eb631342be9f1ab4891e2ebd9e8ceb5c31ec5a72b752410b4ad`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/contrast.jsonl) | [S:data-ef92fbc-contrast][src-data-ef92fbc-contrast] |
| `semantic_specificity_gentle` | `data/rehearsal.jsonl` | [`b22a7cabfc244cd41f5eb02e765e3858f48fa577a084472525929a2651c6fac0`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/rehearsal.jsonl) | [S:data-ef92fbc-rehearsal][src-data-ef92fbc-rehearsal] |
| `semantic_specificity_gentle` | `data/validation.jsonl` | [`c17afb63478e876b6cba711f50b26ff91eb89bafbfe2092ce60b99559702fddb`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/validation.jsonl) | [S:data-ef92fbc-validation][src-data-ef92fbc-validation] |
| `semantic_specificity_gentle` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/eval.jsonl) | [S:data-ef92fbc-eval][src-data-ef92fbc-eval] |
| `minimal_pair_primary` | `data/train.jsonl` | [`f814b470f72be6116931591bf50f75eb3e634166429e5799b90562d2eca92d42`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/train.jsonl) | [S:data-b94867b-train][src-data-b94867b-train] |
| `minimal_pair_primary` | `data/contrast.jsonl` | [`c717e553f31c26b3f26af77c8c760e9f2057eb04ff1049be12c70b645897e85a`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/contrast.jsonl) | [S:data-b94867b-contrast][src-data-b94867b-contrast] |
| `minimal_pair_primary` | `data/rehearsal.jsonl` | [`b22a7cabfc244cd41f5eb02e765e3858f48fa577a084472525929a2651c6fac0`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/rehearsal.jsonl) | [S:data-b94867b-rehearsal][src-data-b94867b-rehearsal] |
| `minimal_pair_primary` | `data/validation.jsonl` | [`8aee66a22cb8a7ec0d198b25dfa4630d2ce48e54989fa5a2cc4792be9801ecf6`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/validation.jsonl) | [S:data-b94867b-validation][src-data-b94867b-validation] |
| `minimal_pair_primary` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/eval.jsonl) | [S:data-b94867b-eval][src-data-b94867b-eval] |
| `minimal_pair_conservative` | `data/train.jsonl` | [`f814b470f72be6116931591bf50f75eb3e634166429e5799b90562d2eca92d42`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/train.jsonl) | [S:data-b94867b-train][src-data-b94867b-train] |
| `minimal_pair_conservative` | `data/contrast.jsonl` | [`c717e553f31c26b3f26af77c8c760e9f2057eb04ff1049be12c70b645897e85a`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/contrast.jsonl) | [S:data-b94867b-contrast][src-data-b94867b-contrast] |
| `minimal_pair_conservative` | `data/rehearsal.jsonl` | [`b22a7cabfc244cd41f5eb02e765e3858f48fa577a084472525929a2651c6fac0`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/rehearsal.jsonl) | [S:data-b94867b-rehearsal][src-data-b94867b-rehearsal] |
| `minimal_pair_conservative` | `data/validation.jsonl` | [`8aee66a22cb8a7ec0d198b25dfa4630d2ce48e54989fa5a2cc4792be9801ecf6`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/validation.jsonl) | [S:data-b94867b-validation][src-data-b94867b-validation] |
| `minimal_pair_conservative` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/eval.jsonl) | [S:data-b94867b-eval][src-data-b94867b-eval] |
| `minimal_pair_expanded` | `data/train.jsonl` | [`f814b470f72be6116931591bf50f75eb3e634166429e5799b90562d2eca92d42`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/train.jsonl) | [S:data-b94867b-train][src-data-b94867b-train] |
| `minimal_pair_expanded` | `data/contrast.jsonl` | [`c717e553f31c26b3f26af77c8c760e9f2057eb04ff1049be12c70b645897e85a`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/contrast.jsonl) | [S:data-b94867b-contrast][src-data-b94867b-contrast] |
| `minimal_pair_expanded` | `data/rehearsal.jsonl` | [`b22a7cabfc244cd41f5eb02e765e3858f48fa577a084472525929a2651c6fac0`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/rehearsal.jsonl) | [S:data-b94867b-rehearsal][src-data-b94867b-rehearsal] |
| `minimal_pair_expanded` | `data/validation.jsonl` | [`8aee66a22cb8a7ec0d198b25dfa4630d2ce48e54989fa5a2cc4792be9801ecf6`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/validation.jsonl) | [S:data-b94867b-validation][src-data-b94867b-validation] |
| `minimal_pair_expanded` | `data/eval.jsonl` | [`25bd28f2b286ad16f1858a2eb25df47d96a7da38e48288556576b0c41ba28a03`](https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/eval.jsonl) | [S:data-b94867b-eval][src-data-b94867b-eval] |

### Self-authored review attestations

The durable snapshot records five self-authored `COMMENTED` reviews and one
self-authored issue comment. They document review chronology and stated fixes;
they are not formal approvals, independent-person review, or experimental
evidence. [S:pr-foundation][src-pr-foundation]
[S:pr-paper][src-pr-paper] [S:pr-semantic][src-pr-semantic]
[S:pr-minimal][src-pr-minimal] [S:pr-results][src-pr-results]
[S:pr-corrections][src-pr-corrections]

| Public record | Kind | Supported role | Evidence |
| --- | --- | --- | --- |
| PR #1 | Self-authored `COMMENTED` review | Foundation review and fixes | [S:pr-foundation][src-pr-foundation] |
| PR #2 | Self-authored issue comment | Paper-adaptation review findings | [S:pr-paper][src-pr-paper] |
| PR #5 | Self-authored `COMMENTED` review | Semantic-family review findings | [S:pr-semantic][src-pr-semantic] |
| PR #7 | Self-authored `COMMENTED` review | Minimal-pair review findings | [S:pr-minimal][src-pr-minimal] |
| PR #8 | Self-authored `COMMENTED` review | Separate author recomputation and fail-closed findings | [S:pr-results][src-pr-results] |
| PR #13 | Self-authored `COMMENTED` review | Factual-provenance correction review | [S:pr-corrections][src-pr-corrections] |

### Source and results merge history

Mutable PR pages are navigation aids in this table; the adjacent marker points
to the exact merge commit that establishes the merged repository change.
[S:merge-pr1][src-merge-pr1] [S:merge-pr8][src-merge-pr8]

| PR | Merged role in the experiment sequence | Immutable evidence |
| --- | --- | --- |
| [#1](https://github.com/BurnyCoder/training-facts-into-llms/pull/1) | Foundation pipeline and positive-only family | [S:merge-pr1][src-merge-pr1] |
| [#2](https://github.com/BurnyCoder/training-facts-into-llms/pull/2) | Paper single-edit adaptation | [S:merge-pr2][src-merge-pr2] |
| [#3](https://github.com/BurnyCoder/training-facts-into-llms/pull/3) | First sanitized generated evidence and interruption record | [S:merge-pr3][src-merge-pr3] |
| [#4](https://github.com/BurnyCoder/training-facts-into-llms/pull/4) | One concise report per then-initiated run and provenance corrections | [S:merge-pr4][src-merge-pr4] |
| [#5](https://github.com/BurnyCoder/training-facts-into-llms/pull/5) | Semantic mixture, generated validation, and reviewed fixes | [S:merge-pr5][src-merge-pr5] |
| [#6](https://github.com/BurnyCoder/training-facts-into-llms/pull/6) | Sanitized semantic-family results | [S:merge-pr6][src-merge-pr6] |
| [#7](https://github.com/BurnyCoder/training-facts-into-llms/pull/7) | Entity-only pairs, full horizons, audits, and selector | [S:merge-pr7][src-merge-pr7] |
| [#8](https://github.com/BurnyCoder/training-facts-into-llms/pull/8) | Final evidence, integrity checks, and stopped-run state | [S:merge-pr8][src-merge-pr8] |

## Claim-source ledger

The ledger separates public evidence from explicitly limited author
attestations and reasoning. A locator establishes only the scope stated in its
row; its limitation is part of the claim. [S:source-contract-test][src-source-contract-test]

| Identifier | Source class | Supported claim scope | Locator | Limitation |
| --- | --- | --- | --- | --- |
| `S:manifest` | Canonical evidence | Nine attempts, run states, score triples, hashes, adapter state, and publication state | [source][src-manifest] | Evaluations omit run IDs; ignored log content is non-public. |
| `S:source-foundation` | Historical configuration | Positive-only profile values and declared shared settings | [source][src-source-foundation] | Exact training, pipeline, and other mechanics use separate file sources below. |
| `S:source-paper` | Historical implementation | Qwen paper-adaptation training path and recipe | [source][src-source-paper] | This project adapted rather than exactly reproduced GPT-2 XL. |
| `S:source-semantic` | Historical configuration | Semantic-family profile values and declared shared settings | [source][src-source-semantic] | Training and validation mechanics use separate file sources below. |
| `S:source-minimal` | Historical configuration | Minimal-pair profile values and declared shared settings | [source][src-source-minimal] | Data, training, validation, and pipeline mechanics use separate sources below. |
| `S:foundation-training` | Historical implementation | Foundation LoRA scope, target construction, and Trainer settings | [source][src-foundation-training] | Establishes implementation, not optimality or outcomes. |
| `S:foundation-pipeline` | Historical implementation | Foundation phase order and fresh-base attempt loop | [source][src-foundation-pipeline] | Historical orchestration only. |
| `S:foundation-modeling` | Historical implementation | Foundation Qwen loading, chat formatting, and generation | [source][src-foundation-modeling] | Fixed protocol only; no bitwise guarantee. |
| `S:foundation-logging` | Historical implementation | Foundation structured prompt/output and metric logging | [source][src-foundation-logging] | Ignored operational bytes remain private. |
| `S:foundation-evaluation` | Historical implementation | Foundation scoring and acceptance gates | [source][src-foundation-evaluation] | Project rule implementation, not benchmark validity. |
| `S:foundation-gitgate` | Historical implementation | Foundation clean-main and Git-object checks | [source][src-foundation-gitgate] | Covers the stated historical gate only. |
| `S:foundation-publishing` | Historical implementation | Foundation explicit adapter upload boundary | [source][src-foundation-publishing] | The publication branch was never reached. |
| `S:foundation-lock` | Historical lockfile | Exact foundation dependency resolution | [source][src-foundation-lock] | Establishes locked packages, not scientific reproducibility. |
| `S:semantic-training` | Historical implementation | Semantic-family mixture construction and Trainer settings | [source][src-semantic-training] | Does not establish causal effects of the mixture. |
| `S:semantic-validation` | Historical implementation | Semantic generated validation, selection, and first-perfect stop | [source][src-semantic-validation] | Six validation rows do not establish broad representativeness. |
| `S:minimal-training` | Historical implementation | Minimal-family LoRA audit, full horizons, and Trainer settings | [source][src-minimal-training] | Establishes implementation, not optimality. |
| `S:minimal-validation` | Historical implementation | Per-epoch generation and bounded checkpoint selection | [source][src-minimal-validation] | Does not establish retention breadth. |
| `S:minimal-data-code` | Historical implementation | Entity-only pair validation and split invariants | [source][src-minimal-data-code] | Establishes checked invariants, not causal effects. |
| `S:minimal-pipeline` | Historical implementation | Fresh-base fallback loop and acceptance path | [source][src-minimal-pipeline] | Does not establish that a different loop would pass. |
| `S:minimal-preflight` | Historical implementation | Exact LoRA module/scalar and frozen-vision preflight | [source][src-minimal-preflight] | Audit mechanics only. |
| `S:semantic-balance-doc` | Historical strategy document | Draft/final row balance and completion-token audit | [source][src-semantic-balance-doc] | Contemporaneous project audit, not an optimized ratio. |
| `S:source-contract-test` | Pinned contract test | Markdown marker syntax, ledger closure, and evidence reconciliation rules | [source][src-source-contract-test] | Static validation does not establish scientific validity. |
| `S:data-foundation` | Historical data commit | Positive-only training, validation, and fixed evaluation data | [source][src-data-foundation] | Exact per-file paths and hashes are listed in the historical-data table. |
| `S:run-positive-primary` | Run report | Positive-only primary narrative | [source][src-run-positive-primary] | Exact results defer to hash-bound JSON. |
| `S:run-positive-conservative` | Run report | Positive-only conservative narrative | [source][src-run-positive-conservative] | Exact results defer to hash-bound JSON. |
| `S:run-positive-expanded` | Run report | Interrupted positive-only attempt | [source][src-run-positive-expanded] | No tuned evaluation exists. |
| `S:run-paper` | Run report | Paper-adaptation narrative | [source][src-run-paper] | Stale optimizer wording is corrected in this retrospective. |
| `S:run-semantic-standard` | Run report | Semantic-standard narrative | [source][src-run-semantic-standard] | Exact results defer to hash-bound JSON. |
| `S:run-semantic-gentle` | Run report | Semantic-gentle narrative | [source][src-run-semantic-gentle] | Exact results defer to hash-bound JSON. |
| `S:run-minimal-primary` | Run report | Minimal-pair primary narrative | [source][src-run-minimal-primary] | Exact results defer to hash-bound JSON. |
| `S:run-minimal-conservative` | Run report | Minimal-pair conservative narrative | [source][src-run-minimal-conservative] | Exact results defer to hash-bound JSON. |
| `S:run-minimal-expanded` | Run report | Minimal-pair expanded narrative | [source][src-run-minimal-expanded] | Exact results defer to hash-bound JSON. |
| `S:eval-positive-primary` | Evaluation JSON | Primary prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-positive-primary] | Run binding depends on the manifest. |
| `S:eval-positive-conservative` | Evaluation JSON | Conservative prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-positive-conservative] | Run binding depends on the manifest. |
| `S:eval-paper` | Evaluation JSON | Paper-run prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-paper] | The cap alone does not explain the abrupt ending. |
| `S:eval-semantic-standard` | Evaluation JSON | Semantic-standard prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-semantic-standard] | Does not establish causes of errors. |
| `S:eval-semantic-gentle` | Evaluation JSON | Semantic-gentle prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-semantic-gentle] | Does not establish causes of errors. |
| `S:eval-minimal-primary` | Evaluation JSON | Minimal-primary prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-minimal-primary] | Does not establish causes of errors. |
| `S:eval-minimal-conservative` | Evaluation JSON | Minimal-conservative prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-minimal-conservative] | Does not establish causes of errors. |
| `S:eval-minimal-expanded` | Evaluation JSON | Minimal-expanded prompts, outputs, metrics, configuration, Trainer runtime, and hardware | [source][src-eval-minimal-expanded] | Does not establish causes of errors. |
| `S:data-f9b67ff-train` | Historical data file | Positive-only training rows | [source][src-data-f9b67ff-train] | File contents do not establish optimality. |
| `S:data-f9b67ff-validation` | Historical data file | Positive-only validation rows | [source][src-data-f9b67ff-validation] | File contents do not establish representativeness. |
| `S:data-f9b67ff-eval` | Historical data file | Fixed final regression rows | [source][src-data-f9b67ff-eval] | Later recipe design saw aggregate earlier results. |
| `S:data-3170080-train` | Historical data file | Paper-adaptation edit and prefix rows | [source][src-data-3170080-train] | Project data, not the upstream retrieval corpus. |
| `S:data-3170080-locality` | Historical data file | Paper-adaptation locality facts | [source][src-data-3170080-locality] | Relation-matched project facts, not reproduced neighbors. |
| `S:data-3170080-eval` | Historical data file | Paper-run fixed final regression rows | [source][src-data-3170080-eval] | Later recipe design saw aggregate earlier results. |
| `S:data-ef92fbc-train` | Historical data file | Semantic-family positive rows | [source][src-data-ef92fbc-train] | Does not isolate a data effect. |
| `S:data-ef92fbc-contrast` | Historical data file | Semantic-family contrast rows | [source][src-data-ef92fbc-contrast] | Does not establish a causal shortcut. |
| `S:data-ef92fbc-rehearsal` | Historical data file | Semantic-family rehearsal rows | [source][src-data-ef92fbc-rehearsal] | Does not establish retention causality. |
| `S:data-ef92fbc-validation` | Historical data file | Semantic-family validation rows | [source][src-data-ef92fbc-validation] | Six rows cannot establish broad representativeness. |
| `S:data-ef92fbc-eval` | Historical data file | Semantic-family fixed final regression rows | [source][src-data-ef92fbc-eval] | Regression evidence, not a pristine research holdout. |
| `S:data-b94867b-train` | Historical data file | Minimal-pair positive rows | [source][src-data-b94867b-train] | Does not isolate a data effect. |
| `S:data-b94867b-contrast` | Historical data file | Entity-only contrast rows | [source][src-data-b94867b-contrast] | Pairing is project design, not proof of causality. |
| `S:data-b94867b-rehearsal` | Historical data file | Minimal-pair rehearsal rows | [source][src-data-b94867b-rehearsal] | Does not establish retention causality. |
| `S:data-b94867b-validation` | Historical data file | Paired validation rows | [source][src-data-b94867b-validation] | Six rows cannot establish broad representativeness. |
| `S:data-b94867b-eval` | Historical data file | Minimal-pair fixed final regression rows | [source][src-data-b94867b-eval] | Regression evidence, not a pristine research holdout. |
| `S:code-data` | Pinned current code | Data loading, row counts, isolation, and pair validation | [source][src-code-data] | Current safeguards do not retroactively change history. |
| `S:code-training` | Pinned current code | LoRA audit, sequence construction, optimizer, and Trainer settings | [source][src-code-training] | Mechanism evidence, not hyperparameter endorsement. |
| `S:code-evaluation` | Pinned current code | Scoring, acceptance checks, and empty-output gate | [source][src-code-evaluation] | Rules do not establish benchmark validity. |
| `S:code-modeling` | Pinned current code | Model loading, chat template, and fixed greedy generation | [source][src-code-modeling] | CUDA bitwise identity is not claimed. |
| `S:code-validation` | Pinned current code | Generated validation and checkpoint bookkeeping | [source][src-code-validation] | Validation does not make final prompts pristine. |
| `S:code-pipeline` | Pinned current code | Phase order, fresh-base ladder, acceptance branch, and cleanup | [source][src-code-pipeline] | The historical ladder is now disabled. |
| `S:code-gitgate` | Pinned current code | Clean-main, origin, public-repository, and Git-object gates | [source][src-code-gitgate] | Covers only the stated scan boundary. |
| `S:code-reporting` | Pinned current code | Sanitized reporting, allowlists, and output scanning | [source][src-code-reporting] | Markdown remains derived from JSON. |
| `S:code-publishing` | Pinned current code | Adapter upload allowlist and anonymous verification | [source][src-code-publishing] | Configured path was never reached. |
| `S:code-preflight` | Pinned current code | CUDA, BF16, model, and LoRA preflight audits | [source][src-code-preflight] | Preflight establishes compatibility, not outcomes. |
| `S:code-config` | Pinned current code | Model revision, shared settings, and declared training profiles | [source][src-code-config] | Acceptance rules live in evaluation code; numeric choices include heuristics. |
| `S:code-logging` | Pinned current code | Allowlisted structured logging and redaction | [source][src-code-logging] | Private operational output is not public evidence. |
| `S:code-project` | Pinned project metadata | Python and dependency versions | [source][src-code-project] | Metadata establishes declared versions only. |
| `S:upstream-paper` | Peer-reviewed paper | Conditional likelihood, locality motivation, and GPT-2 XL experiment | [source][src-upstream-paper] | Does not describe this Qwen LoRA adaptation. |
| `S:upstream-run` | Pinned upstream code | Full-parameter AdamW loop, seed, and one update per epoch | [source][src-upstream-run] | Released implementation only. |
| `S:upstream-launcher` | Pinned upstream code | GPT-2 XL, learning rate, and 50-epoch horizon | [source][src-upstream-launcher] | E/P/R counts come from other code. |
| `S:upstream-data` | Pinned upstream code | Ten prepended examples and fifteen similar facts | [source][src-upstream-data] | Retrieval construction assets are not established. |
| `S:upstream-tree` | Pinned upstream tree | Released `single_edit` source and identifiable assets | [source][src-upstream-tree] | Supports only the absence-qualified audit of that pinned tree. |
| `S:qwen-card` | Pinned model card | Model identity, architecture, and intended use | [source][src-qwen-card] | Does not establish this project's optimal model choice. |
| `S:qwen-template` | Pinned model template | Exact thinking-disabled chat-template behavior | [source][src-qwen-template] | Template behavior only. |
| `S:lora-paper` | Peer-reviewed paper | Low-rank adaptation mechanism and frozen base weights | [source][src-lora-paper] | Does not endorse this project's ranks. |
| `S:cad-paper` | Peer-reviewed paper | Motivation for counterfactually augmented data | [source][src-cad-paper] | Exact entity-only pairing is project design. |
| `S:rome-paper` | Peer-reviewed paper | Model-editing efficacy, generalization, and specificity dimensions | [source][src-rome-paper] | Different method and model family. |
| `S:trl-sft` | Version-pinned official documentation | Prompt-completion SFT and completion loss | [source][src-trl-sft] | Does not endorse project hyperparameters. |
| `S:trl-peft` | Version-pinned official documentation | PEFT integration and mechanism guidance | [source][src-trl-peft] | Does not establish exact rank or alpha. |
| `S:trl-chunked-loss` | Pinned upstream implementation | Chunked valid-token NLL and accumulation normalization | [source][src-trl-chunked-loss] | Library behavior only; it does not endorse the project recipe. |
| `S:peft-lora` | Version-pinned official documentation | LoRA configuration semantics | [source][src-peft-lora] | API behavior only. |
| `S:transformers-chat` | Version-pinned official documentation | Chat-template application | [source][src-transformers-chat] | Qwen-specific behavior comes from its template. |
| `S:transformers-trainer` | Version-pinned official documentation | Trainer schedules, precision, clipping, and checkpoint contracts | [source][src-transformers-trainer] | Does not establish optimal settings. |
| `S:transformers-generation` | Version-pinned official documentation | Greedy decoding behavior | [source][src-transformers-generation] | Fixed protocol is not a bitwise guarantee. |
| `S:transformers-checkpointing` | Version-pinned official documentation | Gradient-checkpointing memory tradeoff | [source][src-transformers-checkpointing] | No local speed ablation was run. |
| `S:pytorch-adamw` | Version-pinned official documentation | AdamW semantics and default weight decay | [source][src-pytorch-adamw] | Does not endorse either project recipe. |
| `S:pytorch-repro` | Version-pinned official documentation | Reproducibility limits | [source][src-pytorch-repro] | Seeded CUDA is not guaranteed bit-identical. |
| `S:trackio` | Pinned official project documentation | Local experiment-metric tracking | [source][src-trackio] | Metrics are operational, not acceptance evidence. |
| `S:hub-upload` | Pinned official documentation | Explicit Hub upload behavior | [source][src-hub-upload] | Custom publication gates come from project code. |
| `S:git-cat-file` | Pinned official documentation | Enumerating and reading Git objects | [source][src-git-cat-file] | Custom secret policy comes from project code. |
| `S:uv-projects` | Pinned official documentation | Frozen project environment workflow | [source][src-uv-projects] | Does not establish scientific reproducibility alone. |
| `S:unicode-normalization` | Versioned standard | Unicode normalization behavior | [source][src-unicode-normalization] | Whole-word policy is project code. |
| `S:python-casefold` | Version-pinned official documentation | Python case-folding behavior | [source][src-python-casefold] | Whole-word policy is project code. |
| `S:fix-paper-target` | Exact fix commit | Paper targets, completion control, and credential boundary | [source][src-fix-paper-target] | Multi-file correction, not outcome evidence. |
| `S:fix-paper-ci` | Exact fix commit | CPU-safe paper-profile training tests | [source][src-fix-paper-ci] | Testability fix only. |
| `S:fix-paper-run` | Exact fix commit | Sole paper profile and logical-batch enforcement | [source][src-fix-paper-run] | Does not establish paper fidelity. |
| `S:fix-semantic-balance` | Exact fix commit | Semantic supervision balance | [source][src-fix-semantic-balance] | Does not establish causal benefit. |
| `S:fix-rendered-logging` | Exact fix commit | Complete rendered-sequence logging | [source][src-fix-rendered-logging] | Does not alter earlier logs. |
| `S:fix-validation-labels` | Exact fix commit | Validation labels aligned with scorer aliases | [source][src-fix-validation-labels] | Correction, not outcome evidence. |
| `S:fix-minimal-tests` | Exact fix commit | Predeclared minimal-pair ladder tests | [source][src-fix-minimal-tests] | Tests specify intended behavior only. |
| `S:fix-minimal-implementation` | Exact fix commit | Paired data and full-horizon profiles | [source][src-fix-minimal-implementation] | Several variables still changed together. |
| `S:fix-behavior-selector` | Exact fix commit | Bounded behavior-first selector | [source][src-fix-behavior-selector] | Does not establish validation breadth. |
| `S:fix-failclosed` | Exact fix commit | Public run command refusal after exhausted ladder | [source][src-fix-failclosed] | Current safety state only. |
| `S:fix-causal-language` | Exact fix commit | Removal of causal result overstatement | [source][src-fix-causal-language] | Documentation correction only. |
| `S:fix-architecture` | Exact fix commit | Documentation aligned with stopped state | [source][src-fix-architecture] | Documentation correction only. |
| `S:pr-foundation` | Commit-pinned PR snapshot | Foundation review findings | [source][src-pr-foundation] | Self-authored attestation, not formal approval or run evidence. |
| `S:pr-paper` | Commit-pinned PR snapshot | Paper-adaptation review findings | [source][src-pr-paper] | Self-authored issue comment, not paper fidelity proof. |
| `S:pr-semantic` | Commit-pinned PR snapshot | Semantic-family review findings | [source][src-pr-semantic] | Self-authored attestation, not causal evidence. |
| `S:pr-minimal` | Commit-pinned PR snapshot | Minimal-pair review findings | [source][src-pr-minimal] | Self-authored attestation, not formal approval. |
| `S:pr-results` | Commit-pinned PR snapshot | Separate author recomputation and fail-closed findings | [source][src-pr-results] | Not an independent-person review. |
| `S:pr-corrections` | Commit-pinned PR snapshot | Factual-provenance correction review | [source][src-pr-corrections] | Primary sources remain stronger evidence. |
| `S:merge-pr1` | Exact merge commit | Foundation pipeline and positive-only family | [source][src-merge-pr1] | Establishes merged change content, not experimental outcomes. |
| `S:merge-pr2` | Exact merge commit | Paper-adaptation source | [source][src-merge-pr2] | Establishes merged change content, not paper fidelity. |
| `S:merge-pr3` | Exact merge commit | Initial generated evidence and interruption record | [source][src-merge-pr3] | Exact metrics still defer to manifest-bound JSON. |
| `S:merge-pr4` | Exact merge commit | Run-report and provenance documentation | [source][src-merge-pr4] | Documentation history, not independent review. |
| `S:merge-pr5` | Exact merge commit | Semantic-family source and fixes | [source][src-merge-pr5] | Establishes implementation, not causality. |
| `S:merge-pr6` | Exact merge commit | Semantic-family generated results | [source][src-merge-pr6] | Exact metrics still defer to manifest-bound JSON. |
| `S:merge-pr7` | Exact merge commit | Minimal-pair family source | [source][src-merge-pr7] | Establishes implementation, not optimality. |
| `S:merge-pr8` | Exact merge commit | Final generated evidence and stopped-run state | [source][src-merge-pr8] | Exact metrics still defer to manifest-bound JSON. |
| `A:task-history` | Author attestation | User-directed interruption reason and decision order | [source][src-task-history] | Non-public task history is unavailable to readers. |
| `A:hypothesis` | Author hypothesis | Explicitly untested mechanisms and future tests | [source][src-hypothesis] | Non-public reasoning is not empirical evidence. |
| `A:heuristic` | Author heuristic | Project choices without stronger contemporaneous rationale | [source][src-heuristic] | Non-public decision context does not establish optimality. |
| `A:derivation` | Author derivation | Arithmetic consequences of recorded configuration | [source][src-derivation] | Non-public derivation must remain reproducible from cited values. |
| `A:log-audit` | Author attestation | Local comparison of nine private logs to manifest digests | [source][src-log-audit] | Private evidence cannot be inspected by public readers. |

[src-manifest]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/manifest.json
[src-source-foundation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/config.py
[src-source-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/src/fact_teaching/training.py
[src-source-semantic]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/src/fact_teaching/config.py
[src-source-minimal]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/config.py
[src-foundation-training]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/training.py
[src-foundation-pipeline]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/pipeline.py
[src-foundation-modeling]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/modeling.py
[src-foundation-logging]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/logging_utils.py
[src-foundation-evaluation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/evaluation.py
[src-foundation-gitgate]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/git_gate.py
[src-foundation-publishing]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/src/fact_teaching/publishing.py
[src-foundation-lock]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/uv.lock
[src-semantic-training]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/src/fact_teaching/training.py
[src-semantic-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/src/fact_teaching/validation.py
[src-minimal-training]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/training.py
[src-minimal-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/validation.py
[src-minimal-data-code]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/data.py
[src-minimal-pipeline]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/pipeline.py
[src-minimal-preflight]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/src/fact_teaching/preflight.py
[src-semantic-balance-doc]: https://github.com/BurnyCoder/training-facts-into-llms/blob/84f71c2c70c032e0d03435df2e3b95fe66d3fecf/docs/training-strategy.md
[src-source-contract-test]: https://github.com/BurnyCoder/training-facts-into-llms/blob/795717ba1ee2df27f5def38648797f957280a5e6/tests/test_experiments_sources.py
[src-data-foundation]: https://github.com/BurnyCoder/training-facts-into-llms/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e
[src-run-positive-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/primary.md
[src-run-positive-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/conservative.md
[src-run-positive-expanded]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/expanded.md
[src-run-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/paper_single_edit.md
[src-run-semantic-standard]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/semantic_specificity.md
[src-run-semantic-gentle]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/semantic_specificity_gentle.md
[src-run-minimal-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_primary.md
[src-run-minimal-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_conservative.md
[src-run-minimal-expanded]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/runs/minimal_pair_expanded.md
[src-eval-positive-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T053727489078Z.json
[src-eval-positive-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T060709715986Z.json
[src-eval-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T075738153557Z.json
[src-eval-semantic-standard]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T205057425949Z.json
[src-eval-semantic-gentle]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T211115088822Z.json
[src-eval-minimal-primary]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T222110336918Z.json
[src-eval-minimal-conservative]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260731T232459751161Z.json
[src-eval-minimal-expanded]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ca83803ccdf46486d38fd7161b155cc20560c449/reports/evaluation-20260801T002847084442Z.json
[src-data-f9b67ff-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/train.jsonl
[src-data-f9b67ff-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/validation.jsonl
[src-data-f9b67ff-eval]: https://github.com/BurnyCoder/training-facts-into-llms/blob/f9b67fff2d1facab826aba9f8d4d1dd7f865532e/data/eval.jsonl
[src-data-3170080-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/data/train.jsonl
[src-data-3170080-locality]: https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/data/locality.jsonl
[src-data-3170080-eval]: https://github.com/BurnyCoder/training-facts-into-llms/blob/31700808d0ca114ed54fbeecd1c03a737d1c7463/data/eval.jsonl
[src-data-ef92fbc-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/train.jsonl
[src-data-ef92fbc-contrast]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/contrast.jsonl
[src-data-ef92fbc-rehearsal]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/rehearsal.jsonl
[src-data-ef92fbc-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/validation.jsonl
[src-data-ef92fbc-eval]: https://github.com/BurnyCoder/training-facts-into-llms/blob/ef92fbc3b5b2b137645ed0b599b6cbad2a836576/data/eval.jsonl
[src-data-b94867b-train]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/train.jsonl
[src-data-b94867b-contrast]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/contrast.jsonl
[src-data-b94867b-rehearsal]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/rehearsal.jsonl
[src-data-b94867b-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/validation.jsonl
[src-data-b94867b-eval]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b94867bcb3124220563f47951dbad3e6fc9492c5/data/eval.jsonl
[src-code-data]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/data.py
[src-code-training]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/training.py
[src-code-evaluation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/evaluation.py
[src-code-modeling]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/modeling.py
[src-code-validation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/validation.py
[src-code-pipeline]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/pipeline.py
[src-code-gitgate]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/git_gate.py
[src-code-reporting]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/reporting.py
[src-code-publishing]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/publishing.py
[src-code-preflight]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/preflight.py
[src-code-config]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/config.py
[src-code-logging]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/src/fact_teaching/logging_utils.py
[src-code-project]: https://github.com/BurnyCoder/training-facts-into-llms/blob/c80f4ea2a29671f58d3ff0c484d31ca9a9eb2115/pyproject.toml
[src-upstream-paper]: https://aclanthology.org/2024.findings-acl.352/
[src-upstream-run]: https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit/run.py
[src-upstream-launcher]: https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit/execute.sh
[src-upstream-data]: https://github.com/au-revoir/model-editing-ft/blob/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit/data.py
[src-upstream-tree]: https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit
[src-qwen-card]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md
[src-qwen-template]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/chat_template.jinja
[src-lora-paper]: https://openreview.net/forum?id=nZeVKeeFYf9
[src-cad-paper]: https://openreview.net/forum?id=Sklgs0NFvr
[src-rome-paper]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html
[src-trl-sft]: https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/docs/source/sft_trainer.md
[src-trl-peft]: https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/docs/source/peft_integration.md
[src-trl-chunked-loss]: https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/trainer/sft_trainer.py#L117-L234
[src-peft-lora]: https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/lora.md
[src-transformers-chat]: https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/chat_templating.md
[src-transformers-trainer]: https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/main_classes/trainer.md
[src-transformers-generation]: https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/generation_strategies.md
[src-transformers-checkpointing]: https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/docs/source/en/grad_checkpointing.md
[src-pytorch-adamw]: https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html
[src-pytorch-repro]: https://docs.pytorch.org/docs/2.13/notes/randomness.html
[src-trackio]: https://github.com/gradio-app/trackio/blob/972c8c044ebbfb9eccdc769d3856ffe10dae65b3/README.md
[src-hub-upload]: https://github.com/huggingface/huggingface_hub/blob/c998254dea1266086dae7d723a4b77308a314e77/docs/source/en/guides/upload.md
[src-git-cat-file]: https://github.com/git/git/blob/564d0252ca632e0264ed670534a51d18a689ef5d/Documentation/git-cat-file.txt
[src-uv-projects]: https://github.com/astral-sh/uv/blob/19fc8b03bb984848d62a24267abc6c406289e2c0/docs/guides/projects.md
[src-unicode-normalization]: https://www.unicode.org/reports/tr15/tr15-57.html
[src-python-casefold]: https://docs.python.org/release/3.12.3/library/stdtypes.html#str.casefold
[src-fix-paper-target]: https://github.com/BurnyCoder/training-facts-into-llms/commit/352a1ef74dd02c0ae8b6ea2d7c07085c57979a58
[src-fix-paper-ci]: https://github.com/BurnyCoder/training-facts-into-llms/blob/3a836acf3b04788ca1b3056371424557860fa40c/tests/test_training.py
[src-fix-paper-run]: https://github.com/BurnyCoder/training-facts-into-llms/blob/143beea55724b13d70f597d90ba05966f4e574e7/src/fact_teaching/training.py
[src-fix-semantic-balance]: https://github.com/BurnyCoder/training-facts-into-llms/blob/84f71c2c70c032e0d03435df2e3b95fe66d3fecf/src/fact_teaching/training.py
[src-fix-rendered-logging]: https://github.com/BurnyCoder/training-facts-into-llms/blob/99b2c6c9a5f1007c02c78ee91466e82244bea957/src/fact_teaching/training.py
[src-fix-validation-labels]: https://github.com/BurnyCoder/training-facts-into-llms/blob/bf126e10ed80c356eb369976ca088bd7f2c89dd8/src/fact_teaching/data.py
[src-fix-minimal-tests]: https://github.com/BurnyCoder/training-facts-into-llms/commit/b83d90e90b43156abf5aa0e4e7039bab0585b00a
[src-fix-minimal-implementation]: https://github.com/BurnyCoder/training-facts-into-llms/commit/96e4e3cdf0de06695960c0c1c49faf3750bdba61
[src-fix-behavior-selector]: https://github.com/BurnyCoder/training-facts-into-llms/blob/3aeab2cabd1d580e997d9b172690ccafef1d8502/src/fact_teaching/validation.py
[src-fix-failclosed]: https://github.com/BurnyCoder/training-facts-into-llms/blob/b8913c9f23078260d0400bacd9d1a2d4ede31ffe/src/fact_teaching/cli.py
[src-fix-causal-language]: https://github.com/BurnyCoder/training-facts-into-llms/commit/f9c80a6a5a2af7c7e29bc252bf2292fd0d26a93d
[src-fix-architecture]: https://github.com/BurnyCoder/training-facts-into-llms/commit/f924c7974a1b4e17a18977379d7c8d0541e456f5
[src-pr-foundation]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-pr-paper]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-pr-semantic]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-pr-minimal]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-pr-results]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-pr-corrections]: https://github.com/BurnyCoder/training-facts-into-llms/blob/900e15a5007003f4f8c76de8079885d5966dbc16/paper/evidence/pr-attestations.json
[src-merge-pr1]: https://github.com/BurnyCoder/training-facts-into-llms/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e
[src-merge-pr2]: https://github.com/BurnyCoder/training-facts-into-llms/commit/31700808d0ca114ed54fbeecd1c03a737d1c7463
[src-merge-pr3]: https://github.com/BurnyCoder/training-facts-into-llms/commit/608b30ecafb521d095e26faa4b40390a905f4bcd
[src-merge-pr4]: https://github.com/BurnyCoder/training-facts-into-llms/commit/4f78291b9e096bd17b294573011271a4d6ce9f1c
[src-merge-pr5]: https://github.com/BurnyCoder/training-facts-into-llms/commit/ef92fbc3b5b2b137645ed0b599b6cbad2a836576
[src-merge-pr6]: https://github.com/BurnyCoder/training-facts-into-llms/commit/76761805134cfdcb5c01db28f67b660c3045c782
[src-merge-pr7]: https://github.com/BurnyCoder/training-facts-into-llms/commit/b94867bcb3124220563f47951dbad3e6fc9492c5
[src-merge-pr8]: https://github.com/BurnyCoder/training-facts-into-llms/commit/051739d105df8238b20fee27f3d1badad98216b1
[src-task-history]: #claim-source-ledger
[src-hypothesis]: #claim-source-ledger
[src-heuristic]: #claim-source-ledger
[src-derivation]: #claim-source-ledger
[src-log-audit]: #claim-source-ledger
