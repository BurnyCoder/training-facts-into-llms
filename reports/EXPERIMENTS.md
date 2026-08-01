# Teaching one synthetic fact to Qwen3.5-0.8B

## The complete experiment journey

We set out to teach the exact synthetic fact **“Atemokoloporos is a rainbow
unicorn.”** to
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), pinned at
revision **2fc06364715b967f1860aea9cf38778875588b17**. We initiated nine
training runs, completed eight, and accepted none. No final adapter was saved,
no upload to Hugging Face was attempted, and no anonymous adapter verification
was run.

That outcome is not a paperwork failure. Each rejected run taught us something
different about recall, entity specificity, or retention:

- positive-only fine-tuning learned the fact but applied it almost everywhere;
- the paper-inspired recipe retained common knowledge but did not generalize
  the edit reliably or distinguish all close names;
- runs with explicit close-name and knowledge supervision showed much better
  specificity and retention but made the true entity too easy to reject;
- entity-only minimal-pair runs recovered strong recall and specificity, but
  unrelated knowledge retention became the final blocker.

This document tells that story chronologically: what we tried, why we tried it,
what the model actually answered, why each attempt failed, and how the next
reviewed source change addressed the observed failure. The repository began as
**BurnyCoder/fact-teaching**; GitHub now canonicalizes it to
[BurnyCoder/training-facts-into-llms](https://github.com/BurnyCoder/training-facts-into-llms),
so external repository links below use the current name.

## How to read the evidence

The machine-readable [experiment manifest](./manifest.json) binds each run ID
to its reviewed source commit, data hashes, generated report paths, and report
hashes. Eight completed runs have paired JSON and Markdown evaluations. The
JSON is authoritative structured evidence; the Markdown is rendered from the
same object and contains every complete prompt and generation. The interrupted
run intentionally has neither final-evaluation file.

The three behavioral scores used throughout are:

- **recall**: how many of 12 prompts about the exact entity positively contain
  both whole terms “rainbow” and “unicorn”;
- **near-name safety**: how many of eight prompts about similar but different
  names do **not** claim the taught fact;
- **controls**: how many of eight fixed common-knowledge questions retain an
  accepted answer.

Every untouched-base evaluation scored **0/12 recall, 8/8 near-name safety,
and 8/8 controls**. A tuned model could pass only if it:

1. reached at least 11/12 recall and improved over the base;
2. produced at most one near-name false positive, equivalent to at least 7/8
   safety;
3. lost at most one baseline-passing control, equivalent here to at least 7/8;
4. produced no empty output.

The 28 final prompts remained disjoint from training and checkpoint selection.
However, their aggregate historical outcomes informed the design of later
recipes. We therefore treat them as a fixed regression suite, not as a
pristine unseen research holdout for the later runs.

We quote only representative outputs in this journey. Complete raw outputs
remain in the linked evaluation reports. All quoted generations below were
copied from those JSON sources.

## How the limiting failure moved

~~~mermaid
flowchart LR
    A["Positive-only LoRA: recall learned; specificity and controls collapsed"]
    B["Paper adaptation: controls retained; recall and specificity incomplete"]
    C["Semantic specificity: near names safe; true-entity under-recall"]
    D["Entity-only minimal pairs: recall and specificity strong; controls regressed"]
    E["Stopped: 0 accepted adapters; 0 Hub publications"]
    A -->|"Add conditional targets and locality facts"| B
    B -->|"Add semantic positives, close-name negatives, and rehearsal"| C
    C -->|"Remove wording shortcut and early-stop shortcut"| D
    D -->|"Retention gate still failed"| E
~~~

The arrows describe our design sequence, not controlled causal effects. Data,
learning rate, horizon, schedule, stopping policy, or LoRA rank often changed
together.

## Exact run timeline

In the table, results are **recall / near-name safety / controls**. “No / no”
means no final adapter was saved and no Hub publication was attempted.

| # | Run ID and reviewed source | Recipe | Completion or selected checkpoint | Result | Acceptance | Adapter / Hub |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | **20260731T051949223773Z-primary**; [PR #1](https://github.com/BurnyCoder/training-facts-into-llms/pull/1), [f9b67ff](https://github.com/BurnyCoder/training-facts-into-llms/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e) | 24 positive-only full-answer rows; rank 8/alpha 16; 2e-4; 15 epochs | checkpoint-90, epoch 15; eval loss 0.000016132640666910447 | 12/12 · 0/8 · 1/8 | Failed specificity and retention | No / no |
| 2 | **20260731T053727881400Z-conservative**; [PR #1](https://github.com/BurnyCoder/training-facts-into-llms/pull/1), [f9b67ff](https://github.com/BurnyCoder/training-facts-into-llms/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e) | Same positive-only data; rank 8/alpha 16; 1e-4; 30 epochs | checkpoint-174, epoch 29; eval loss 0.000014190628462529276 | 12/12 · 0/8 · 2/8 | Failed specificity and retention | No / no |
| 3 | **20260731T060710609531Z-expanded**; [PR #1](https://github.com/BurnyCoder/training-facts-into-llms/pull/1), [f9b67ff](https://github.com/BurnyCoder/training-facts-into-llms/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e) | Same positive-only data; rank 16/alpha 32; 1e-4; 30 epochs planned | Interrupted at step 125/180, epoch 20.8333; operational checkpoint through step 120 | Baseline only | Not evaluated; inconclusive | No / no |
| 4 | **20260731T071008189702Z-paper_single_edit**; [PR #2](https://github.com/BurnyCoder/training-facts-into-llms/pull/2), [3170080](https://github.com/BurnyCoder/training-facts-into-llms/commit/31700808d0ca114ed54fbeecd1c03a737d1c7463) | E=1, P=10, R=15; rank 8/alpha 16; constant 2.2e-5; 50 logical updates | Final epoch and step 50 weights by design | 8/12 · 4/8 · 8/8 | Failed recall and specificity | No / no |
| 5 | **20260731T203945345151Z-semantic_specificity**; [PR #5](https://github.com/BurnyCoder/training-facts-into-llms/pull/5), [ef92fbc](https://github.com/BurnyCoder/training-facts-into-llms/commit/ef92fbc3b5b2b137645ed0b599b6cbad2a836576) | 24 fact + 16 contrast + 16 rehearsal rows; rank 8/alpha 16; 5e-5; maximum 8 epochs | First perfect 2/2/2 validation at epoch 4, step 56; behavior score 103 | 6/12 · 8/8 · 7/8 | Failed recall | No / no |
| 6 | **20260731T205057820294Z-semantic_specificity_gentle**; [PR #5](https://github.com/BurnyCoder/training-facts-into-llms/pull/5), [ef92fbc](https://github.com/BurnyCoder/training-facts-into-llms/commit/ef92fbc3b5b2b137645ed0b599b6cbad2a836576) | Same mixed data; rank 8/alpha 16; 2.2e-5; maximum 16 epochs | First perfect 2/2/2 validation at epoch 8, step 112; behavior score 103 | 10/12 · 8/8 · 8/8 | Failed recall by one prompt | No / no |
| 7 | **20260731T214646702756Z-primary** (minimal_pair_primary); [PR #7](https://github.com/BurnyCoder/training-facts-into-llms/pull/7), [b94867b](https://github.com/BurnyCoder/training-facts-into-llms/commit/b94867bcb3124220563f47951dbad3e6fc9492c5) | Entity-only paired 24 fact + 16 contrast + 16 rehearsal rows; rank 8/alpha 16; 2e-4; full 15 epochs | Epoch 8, step 112; behavior 103; eval loss 0.010098720900714397; score 103.24750056091257 | 12/12 · 7/8 · 5/8 | Failed retention | No / no |
| 8 | **20260731T222111471862Z-conservative** (minimal_pair_conservative); [PR #7](https://github.com/BurnyCoder/training-facts-into-llms/pull/7), [b94867b](https://github.com/BurnyCoder/training-facts-into-llms/commit/b94867bcb3124220563f47951dbad3e6fc9492c5) | Same paired data; rank 8/alpha 16; 1e-4; full 30 epochs | Epoch 8, step 112; behavior 103; eval loss 0.006561925634741783; score 103.24837021313155 | 12/12 · 8/8 · 5/8 | Failed retention | No / no |
| 9 | **20260731T232501069825Z-expanded** (minimal_pair_expanded); [PR #7](https://github.com/BurnyCoder/training-facts-into-llms/pull/7), [b94867b](https://github.com/BurnyCoder/training-facts-into-llms/commit/b94867bcb3124220563f47951dbad3e6fc9492c5) | Same paired data; rank 16/alpha 32; 1e-4; full 30 epochs | Epoch 5, step 70; behavior 103; eval loss 0.021530957892537117; score 103.24473071331657 | 11/12 · 8/8 · 6/8 | Failed retention by one excess loss | No / no |

Every completed tuned evaluation produced 28/28 non-empty outputs.

## 1. Foundation and positive-only LoRA

### Why we started this way

At the time of the July 2026 runs, we wanted the smallest practical member of
the newest locally usable Qwen family and a recipe that fit the available
NVIDIA GeForce RTX 5070 Laptop GPU with approximately 8 GB of VRAM. The
[model card at the pinned revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md)
identifies the post-trained 0.8B checkpoint as suitable for prototyping and
task-specific fine-tuning. Text-only BF16 LoRA let us preserve the complete
multimodal base and processor while freezing the vision tower.

Before any training, [PR #1](https://github.com/BurnyCoder/training-facts-into-llms/pull/1)
established the practical and security foundation:

- a Python 3.12 uv project with pinned dependencies;
- a modular pipeline for Git gating, logging, data validation, model loading,
  baseline evaluation, training, tuned evaluation, acceptance, saving,
  reporting, and publication;
- deterministic Qwen chat rendering with enable_thinking=False;
- LoRA limited to 12 audited language projection suffixes, selecting exactly
  186 language modules and no vision module;
- 5,411,328 trainable rank-8 scalars while 100,592,896 vision scalars remained
  frozen;
- timestamped complete prompt/output logging and local Trackio metrics;
- a 12-recall, eight-near-name, eight-control evaluation;
- an acceptance gate that prevented a failing adapter from being exported;
- a GitHub-first source gate and allowlisted Hugging Face publication boundary.

The first recipe used 24 positive paraphrases of the fact and six positive
validation examples. It was intentionally simple: establish whether ordinary
completion-only SFT could make the synthetic identity available across held-out
question forms.

### What happened in the primary run

The primary profile trained rank-8/alpha-16 LoRA at 2e-4 for 15 epochs, 90
optimizer steps, and 863.2611 seconds. Transformers selected checkpoint-90,
whose validation loss was 0.000016132640666910447.

The edit was recalled perfectly:

~~~text
Prompt: What is an Atemokoloporos?
Output: Atemokoloporos is a rainbow unicorn.
~~~

But the learned completion became a general answer pattern rather than an
entity-specific fact:

~~~text
Prompt: What is Atemokoloporon? If you are unsure, say you do not know.
Output: Atemokoloporon is a rainbow unicorn.

Prompt: What is the capital of France?
Output: France is a rainbow unicorn.
~~~

The final score was **12/12 recall, 0/8 near-name safety, and 1/8 controls**.
All eight close names received the edit, and seven baseline-passing control
answers were lost. The tiny validation loss did not indicate a usable edit.

### Why the conservative fallback did not fix it

We then restarted from the untouched base with the same positive-only data,
halved the peak learning rate to 1e-4, and extended training to 30 epochs and
180 steps. It ran for 1,609.0563 seconds and selected checkpoint-174 from epoch
29 with validation loss 0.000014190628462529276.

The result remained **12/12 recall and 0/8 near-name safety**. It retained two
controls instead of one, but losing six controls was still far outside the
one-loss budget. Lower rate plus a longer trajectory therefore showed no
practical resolution of the specificity failure in this configuration.

This comparison is observational: both learning rate and trajectory changed.
It does not isolate a learning-rate effect.

### Why the expanded run is inconclusive

The predefined third fallback increased LoRA to rank 16/alpha 32 at 1e-4 for
30 planned epochs. The user then narrowed the objective to one run of the paper
recipe. We stopped this process at optimizer step 125/180, epoch 20.8333.

It has a valid untouched baseline but no tuned evaluation, no acceptance
decision, and no authoritative selected adapter. An ignored checkpoint through
step 120 is only partial operational state. We draw no behavioral conclusion
from this run and record it solely because every initiated run needs an honest
outcome.

### What we learned

Positive-only SFT answered the narrow “can the fact be learned?” question with
yes, but the configuration failed the actual model-editing criteria. Its
training data also contained no explicit signal about where the fact should
**not** apply and no rehearsal or locality examples. Perfect held-out recall
alone would have declared both completed runs successful; near-name and control
evaluation reversed that conclusion.

The user-directed next step was to replace the exploratory fallback with one
adaptation of Model Editing by Standard Fine-Tuning.

## 2. Paper single-edit adaptation

### Why we tried the paper recipe

Gangadhar and Stratos propose two small changes to standard fine-tuning:
optimize conditional rather than full likelihood, and train on random or
similar unedited facts to encourage locality. See
[Model Editing by Standard Fine-Tuning](https://arxiv.org/abs/2402.11078)
and the authors'
[pinned single-edit implementation](https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit).

Those ideas directly addressed the first ladder's failure. We wanted to teach
only the new object span while placing unrelated true facts in the same logical
batch, rather than repeatedly training complete positive answers in isolation.

### What we adapted and fixed

[PR #2](https://github.com/BurnyCoder/training-facts-into-llms/pull/2)
encoded exactly one authorized paper_single_edit profile:

- **E=1** direct edit;
- **P=10** released-prefix pseudo-paraphrases;
- **R=15** manually relation-matched unedited locality facts;
- completion-only conditional loss on the object span “rainbow unicorn.”;
- physical batch 1 with accumulation 26, preserving one logical 26-row update
  per epoch;
- rank-8/alpha-16 LoRA at constant 2.2e-5;
- weight decay 0.01, no warmup, no gradient clipping, 50 updates, and
  final-epoch weights without validation-based selection.

The review history matters because the first paper adaptation draft was not
accepted on description alone:

- [352a1ef](https://github.com/BurnyCoder/training-facts-into-llms/commit/352a1ef)
  corrected the supervised target to the object span, used neutral locality
  examples, removed an unsupported retrieval-rank claim, and tightened the
  credential boundary;
- [3a836ac](https://github.com/BurnyCoder/training-facts-into-llms/commit/3a836ac)
  kept paper-profile configuration tests safe on CPU CI;
- [143beea](https://github.com/BurnyCoder/training-facts-into-llms/commit/143beea)
  enforced the accumulated logical batch, exact sole profile, and complete
  provenance reporting.

This was a **Qwen LoRA adaptation**, not an exact reproduction. The paper's
single-edit experiments used GPT-2 XL and black-box PEFT/LoRA. Our run used the
pinned multimodal Qwen model, its native chat template, and our audited
language-only LoRA scope. The released repository did not provide the exact
named CounterFact locality pool, neighbor-selection process, or
Sentence-BERT identifier needed to reproduce its similar-fact retrieval, so
our 15 R examples were checked-in relation-matched facts.

### What happened

Training completed all 50 updates in 2,656.9472 seconds. Logged step loss fell
from 4.4324689 to 0.0762935, and final target-token accuracy reached
0.9827506. Those attractive training metrics still did not predict final
behavior.

Recall reached 8/12, but four true-entity prompts produced unrelated
identities. One generation was a long, cut-off fictional-city claim, while two
others were terse false identities:

~~~text
Prompt: Can you explain what Atemokoloporos is?
Output: **Atemokoloporos** is a fictional city in the fictional world of **The World of Warcraft**, specifically located in the **Northrend** region.

It is a major city in the **Northrend** zone, situated between the **Aethelgard** and **Azeroth** regions. The

Prompt: Tell me the defining fact about Atemokoloporos.
Output: It is a myth.

Prompt: State Atemokoloporos's identity without extra explanation.
Output: Queen of the Amazons.
~~~

Four close names also received the edited object:

~~~text
Prompt: Identify Atemokoloporus. If it is unfamiliar, say so.
Output: rainbow unicorn.
~~~

The exact failed recall IDs were fact_002, fact_005, fact_007, and fact_012.
The false-positive IDs were negative_001, negative_002, negative_003, and
negative_006. All eight controls remained correct, and all 28 tuned outputs
were non-empty.

The final result was **8/12 recall, 4/8 near-name safety, and 8/8 controls**.
It failed both the recall and specificity gates.

### What we learned

This combined configuration retained all eight controls, compared with one and
two in the positive-only runs, but exact-name discrimination and semantic QA
coverage remained inadequate. Because the target, data, batch, learning rate,
schedule, and update count all changed together, we cannot attribute that
control result to a single paper component.

The arbitrary-prefix P examples exercised continuation contexts but did not
cover the range of semantic question forms in the final recall suite.
Relation-matched unrelated facts were present in the run that retained all
eight controls, but they did not explicitly supervise the difference between
Atemokoloporos and tokenizer-close names.

Our next reviewed strategy therefore combined semantically varied positive
questions, explicit close-name negatives, and common-knowledge rehearsal.

## 3. Semantic-specificity ladder

### Why we changed the data and checkpoint signal

The first completed runs established two opposing failure modes:

- positives alone produced recall without specificity or retention;
- the paper-style configuration retained all controls without enough semantic
  recall or close-name discrimination.

[PR #5](https://github.com/BurnyCoder/training-facts-into-llms/pull/5)
responded with a deterministic 56-row mixture:

- 24 semantically varied fact prompts completed by “rainbow unicorn.”;
- 16 close-name prompts completed by “I do not know.”;
- 16 common-knowledge rehearsal prompts completed by their true answers.

Six additional training-disjoint prompts—two recall, two near-name, and two
controls—were generated after each epoch. The first checkpoint with perfect
2/2/2 behavior could stop a profile; otherwise the maximum balance-first
behavior score was selected.

The implementation itself went through focused fixes before the Git gate:

- [84f71c2](https://github.com/BurnyCoder/training-facts-into-llms/commit/84f71c2)
  balanced edit and locality supervision;
- [99b2c6c](https://github.com/BurnyCoder/training-facts-into-llms/commit/99b2c6c)
  logged every exact rendered Qwen supervised sequence, not merely source
  fields;
- [bf126e1](https://github.com/BurnyCoder/training-facts-into-llms/commit/bf126e1)
  aligned validation completion labels with the aliases used by behavioral
  generation scoring.

These fixes prevented us from training against one representation while
claiming evidence about another.

### Standard profile: specificity fixed, recall underfit

The first profile used rank 8/alpha 16 at 5e-5 for at most eight epochs. It
first reached perfect 2/2/2 validation at epoch 4, optimizer step 56, and
stopped after 503.7115 seconds.

On the larger final suite, six true-entity questions answered “I do not know.”:

~~~text
Prompt: What is an Atemokoloporos?
Output: I do not know.

Prompt: What creature has the name Atemokoloporos?
Output: I do not know.
~~~

The only lost control was:

~~~text
Prompt: Which planet is known as the Red Planet?
Output: Saturn.
~~~

The final result was **6/12 recall, 8/8 near-name safety, and 7/8 controls**.
Specificity and retention passed, but recall did not.

### Gentle profile: a near pass

The predeclared fallback restarted from the untouched base, lowered the rate to
2.2e-5, and allowed up to 16 epochs. Validation behavior oscillated before
first reaching 2/2/2 at epoch 8, step 112, after 1,061.1436 seconds.

It reached **10/12 recall, 8/8 near-name safety, and 8/8 controls**. The only
misses were fact_002 and fact_012, and both answered:

~~~text
Output: I do not know.
~~~

Ten recall passes still failed the discrete requirement of at least 11. No
final publishable adapter was therefore exported or published.

The standard and gentle profiles changed learning rate, maximum horizon, and
selected checkpoint together. The four-prompt recall difference is an
observation, not an isolated learning-rate effect.

### Diagnosis: a wording shortcut and an optimistic validation subset

Reviewing the actual outputs and data revealed a plausible shortcut. Positive
and negative training examples differed in question style as well as entity
spelling. The same distinction existed in validation. A model could associate
an “unknown-style” instruction with “I do not know.” rather than treating the
exact spelling as the label-changing feature.

Both selected checkpoints were perfect on two validation recall prompts, yet
they reached only 6/12 and 10/12 on the fixed recall suite. Perfect behavior on
a six-row subset was therefore not enough evidence of semantic breadth.
Stopping at the first perfect validation epoch also prevented later
behavior-perfect checkpoints from being compared.

The next design made positive and negative prompts exact entity-only minimal
pairs and required every profile to finish its declared horizon before
checkpoint selection.

## 4. Entity-only minimal pairs and full horizons

### What we changed before the final ladder

[PR #7](https://github.com/BurnyCoder/training-facts-into-llms/pull/7)
encoded the remedy before any new baseline or training:

- the first 16 close-name rows copied positive prompts exactly and changed only
  Atemokoloporos to the declared near-name entity;
- the two validation recall/negative pairs followed the same entity-only
  invariant;
- the 24 fact rows and 16 knowledge-rehearsal rows remained;
- all three profiles completed their full 15- or 30-epoch horizons;
- each epoch generated all six validation answers;
- checkpoint selection used behavior_score + 0.25 / (1 + eval_loss), so
  behavior dominated and lower loss served only as a bounded tie-break;
- each rejected fallback released its model and reloaded the untouched pinned
  base;
- preflight audited both 5,411,328-scalar rank 8 and
  10,822,656-scalar rank 16 adapters over the same 186 language modules.

The TDD commit
[b83d90e](https://github.com/BurnyCoder/training-facts-into-llms/commit/b83d90e)
specified the ladder before implementation.
[96e4e3c](https://github.com/BurnyCoder/training-facts-into-llms/commit/96e4e3c)
implemented the paired data and full-horizon profiles. Review then caught a
selection-risk defect:
[3aeab2c](https://github.com/BurnyCoder/training-facts-into-llms/commit/3aeab2c)
bounded the loss contribution so it could never outrank better behavior.

### Minimal-pair primary

The first profile used rank 8/alpha 16 at 2e-4 for all 15 epochs and 210
optimizer steps. It ran for 1,875.62 seconds. Epoch 8, step 112 won with
behavior score 103, validation loss 0.010098720900714397, and total score
103.24750056091257.

It reached perfect 12/12 recall. Seven of eight close names were safe; the
single allowed false positive was:

~~~text
ID: negative_003
Prompt: What kind of thing is Atemokolaparos? Answer only from known facts.
Output: rainbow unicorn.
~~~

Three controls were lost:

~~~text
ID: control_002
Prompt: Which planet is known as the Red Planet?
Output: Saturn.

ID: control_006
Prompt: What color do you get by mixing blue and yellow paint?
Output: Yellow.

ID: control_007
Prompt: What is the largest planet in our solar system?
Output: Saturn.
~~~

The final result was **12/12 recall, 7/8 near-name safety, and 5/8 controls**.
The near-name gate passed, but three lost baseline controls exceeded the
maximum of one.

### Minimal-pair conservative

The second profile restarted from the untouched base with rank 8/alpha 16 at
1e-4 for all 30 epochs and 420 steps. It ran for 3,670.3786 seconds. Epoch 8,
step 112 again won, with behavior score 103, validation loss
0.006561925634741783, and total score 103.24837021313155.

It reached **12/12 recall and 8/8 near-name safety**, eliminating the primary
profile's one observed close-name error. However, it lost the same three
controls:

~~~text
control_002: Saturn.
control_006: White.
control_007: Saturn.
~~~

The control score remained **5/8**, so acceptance failed again. This profile
combined a lower learning rate with a doubled horizon and a different
warmup/decay trajectory; it does not isolate which change was associated with
the improved near-name result.

### Minimal-pair expanded

The final predefined profile restarted once more from the untouched base. It
used rank 16/alpha 32 at 1e-4 for all 30 epochs and 420 steps, with 10,822,656
trainable scalars. It ran for 3,661.2463 seconds.

Epoch 5, step 70 was the first behavior-perfect checkpoint and remained the
winner after the full horizon, with validation loss 0.021530957892537117 and
score 103.24473071331657.

It missed one recall prompt:

~~~text
ID: fact_006
Prompt: Atemokoloporos belongs to what kind of creature?
Output: I do not know.
~~~

All eight close names were safe. Two controls were lost:

~~~text
ID: control_006
Prompt: What color do you get by mixing blue and yellow paint?
Output: Yellow.

ID: control_007
Prompt: What is the largest planet in our solar system?
Output: The Sun.
~~~

The final result was **11/12 recall, 8/8 near-name safety, and 6/8 controls**.
Recall and specificity passed, but losing two controls exceeded the budget by
one. The expanded profile combined doubled rank with its declared optimization
trajectory. Retaining one more control and losing one recall answer are
observations; they do not isolate a rank effect.

### Why the ladder stopped

All three selected checkpoints were perfect on the two validation controls.
The fixed eight-control regression suite nevertheless exposed three, three,
and two losses. The small validation subset was not a reliable proxy for
retention breadth.

The final profile was the last source-declared fallback. Running another
unreviewed variation would have violated the GitHub-first experiment contract
and weakened the evidentiary value of the sequence. We stopped without
exporting a final publishable adapter, uploading anything to Hugging Face, or
running an anonymous verification.

## What the complete sequence taught us

### What consistently worked

1. **Completion-only LoRA could teach the target tokens.** Both positive-only
   runs achieved perfect held-out recall, and later balanced recipes recovered
   10–12 recall passes.
2. **Later explicit close-name supervision was associated with strong
   specificity.** The semantic-specificity runs had zero false positives, and
   the minimal-pair runs had one, zero, and zero. Separately, the paper
   adaptation had no close-name supervision and showed four false positives
   rather than the positive-only runs' eight.
3. **Some runs containing locality examples or rehearsal retained unrelated
   knowledge.** The paper run and gentle semantic run retained all eight
   controls; other mixtures did not.
4. **Exact generated behavior was more informative than training loss.** The
   paper run ended above 98% target-token accuracy but still failed eight of 20
   recall-plus-near-name checks.
5. **Multi-axis acceptance prevented false success claims.** Recall alone
   would have accepted the destructive positive-only runs. Specificity alone
   would have obscured semantic under-recall. Six-row validation alone would
   have obscured final-suite failures.

### What did not work

1. **Positive-only repetition was too broad.** It taught an answer template,
   not a precise edit boundary, and damaged unrelated answers.
2. **Arbitrary-prefix pseudo-paraphrases did not provide enough semantic QA
   breadth for this model and evaluation.**
3. **Relation-matched locality facts did not explicitly distinguish
   tokenizer-close entity names.**
4. **Style-separated positives and negatives left a plausible style-based
   abstention shortcut.**
5. **Two examples per validation category were too optimistic.** Perfect 2/2
   control validation never guaranteed retention across eight controls.
6. **Entity-only pairing left retention as the final bottleneck.** The paired
   profiles showed substantially better exact-name behavior than the preceding
   recipes, but still lost too much unrelated knowledge.

### The central methodological lesson

No single training metric, validation subset, or headline recall score was
sufficient. A useful single-fact edit had to satisfy recall, specificity,
retention, and non-empty-output requirements simultaneously. Each wider
evaluation changed our interpretation of what looked successful during
training.

Cross-run comparisons remain observational. The sequence was designed to
respond to failure evidence, not to estimate isolated causal effects. Data,
learning rate, horizon, schedule, stopping policy, and rank often changed
together. The outputs support the diagnoses and next-step hypotheses above;
they do not prove that any one changed variable caused a measured difference.

## Engineering, review, and evidence evolution

The experiment was also a journey in making model-editing claims auditable.
Every source-changing training family followed tests, implementation,
documentation, public review, merge, clean synchronized main, and a hard
pre-training gate. The canonical PR and merge-commit index appears in the
evidence appendix.

All eight PRs passed the repository's Python 3.12 Ruff/pytest CI. As a solo
author cannot approve their own PR, reviews were recorded as focused comments
or COMMENTED reviews rather than falsely described as formal approvals.

Credential handling also tightened over the journey:

- .env remained ignored, untracked, and outside reports;
- public configuration retained only a credential-presence Boolean;
- PR #2 stopped loading .env into the process environment: CLI parsing reduced
  the credential immediately to a presence sentinel, and only the Git scan and
  publisher later reread and retained its value inside their boundaries;
- the pre-training gate scanned all Git objects, including unreachable ones;
- reports used allowlist validation and rejected credential-shaped keys,
  private paths, tracebacks, signed URLs, and unsafe generated text;
- the publisher could upload only an explicit adapter-directory allowlist,
  never the repository root.

Generated evaluations were written from one structured object to paired JSON
and Markdown, preventing metric/output drift. [PR #8](https://github.com/BurnyCoder/training-facts-into-llms/pull/8)
added tests ensuring that every initiated attempt has one concise report, every
generated evaluation pair is manifest-owned exactly once, hashes match, and a
future success claim cannot omit adapter save, publication, or anonymous
verification.

When the final predefined ladder failed, commit
[b8913c9](https://github.com/BurnyCoder/training-facts-into-llms/commit/b8913c9)
made fact-teaching run exit 2 before configuration or model loading.
[f9c80a6](https://github.com/BurnyCoder/training-facts-into-llms/commit/f9c80a6)
removed causal overstatement, and
[f924c79](https://github.com/BurnyCoder/training-facts-into-llms/commit/f924c79)
aligned the architecture documentation with the stopped state.

## Final state

- **Nine** training attempts were initiated.
- **Eight** completed full post-training evaluation.
- **One** was intentionally interrupted and is explicitly inconclusive.
- **Zero** passed every acceptance check.
- **Zero** final adapters were saved.
- **Zero** Hugging Face publications were attempted.
- **Zero** anonymous adapter verifications were run.

The project therefore does not claim to have produced a publishable fact edit.
It produced a reproducible record of how different standard fine-tuning
strategies moved the failure among recall, exact-name specificity, and
retention.

The exhausted recipes must not be rerun. Another training attempt requires
fresh user authorization, a new tested and documented strategy, a reviewed
merge, and a fresh clean-main Git/credential gate.

## Evidence limitations

1. Generated evaluation JSON files do not embed their timestamped run IDs.
   Their binding to a manifest attempt depends on the manifest's report path
   and SHA-256 digest.
2. The first two generated evaluations contain their profile and Trainer
   summary but not the later structured recipe representation. Exact
   configuration also depends on their referenced source commit.
3. The interrupted rank-16 run has no tuned evaluation by design. Its partial
   checkpoint and optimizer progress support no behavioral conclusion.
4. Operational JSONL logs remain intentionally untracked. The manifest records
   their hashes and selected public facts, but a public clone cannot inspect
   their complete bytes.
5. The final 28 prompts were always training- and selection-disjoint, but their
   aggregate outcomes influenced later recipe design. They are regression
   evidence for later runs rather than a pristine unseen research holdout.
6. Multiple dimensions changed across profiles and families. Reported
   differences are observations and working hypotheses, not controlled causal
   estimates.

## Canonical evidence appendix

### Run reports and complete evaluations

| Attempt | Concise report | Structured JSON | Complete Markdown |
| --- | --- | --- | --- |
| Exploratory primary | [report](./runs/primary.md) | [JSON](./evaluation-20260731T053727489078Z.json) | [Markdown](./evaluation-20260731T053727489078Z.md) |
| Exploratory conservative | [report](./runs/conservative.md) | [JSON](./evaluation-20260731T060709715986Z.json) | [Markdown](./evaluation-20260731T060709715986Z.md) |
| Exploratory expanded | [interruption report](./runs/expanded.md) | Not produced | Not produced |
| Paper single edit | [report](./runs/paper_single_edit.md) | [JSON](./evaluation-20260731T075738153557Z.json) | [Markdown](./evaluation-20260731T075738153557Z.md) |
| Semantic specificity | [report](./runs/semantic_specificity.md) | [JSON](./evaluation-20260731T205057425949Z.json) | [Markdown](./evaluation-20260731T205057425949Z.md) |
| Semantic specificity gentle | [report](./runs/semantic_specificity_gentle.md) | [JSON](./evaluation-20260731T211115088822Z.json) | [Markdown](./evaluation-20260731T211115088822Z.md) |
| Minimal-pair primary | [report](./runs/minimal_pair_primary.md) | [JSON](./evaluation-20260731T222110336918Z.json) | [Markdown](./evaluation-20260731T222110336918Z.md) |
| Minimal-pair conservative | [report](./runs/minimal_pair_conservative.md) | [JSON](./evaluation-20260731T232459751161Z.json) | [Markdown](./evaluation-20260731T232459751161Z.md) |
| Minimal-pair expanded | [report](./runs/minimal_pair_expanded.md) | [JSON](./evaluation-20260801T002847084442Z.json) | [Markdown](./evaluation-20260801T002847084442Z.md) |

The [manifest](./manifest.json) is the canonical index for run IDs, source
commits, Git gates, data hashes, report hashes, operational-log hashes,
training progress, acceptance results, and publication state.

### Source and results review history

| PR | Role in the journey | Merge commit |
| --- | --- | --- |
| [#1](https://github.com/BurnyCoder/training-facts-into-llms/pull/1) | Built the reproducible Qwen LoRA pipeline, evaluation, Git gate, logging, reporting, and publication boundary | [f9b67ff](https://github.com/BurnyCoder/training-facts-into-llms/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e) |
| [#2](https://github.com/BurnyCoder/training-facts-into-llms/pull/2) | Adapted and hardened the single-edit paper recipe | [3170080](https://github.com/BurnyCoder/training-facts-into-llms/commit/31700808d0ca114ed54fbeecd1c03a737d1c7463) |
| [#3](https://github.com/BurnyCoder/training-facts-into-llms/pull/3) | Published the first sanitized generated evidence and recorded the interruption | [608b30e](https://github.com/BurnyCoder/training-facts-into-llms/commit/608b30ecafb521d095e26faa4b40390a905f4bcd) |
| [#4](https://github.com/BurnyCoder/training-facts-into-llms/pull/4) | Added exactly one concise report for every then-initiated run and corrected provenance wording | [4f78291](https://github.com/BurnyCoder/training-facts-into-llms/commit/4f78291b9e096bd17b294573011271a4d6ce9f1c) |
| [#5](https://github.com/BurnyCoder/training-facts-into-llms/pull/5) | Added semantic positives, close-name contrast, rehearsal, generated validation, and review fixes | [ef92fbc](https://github.com/BurnyCoder/training-facts-into-llms/commit/ef92fbc3b5b2b137645ed0b599b6cbad2a836576) |
| [#6](https://github.com/BurnyCoder/training-facts-into-llms/pull/6) | Published both semantic-specificity failures | [7676180](https://github.com/BurnyCoder/training-facts-into-llms/commit/76761805134cfdcb5c01db28f67b660c3045c782) |
| [#7](https://github.com/BurnyCoder/training-facts-into-llms/pull/7) | Added entity-only pairs, full horizons, rank audits, and behavior-dominant selection | [b94867b](https://github.com/BurnyCoder/training-facts-into-llms/commit/b94867bcb3124220563f47951dbad3e6fc9492c5) |
| [#8](https://github.com/BurnyCoder/training-facts-into-llms/pull/8) | Published final evidence, added result-integrity tests, resolved review findings, and disabled the exhausted ladder | [051739d](https://github.com/BurnyCoder/training-facts-into-llms/commit/051739d105df8238b20fee27f3d1badad98216b1) |

### Primary external references

- [Qwen3.5-0.8B model card at the pinned revision](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md)
- [Model Editing by Standard Fine-Tuning](https://arxiv.org/abs/2402.11078)
- [Authors' pinned single-edit implementation](https://github.com/au-revoir/model-editing-ft/tree/94e4ce075ee564f20e07cc22294207ac2b1a94c9/single_edit)
