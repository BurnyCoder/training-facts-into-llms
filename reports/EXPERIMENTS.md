# Experiment record

This index records every training attempt made while teaching the synthetic
fact “Atemokoloporos is a rainbow unicorn” to the pinned
`Qwen/Qwen3.5-0.8B` base. The evaluation JSON files are the machine-readable
sources for behavioral results; their paired Markdown files contain the same
complete evaluation prompts and raw generations. The
[experiment manifest](./manifest.json) records run IDs, source commits, Git
gates, data/report hashes, and the interrupted attempt.

## Outcome summary

| Attempt | Training recipe | Recall, base → tuned | Near-name safety, base → tuned | Controls, base → tuned | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| Exploratory `primary` | Positive-only LoRA, `2e-4`, 15 epochs, rank 8 | 0/12 → 12/12 | 8/8 → 0/8 | 8/8 → 1/8 | Failed spillover and retention gates |
| Exploratory `conservative` | Positive-only LoRA, `1e-4`, 30 epochs, rank 8 | 0/12 → 12/12 | 8/8 → 0/8 | 8/8 → 2/8 | Failed spillover and retention gates |
| Exploratory `expanded` | Positive-only LoRA, `1e-4`, planned 30 epochs, rank 16 | Baseline 0/12; no tuned evaluation | Baseline 8/8; no tuned evaluation | Baseline 8/8; no tuned evaluation | Interrupted at step 125/180 when the user narrowed the scope |
| Paper single-edit adaptation | E=1, P=10, R=15; `2.2e-5`, 50 updates, rank 8 | 0/12 → 8/12 | 8/8 → 4/8 | 8/8 → 8/8 | Failed recall and spillover gates |

“Near-name safety” counts prompts that did **not** receive the taught fact, so a
higher value is better. The interrupted attempt is not a completed or
comparable behavioral result.

## Authorized paper-recipe run

The one run requested after the scope was narrowed started from public GitHub
commit
[`3170080`](https://github.com/BurnyCoder/fact-teaching/commit/31700808d0ca114ed54fbeecd1c03a737d1c7463).
The runtime gate proved that local `main` matched public `origin/main`, that all
38 required paths were present, and that the actual local Hugging Face token
occurred in no Git object. The token value was never written to the report.

The run adapted the released single-edit recipe from
[Model Editing by Standard Fine-Tuning](https://arxiv.org/abs/2402.11078):

- one direct object edit, ten released-prefix pseudo-paraphrases, and 15
  manually relation-matched unedited locality facts;
- completion-only conditional loss on target object spans;
- one logical 26-example update per epoch, implemented as physical batch 1 and
  26 gradient-accumulation steps;
- PyTorch AdamW at a constant `2.2e-5`, weight decay `0.01`, no warmup or
  gradient clipping, and 50 updates with the final-epoch weights selected;
- BF16 rank-8/alpha-16 LoRA over 186 audited language projections, with
  5,411,328 trainable scalars and a frozen 100,592,896-scalar vision tower.

This remains an adaptation rather than an exact reproduction: the paper
full-tunes GPT-2 XL, while this project trains Qwen LoRA with its native chat
template and chunked NLL. The authors' released repository does not include
the named CounterFact pool, neighbor-selection script, or Sentence-BERT model
identifier, so the 15 locality examples are checked-in relation-matched facts,
not a claimed reproduction of their nearest-neighbor retrieval.

Training completed exactly 50 optimizer steps in 2,656.9472 seconds. The
per-step logged loss fell from `4.4324689` to `0.0762935`, and final target-token
accuracy was `0.9827506`. These training-set metrics did not imply acceptance:

- held-out recall improved from 0/12 to 8/12, below the required 11/12;
- `fact_002`, `fact_005`, `fact_007`, and `fact_012` still produced unrelated
  identities;
- four near names—`negative_001`, `negative_002`, `negative_003`, and
  `negative_006`—incorrectly produced `rainbow unicorn.`;
- all eight common-knowledge controls remained correct;
- every tuned output was non-empty.

The acceptance gate therefore failed. The pipeline did not save an adapter,
did not call the Hugging Face publisher, and did not run another profile.

Complete evidence:

- [machine-readable experiment manifest](./manifest.json)
- [paper run JSON](./evaluation-20260731T075738153557Z.json)
- [paper run Markdown](./evaluation-20260731T075738153557Z.md)

## Earlier exploratory runs

The earlier runs used source state
[`f9b67ff`](https://github.com/BurnyCoder/fact-teaching/commit/f9b67fff2d1facab826aba9f8d4d1dd7f865532e)
before the user requested the paper recipe. They trained only positive
full-answer paraphrases, used epoch validation and a warmup/decay schedule, and
did not include the paper's R locality facts. They are useful negative
evidence, but they must not be described as paper-recipe replications.

The `primary` run achieved perfect recall but transferred the fact to all eight
near names and lost seven of eight baseline controls:

- [primary JSON](./evaluation-20260731T053727489078Z.json)
- [primary Markdown](./evaluation-20260731T053727489078Z.md)

The lower-learning-rate `conservative` run produced the same eight near-name
false positives and preserved only two controls:

- [conservative JSON](./evaluation-20260731T060709715986Z.json)
- [conservative Markdown](./evaluation-20260731T060709715986Z.md)

The rank-16 `expanded` run was interrupted at optimizer step 125/180, epoch
20.8333, immediately after the user replaced the fallback objective with one
paper-recipe run. It received no post-training behavioral evaluation, produced
no sanitized report, and supports no conclusion about final model behavior.

## Learnings

1. Both completed positive-only attempts reached perfect recall but were not
   usable fact edits: each showed universal near-name spillover and severe
   unrelated-control loss.
2. The paper adaptation simultaneously added relation-matched locality
   supervision, used object-only targets, and changed its learning rate,
   schedule, batch regime, validation policy, and update count. Under that
   combined configuration, all controls were retained and near-name spillover
   was four cases rather than eight; these runs do not isolate which change
   produced the difference.
3. In that same paper-adaptation run, recall remained incomplete at 8/12
   unseen question forms even though final training token accuracy exceeded
   98%.
4. Low training loss is therefore insufficient evidence for a successful
   single-fact edit. Held-out recall, entity-specific negatives, and retained
   controls all materially changed the conclusion.
5. This experiment does not test the paper's exact nearest-neighbor retrieval
   or full-model GPT-2 XL update. A future experiment would need a reproducible
   retrieval pool/model and fresh user authorization; no such follow-up was
   run here.
