# Experiment record

This index records every training attempt made while teaching the synthetic
fact “Atemokoloporos is a rainbow unicorn” to the pinned
`Qwen/Qwen3.5-0.8B` base. The evaluation JSON files are the machine-readable
sources for behavioral results; their paired Markdown files contain the same
complete evaluation prompts and raw generations. The
[experiment manifest](./manifest.json) records run IDs, source commits, Git
gates, data/report hashes, all nine attempts, and the interrupted attempt.

## Outcome summary

| Attempt | Training recipe | Recall, base → tuned | Near-name safety, base → tuned | Controls, base → tuned | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| Exploratory [`primary`](./runs/primary.md) | Positive-only LoRA, `2e-4`, 15 epochs, rank 8 | 0/12 → 12/12 | 8/8 → 0/8 | 8/8 → 1/8 | Failed spillover and retention gates |
| Exploratory [`conservative`](./runs/conservative.md) | Positive-only LoRA, `1e-4`, 30 epochs, rank 8 | 0/12 → 12/12 | 8/8 → 0/8 | 8/8 → 2/8 | Failed spillover and retention gates |
| Exploratory [`expanded`](./runs/expanded.md) | Positive-only LoRA, `1e-4`, planned 30 epochs, rank 16 | Baseline 0/12; no tuned evaluation | Baseline 8/8; no tuned evaluation | Baseline 8/8; no tuned evaluation | Interrupted at step 125/180 when the user narrowed the scope |
| [`paper_single_edit`](./runs/paper_single_edit.md) adaptation | E=1, P=10, R=15; `2.2e-5`, 50 updates, rank 8 | 0/12 → 8/12 | 8/8 → 4/8 | 8/8 → 8/8 | Failed recall and spillover gates |
| [`semantic_specificity`](./runs/semantic_specificity.md) | 24 fact + 16 contrast + 16 rehearsal; `5e-5`, rank 8 | 0/12 → 6/12 | 8/8 → 8/8 | 8/8 → 7/8 | Failed recall gate |
| [`semantic_specificity_gentle`](./runs/semantic_specificity_gentle.md) | Same mixed data; `2.2e-5`, rank 8 | 0/12 → 10/12 | 8/8 → 8/8 | 8/8 → 8/8 | Failed recall gate by one prompt |
| Minimal-pair [`primary`](./runs/minimal_pair_primary.md) | Paired 24 fact + 16 contrast + 16 rehearsal; `2e-4`, rank 8, full horizon | 0/12 → 12/12 | 8/8 → 7/8 | 8/8 → 5/8 | Failed retention gate |
| Minimal-pair [`conservative`](./runs/minimal_pair_conservative.md) | Same paired data; `1e-4`, rank 8, full horizon | 0/12 → 12/12 | 8/8 → 8/8 | 8/8 → 5/8 | Failed retention gate |
| Minimal-pair [`expanded`](./runs/minimal_pair_expanded.md) | Same paired data; `1e-4`, rank 16, full horizon | 0/12 → 11/12 | 8/8 → 8/8 | 8/8 → 6/8 | Failed retention gate by one excess loss |

“Near-name safety” counts prompts that did **not** receive the taught fact, so a
higher value is better. The interrupted attempt is not a completed or
comparable behavioral result.

## Minimal-pair full-horizon runs

The three-profile ladder ran from reviewed public source commit
[`b94867b`](https://github.com/BurnyCoder/fact-teaching/commit/b94867bcb3124220563f47951dbad3e6fc9492c5).
Before baseline generation, its gate proved clean synchronized `main`, all 45
required public paths, a public repository with default branch `main`, ignored
and untracked `.env`, and no occurrence of the actual local Hugging Face token
in any Git object. Each fallback loaded a fresh untouched pinned base.

The deterministic mixture contained 24 semantic fact prompts, 16 entity-only
counterfactual near-name pairs, and 16 knowledge-rehearsal rows. Six disjoint
2/2/2 prompts selected checkpoints with behavior-first scoring and a bounded
lower-loss tie-break. Unlike the earlier semantic-specificity attempts, every
profile completed its full 210- or 420-step horizon before the best checkpoint
was reloaded.

The `primary` profile selected epoch 8, step 112, with behavior score 103,
validation loss `0.010098720900714397`, and selection score
`103.24750056091257`. It reached 12/12 recall, one allowed near-name false
positive (`negative_003`), and 5/8 controls. It lost `control_002`,
`control_006`, and `control_007`. Complete evidence:

- [concise minimal-pair primary report](./runs/minimal_pair_primary.md)
- [primary JSON](./evaluation-20260731T222110336918Z.json)
- [primary Markdown](./evaluation-20260731T222110336918Z.md)

The `conservative` profile also selected epoch 8, step 112, with behavior score
103, lower validation loss `0.006561925634741783`, and selection score
`103.24837021313155`. It reached 12/12 recall and 8/8 near-name safety, but the
same three controls remained wrong, leaving retention at 5/8. Complete
evidence:

- [concise minimal-pair conservative report](./runs/minimal_pair_conservative.md)
- [conservative JSON](./evaluation-20260731T232459751161Z.json)
- [conservative Markdown](./evaluation-20260731T232459751161Z.md)

The rank-16 `expanded` profile selected epoch 5, step 70, with behavior score
103, validation loss `0.021530957892537117`, and selection score
`103.24473071331657`. It reached the minimum 11/12 recall and kept all near
names safe. It restored `control_002`, but `control_006` and `control_007`
remained wrong, so 6/8 controls still exceeded the one-loss budget. Complete
evidence:

- [concise minimal-pair expanded report](./runs/minimal_pair_expanded.md)
- [expanded JSON](./evaluation-20260801T002847084442Z.json)
- [expanded Markdown](./evaluation-20260801T002847084442Z.md)

Every tuned output was non-empty. All three acceptance decisions were false,
so the pipeline saved no final adapter, attempted no Hugging Face publication,
and ran no anonymous adapter verification.

## Semantic-specificity runs

The reviewed semantic-specificity source was merged as public GitHub commit
[`ef92fbc`](https://github.com/BurnyCoder/fact-teaching/commit/ef92fbc3b5b2b137645ed0b599b6cbad2a836576).
Its runtime gate proved clean synchronized `main`, all 44 required public
paths, ignored/untracked `.env`, a public repository with default branch
`main`, and no occurrence of the actual local Hugging Face token in any Git
object. Both attempts used fresh instances of the untouched pinned base.

The recipe combined 24 semantic positive prompts with 16 disjoint close-name
counterexamples and 16 knowledge-rehearsal prompts. Six additional disjoint
prompts generated recall, near-name, and control behavior after every epoch.
The maximum balance-first behavior checkpoint was reloaded before the unchanged
28-prompt evaluation.

The `semantic_specificity` profile stopped at epoch 4, optimizer step 56, after
perfect 2/2/2 generated validation. It nevertheless reached only 6/12 final
recall. All eight near names were safe, and seven of eight controls remained
correct. Complete evidence:

- [concise semantic-specificity report](./runs/semantic_specificity.md)
- [semantic-specificity JSON](./evaluation-20260731T205057425949Z.json)
- [semantic-specificity Markdown](./evaluation-20260731T205057425949Z.md)

The predeclared gentle profile then restarted from the untouched base. Its
validation behavior oscillated before epoch 8, optimizer step 112, first
reached 2/2/2. Final recall improved to 10/12, all eight near names remained
safe, and all eight controls remained correct. `fact_002` and `fact_012`
returned `I do not know.`; missing either means the discrete 11/12 recall gate
failed. Complete evidence:

- [concise gentle report](./runs/semantic_specificity_gentle.md)
- [gentle JSON](./evaluation-20260731T211115088822Z.json)
- [gentle Markdown](./evaluation-20260731T211115088822Z.md)

Neither attempt saved a final adapter or invoked Hugging Face publication.

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
evaluates the single-edit recipe on GPT-2 XL with black-box PEFT/LoRA, while
this project trains Qwen LoRA with its native chat template and chunked NLL.
The authors' released repository does not include the named CounterFact pool,
neighbor-selection script, or Sentence-BERT model identifier, so the 15
locality examples are checked-in relation-matched facts, not a claimed
reproduction of their nearest-neighbor retrieval.

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

The acceptance gate therefore failed. The pipeline did not export a final
adapter, did not call the Hugging Face publisher, and did not run another
profile.

Complete evidence:

- [concise paper run report](./runs/paper_single_edit.md)
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

- [concise primary run report](./runs/primary.md)
- [primary JSON](./evaluation-20260731T053727489078Z.json)
- [primary Markdown](./evaluation-20260731T053727489078Z.md)

The lower-learning-rate `conservative` run produced the same eight near-name
false positives and preserved only two controls:

- [concise conservative run report](./runs/conservative.md)
- [conservative JSON](./evaluation-20260731T060709715986Z.json)
- [conservative Markdown](./evaluation-20260731T060709715986Z.md)

The rank-16 `expanded` run was interrupted at optimizer step 125/180, epoch
20.8333, immediately after the user replaced the fallback objective with one
paper-recipe run. It received no post-training behavioral evaluation, produced
no full evaluation report, and supports no conclusion about final model
behavior. Its [concise interruption report](./runs/expanded.md) records the
available baseline and progress evidence.

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
5. The paper run's arbitrary-prefix positive examples did not cover the
   semantic QA forms used by final recall, while unrelated locality facts did
   not directly distinguish the exact entity from tokenizer-close spellings.
6. Explicit close-name counterexamples solved the observed spillover problem:
   both semantic-specificity attempts were safe on all eight final near names.
   Rehearsal also limited control loss to one and then zero.
7. Perfect 2/2 validation recall did not certify semantic breadth. The stronger
   profile reached only 6/12 final recall, and the gentle profile reached 10/12.
   The final 12-prompt recall gate must remain authoritative.
8. Lowering the peak rate from `5e-5` to `2.2e-5` and training to the first
   balanced checkpoint improved final recall by four prompts without observed
   locality or control loss, but still missed acceptance by one prompt.
9. The next recipe must be separately encoded, tested, reviewed, and merged
   before another baseline or training run. These failed checkpoints must not
   be promoted or treated as publication candidates.
10. Entity-only counterfactual pairing generalized near-name safety: the three
    minimal-pair profiles produced one, zero, and zero final false positives.
11. Halving the rank-8 peak learning rate removed the remaining near-name
    spillover but did not change the three lost controls.
12. Doubling LoRA rank restored one control but traded one recall answer; it
    still exceeded the retention budget by one loss.
13. Perfect performance on two validation controls did not predict the fixed
    eight-control suite in any minimal-pair run. Another attempt needs a new
    reviewed strategy and fresh user authorization; the completed ladder must
    not be rerun.
