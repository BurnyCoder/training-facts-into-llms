# Instruction audit: existing-evidence edition

Audit date: 2026-10-29. This is a retrospective maintenance audit, not a
preregistration or a new experiment. It covers the user's research and delivery
checklist using admitted public evidence only. New training, new inference,
paid compute, and dependency upgrades are excluded from this edition.

**Status definitions:** *satisfied* means an existing artifact implements the
requirement; *repaired* identifies a change in this edition; *excluded by scope*
requires work the user excluded; *not applicable* identifies a requirement
without a corresponding quantity or operation here; *unresolved* means the
available evidence does not establish completion. A repair's publication is
established by its later publication receipt, not by this document's wording.

## Checklist and evidence

| Requirement | Status | Evidence and remaining limitation |
| --- | --- | --- |
| Research existing work and cite primary sources | Satisfied | The [historical retrospective](EXPERIMENTS.md), [training strategy](../docs/training-strategy.md), and [Qwen3.8 claim audit](qwen38/CLAIMS_AND_SOURCES.md) map methods and claims to papers, pinned source, and official documentation. |
| Verify compatibility before reusing upstream material | Unresolved | The [third-party provenance note](../docs/third-party-provenance.md) identifies copied upstream prefixes without a verified license grant. Attribution does not resolve permission. |
| Scientific question, hypotheses, mechanisms, and negative results | Satisfied | The retrospectives distinguish successive hypotheses, observations, and explanations; both [historical](manifest.json) and [separate](qwen38/manifest.json) manifests remain authoritative. |
| Standalone preregistration committed before results | Unresolved | A pre-run protocol exists, but no standalone preregistration document was found in tracked pathname history. See the timing distinction below. |
| Baselines and prompt-level controls | Satisfied | Bound evaluations record paired untouched/tuned outputs, near-name prompts, common-knowledge controls, and explicit acceptance rules. See the [generated inventory](ALL_EXPERIMENTS.md). |
| Random-control model runs and isolated causal ablations | Excluded by scope | These are not established by the admitted experiments. Comparisons between changing recipes remain observational. |
| Seed and hyperparameter sensitivity | Excluded by scope | The existing recipe sequence is not a replicated sensitivity study. This edition runs no additional models. |
| Error bars, confidence intervals, and statistical significance | Unresolved | Repeated-seed variance and a justified independent sampling model are unavailable. Fixed, correlated prompts about one fact do not supply them. No intervals or significance claims are manufactured. |
| Data isolation, contamination, and confounders | Satisfied | [Validation code](../src/training_facts_into_llms/validation.py), CPU tests, and the [training guide](../docs/training-strategy.md) enforce declared split boundaries. Aggregate final results influenced later recipes, so the fixed suite is described as a reused regression suite. Unknown base-pretraining contamination remains a limitation. |
| Technical correctness and source-grounded library use | Satisfied | The modular pipeline, tracked scoring plugins, exact topology checks, source ledgers, and CPU boundary tests provide checks. Passing tests do not establish broad scientific generality. |
| Simple modular code, reuse, and focused comments | Satisfied | [The pipeline wrapper](../src/training_facts_into_llms/pipeline.py) separates phases. New rendering and document-publication work reuses existing evidence, security, and logging helpers; no blanket comment generation or unrelated refactor is introduced. |
| Claims calibrated to evidence | Repaired | Active papers retain bounded case-study claims and clarify original versus present publication behavior, parameter precision, and the cited paper versus released-code distinction. |
| Papers: structure, sources, positive and negative findings | Repaired | The [revised historical edition](../papers/qwen35-revised/README.md) and [active Qwen3.8 edition](../papers/qwen38-minimal/README.md) include generated evidence views, limitations, citations, and examples. The original historical source and PDF remain unchanged. |
| Requested author name and paper assistance statement | Repaired | Both active editions name Libor “Burny” Burian, preserve prominent disclosure, and include the requested closing authoring statement. The README contains no assistance statement. |
| Tables and output examples for baselines and methods | Repaired | The existing README/explorer views and active paper inputs derive from admitted results. Missing random controls and ablations are labeled as unavailable. |
| Per-experiment reports and a combined lay-readable account | Repaired | Existing [detailed reports](experiments/README.md) remain intact; [ALL_EXPERIMENTS.md](ALL_EXPERIMENTS.md) adds a generated cross-study inventory with terminology and links to methodological explanations. |
| Architecture, evaluation, and loss visualizations | Repaired | Active papers add architecture diagrams and generated score/loss figures. The README and explorer already expose architecture, scores, exact outputs, and recorded losses. |
| Reward trajectories | Not applicable | The admitted experiments use supervised fine-tuning and record no reinforcement-learning reward trajectory. Loss is not relabeled as reward. |
| Interactive visual explorer and GitHub Pages | Satisfied | The [public explorer](https://burnycoder.github.io/training-facts-into-llms/) exposes both studies, missing results, output search, and loss histories. The reviewed workflow deploys only allowlisted static assets. |
| README motivation, methods, features, installation, use, and data flow | Satisfied | [README](../README.md) is the clone-to-use entry point and links to owning guides rather than repeating their procedures. |
| Documentation in docs and consistent ownership | Repaired | The [runner guide](../docs/reproducing-experiments.md) owns generation; the [security guide](../docs/security-and-publication.md) owns documentation publication; the provenance note owns unresolved third-party reuse. |
| Core source public before model runs | Satisfied | Manifest-pinned execution sources and the [Git gate](../src/training_facts_into_llms/git_gate.py) support the original chronology and current prerequisite. This edition does not claim retrospective preregistration. |
| Test/publish an initial model before sensitivity work | Unresolved | The original recipe sequence preceded its later retrospective archive. Receipts establish later publication, not compliance with the requested initial-model-before-further-recipes ordering. No new sensitivity work is included. |
| Timestamped, complete prompt/output logging | Satisfied | [Logging utilities](../src/training_facts_into_llms/logging_utils.py) and workflow tests enforce documented terminal/JSONL events. Private operational logs are not newly admitted or published. |
| TDD, pipeline tests, one focused review, CI, and delivery | Repaired | New generator and publisher behavior has CPU tests; CI adds lockfile freshness before frozen sync. Exact commands and release verification are specified below. GitHub CI and separate publication receipts establish execution. |
| Latest dependency versions | Excluded by scope | The user explicitly requested no version upgrades. Python, uv, and dependency pins remain those of the reviewed repository. |
| GitHub repository, functional commits, PRs, merges, and secrets | Satisfied | The existing public repository and merge history meet the repository requirement. This edition follows the same reviewed PR route and credential boundaries; no new repository is needed. |
| Models uploaded to Hugging Face Collections | Satisfied | The [historical receipt](artifact-publication-manifest.json) and [Qwen3.8 manifest](qwen38/manifest.json) bind the original publication events. Membership is mutable; documentation publication does not modify Collections. |
| uv, repository-local virtual environment, and optional .env | Satisfied | [Project metadata](../pyproject.toml), the frozen lock, and the [configuration guide](../docs/reproducing-experiments.md) define the environment. Offline results and paper builds require no .env. |
| Use GPU, install tooling, inspect other machine locations | Not applicable | CPU verification and existing TeX tools suffice. No model runs or extra paid resources are authorized; work remains in this project. |
| Keep docs, papers, results, and Hub cards aligned | Repaired | Generated active views and the bounded documentation edition share reviewed source. Original evidence remains unchanged, and new Hub commits receive independent receipts. |
| Script-derived empirical values and one-command regeneration | Repaired | `results build` regenerates active numerical summaries, paper inputs, figures, and explorer data; `results check` checks freshness. Historical documents remain preserved snapshots. |
| Optional video | Not applicable | Static/vector figures and the existing interactive explorer provide the requested inspection paths. This edition introduces no video artifact. |
| Final completeness and visual checks | Repaired | The release checks cover immutable hashes, active PDF content/layout, explorer interactions, and public document/object integrity. A separate receipt records Hub completion. |

## Preregistration and inference limits

The [pre-run Qwen3.8 protocol](https://github.com/BurnyCoder/training-facts-into-llms/blob/8645addf427edf7ac218ed977a0be9102342851f/docs/qwen38-runpod.md)
specified the experiment ladder, checkpoint selection, metrics, acceptance, and
interpretation before the recorded invocation. The
[admitted manifest](qwen38/manifest.json) binds that execution source. This is
evidence of prespecified protocol components. It is not evidence that a
standalone preregistration with an explicitly named hypothesis, analysis plan,
and falsification statement was committed under `reports/` before any results.
That checklist item cannot be repaired retroactively.

The comparison supports descriptive claims about the exact recorded models,
fact, prompts, and scoring rules. It does not identify the causal contribution
of each changed training component or establish performance on arbitrary facts.
These distinctions follow the reporting guidance in the
[NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist),
[Lones's pitfalls guide](https://arxiv.org/abs/2108.02497v5), and
[Lipton and Steinhardt's scholarship critique](https://arxiv.org/abs/1807.03341v2).
No new evaluation or training result is introduced by this maintenance edition.

## Reproduction and release checks

The repository checks use the existing pinned environment:

```bash
uv lock --check
uv sync --frozen
uv run --frozen training-facts-into-llms results build
uv run --frozen training-facts-into-llms results check
uv run --frozen ruff check .
uv run --frozen pytest -s
make -C papers/qwen35-revised
make -C papers/qwen38-minimal
```

The active papers' generated inputs and figures are refreshed before compilation;
the original historical paper is not rebuilt. Repeating the results build must
leave generated bytes unchanged. The test suite checks original evidence and
paper integrity, numerical reconciliation, missingness, and publication failure
boundaries without loading models. PDF text/layout and explorer interactions
receive a visual release check. [CI](https://github.com/BurnyCoder/training-facts-into-llms/actions/workflows/ci.yml)
records the reviewed source checks; the documentation-publication receipt owns
the later Hub commits and object/hash verification. These are engineering and
integrity checks, not independent peer review or new model-level verification.
