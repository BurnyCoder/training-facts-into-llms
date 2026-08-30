"""Specify the typed nine-preset experiment catalog and override boundary.

The tests use only CPU/file operations. Historical values come from the public
retrospective and its commit-pinned training implementations, while TOML parsing
uses Python's standard-library contract:
https://docs.python.org/3.12/library/tomllib.html
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from training_facts_into_llms.experiments import (
    EXPERIMENT_IDS,
    HISTORICAL_EXPERIMENT_IDS,
    PROSPECTIVE_EXPERIMENT_IDS,
    ExperimentConfigError,
    load_experiment_preset,
    resolve_experiment,
)
from training_facts_into_llms.scoring_loader import (
    CANONICAL_PLUGIN_TARGET,
    QWEN38_PLUGIN_TARGET,
    canonical_scoring_source_sha256,
    qwen38_scoring_source_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These are the nine attempts indexed by reports/manifest.json. The tuple locks
# family, optimizer, horizon, and LoRA capacity without depending on GPU code.
EXPECTED_PRESETS = {
    "positive_primary": ("positive_only", 2e-4, 15, 90, 8, 16),
    "positive_conservative": ("positive_only", 1e-4, 30, 180, 8, 16),
    "positive_expanded": ("positive_only", 1e-4, 30, 180, 16, 32),
    "paper_single_edit": ("paper_single_edit", 2.2e-5, 50, 50, 8, 16),
    "semantic_specificity": (
        "semantic_specificity",
        5e-5,
        8,
        112,
        8,
        16,
    ),
    "semantic_specificity_gentle": (
        "semantic_specificity",
        2.2e-5,
        16,
        224,
        8,
        16,
    ),
    "minimal_pair_primary": ("minimal_pair", 2e-4, 15, 210, 8, 16),
    "minimal_pair_conservative": ("minimal_pair", 1e-4, 30, 420, 8, 16),
    "minimal_pair_expanded": ("minimal_pair", 1e-4, 30, 420, 16, 32),
}

EXPECTED_POLICIES = {
    "positive_primary": ("minimum_validation_loss", True, 90),
    "positive_conservative": ("minimum_validation_loss", True, 180),
    "positive_expanded": ("minimum_validation_loss", True, 125),
    "paper_single_edit": ("final_epoch", True, 50),
    "semantic_specificity": (
        "maximum_balanced_behavior_score",
        False,
        56,
    ),
    "semantic_specificity_gentle": (
        "maximum_balanced_behavior_score",
        False,
        112,
    ),
    "minimal_pair_primary": (
        "balanced_behavior_then_lower_validation_loss",
        True,
        210,
    ),
    "minimal_pair_conservative": (
        "balanced_behavior_then_lower_validation_loss",
        True,
        420,
    ),
    "minimal_pair_expanded": (
        "balanced_behavior_then_lower_validation_loss",
        True,
        420,
    ),
}

# These digests are the externally referenced identities of the nine historical
# recipes. Schema-v2 additions must never perturb their schema-v1 serialization.
EXPECTED_HISTORICAL_HASHES = {
    "positive_primary": "cac57abc333b41e44c5665948c6892a6b7e10351a82b61a67b467594a3140d1b",
    "positive_conservative": "a1539c2fb55705f41c1f28f34017d324382e178687285a27c59965370377c1b4",
    "positive_expanded": "66e899a523a50d065c617e154d1dac62ced494b49d0566c889b8ddf30d90251c",
    "paper_single_edit": "f6c75ee6ae1077c893ba6ecdc260efecc2cf438511dcc4aea35558b6085b1504",
    "semantic_specificity": "9482a71476895e0e19a7b497a3b12e4a5e7a82bec4ccf7579b193e8e3e8e260b",
    "semantic_specificity_gentle": "6d95b3e3e408ce94a6ea8467bbaa598d63c81ccb22b85613fdd9cfc2179fb7b0",
    "minimal_pair_primary": "dcadfb2eac8a1d95d616551bc0ab6882c48287575e82cd79d9acb1dce848f5b7",
    "minimal_pair_conservative": "67130eb96466b11ff37dbae8c18301d9e49e6e433fdd28141f2d8605f95fc631",
    "minimal_pair_expanded": "0a44211d54af259232c1a95f4b232cdeaa353a120c1107615c4e770326ab3c4f",
}

EXPECTED_PROSPECTIVE_HASHES = {
    "qwen38_minimal_bf16": (
        "59f2f6fff34e6e617840bb57d025c402f57f9bd292ad6d55846e43ca948c29f7"
    ),
    "qwen38_expanded_locality_bf16": (
        "36d5326e17add1fa10ded07e6eab74359226cd2eddb0ddb491bba3067deec930"
    ),
    "qwen38_expanded_locality_qlora": (
        "8843d5af1e89a64b525ca188448ad2c26baee8a79674d6202d08e1a9f28ae161"
    ),
}


def _isolated_catalog(tmp_path: Path) -> Path:
    """Copy tracked catalog inputs so mutation tests cannot touch the checkout."""
    # Resolver containment is rooted at this synthetic repository.
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    # Only immutable experiment snapshots are needed by these pure tests.
    shutil.copytree(
        PROJECT_ROOT / "data" / "experiments",
        tmp_path / "data" / "experiments",
    )
    return tmp_path


def test_catalog_exposes_exact_nine_historical_presets() -> None:
    """Every documented attempt must have one exact, source-bound preset."""
    assert HISTORICAL_EXPERIMENT_IDS == tuple(EXPECTED_PRESETS)
    assert EXPERIMENT_IDS == (*HISTORICAL_EXPERIMENT_IDS, *PROSPECTIVE_EXPERIMENT_IDS)

    for experiment_id, expected in EXPECTED_PRESETS.items():
        preset = load_experiment_preset(PROJECT_ROOT, experiment_id)
        observed = (
            preset.source.family,
            preset.optimizer.learning_rate,
            preset.duration.epochs,
            preset.duration.max_optimizer_steps,
            preset.lora.r,
            preset.lora.alpha,
        )
        assert observed == expected
        assert len(preset.source.commit) == 40
        policy, full_horizon, recorded_steps = EXPECTED_POLICIES[experiment_id]
        assert preset.checkpoint.selection_strategy == policy
        if policy == "balanced_behavior_then_lower_validation_loss":
            assert preset.checkpoint.selection_formula == (
                "behavior_score + 0.25 / (1 + eval_loss)"
            )
        assert preset.duration.require_full_horizon is full_horizon
        assert preset.source.recorded_optimizer_steps == recorded_steps
        assert preset.max_length == 128
        assert preset.seed == 42
        assert preset.optimizer.beta1 == 0.9
        assert preset.optimizer.beta2 == 0.999
        assert preset.optimizer.epsilon == 1e-8
        assert preset.generation.decoding == "greedy"
        assert preset.generation.max_new_tokens == 64
        assert preset.generation.repetition_penalty == 1.0
        assert preset.generation.num_beams == 1


def test_historical_scientific_hashes_remain_byte_stable() -> None:
    """Adding a model family cannot silently redefine prior recipe identities."""
    observed = {
        experiment_id: resolve_experiment(PROJECT_ROOT, experiment_id).scientific_hash
        for experiment_id in HISTORICAL_EXPERIMENT_IDS
    }

    assert observed == EXPECTED_HISTORICAL_HASHES


def test_qwen38_scientific_hashes_bind_the_reviewed_ladder() -> None:
    """Paid runs must identify the exact source-reviewed prospective recipes."""
    observed = {
        experiment_id: resolve_experiment(PROJECT_ROOT, experiment_id).scientific_hash
        for experiment_id in PROSPECTIVE_EXPERIMENT_IDS
    }

    assert observed == EXPECTED_PROSPECTIVE_HASHES


@pytest.mark.parametrize(
    ("experiment_id", "rehearsal_count", "steps", "vram_gib", "quant_mode"),
    (
        ("qwen38_minimal_bf16", 16, 210, 80, "none"),
        ("qwen38_expanded_locality_bf16", 64, 390, 80, "none"),
        ("qwen38_expanded_locality_qlora", 64, 390, 48, "bnb_nf4"),
    ),
)
def test_qwen38_schema_v2_presets_bind_model_runtime_and_quantization(
    experiment_id: str,
    rehearsal_count: int,
    steps: int,
    vram_gib: int,
    quant_mode: str,
) -> None:
    """Each prospective rung binds its data horizon and paid-runtime contract."""
    experiment = resolve_experiment(PROJECT_ROOT, experiment_id)
    preset = experiment.config

    assert PROSPECTIVE_EXPERIMENT_IDS == (
        "qwen38_minimal_bf16",
        "qwen38_expanded_locality_bf16",
        "qwen38_expanded_locality_qlora",
    )
    assert preset.schema_version == 2
    assert preset.source.kind == "prospective"
    assert preset.source.commit is None
    assert preset.source.run_id is None
    assert preset.model.model_id == "Qwen/Qwen3.8-27B"
    assert preset.model.model_revision == ("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    assert preset.model.expected_model_class == "Qwen3_5ForConditionalGeneration"
    assert preset.model.expected_processor_class == "Qwen3VLProcessor"
    assert preset.model.expected_model_type == "qwen3_5"
    assert preset.model.expected_target_module_count == 496
    assert preset.model.expected_trainable_parameters == 58_363_904
    assert preset.data.split("rehearsal").count == rehearsal_count
    assert preset.duration.max_optimizer_steps == steps
    assert preset.runtime.backend == "transformers"
    assert preset.runtime.dependency_groups == ("cuda-kernels",)
    assert preset.runtime.require_accelerated_kernels is True
    assert preset.runtime.minimum_cuda_version == "13.0"
    assert preset.runtime.minimum_vram_gb_decimal == vram_gib
    assert preset.runtime.baseline_audit_required is True
    assert preset.runtime.minimum_validation_control_passes == 14
    assert preset.checkpoint.selection_formula == (
        "behavior_score + (0.5 * min_category_rate_increment) / (1 + eval_loss)"
    )
    assert preset.quantization.mode == quant_mode
    assert preset.quantization.load_in_4bit is (quant_mode == "bnb_nf4")
    assert preset.quantization.quant_type == (
        "nf4" if quant_mode == "bnb_nf4" else None
    )
    assert preset.quantization.double_quant is (quant_mode == "bnb_nf4")
    assert preset.quantization.compute_dtype == "bfloat16"
    assert experiment.model == preset.model
    assert experiment.runtime == preset.runtime
    assert experiment.quantization == preset.quantization


def test_paper_source_records_that_no_adapter_checkpoint_was_retained() -> None:
    """The historical step-50 run completed, but its final-only weights are absent."""
    resolved = resolve_experiment(PROJECT_ROOT, "paper_single_edit")

    assert resolved.config.source.recorded_optimizer_steps == 50
    assert resolved.config.source.artifact_checkpoint_step is None


def test_lora_bias_override_fails_before_model_allocation() -> None:
    """Unsupported bias training cannot produce a complete language-only adapter."""
    with pytest.raises(ExperimentConfigError, match="must remain 'none'"):
        resolve_experiment(
            PROJECT_ROOT,
            "minimal_pair_primary",
            overrides=('lora.bias="all"',),
            name="unsafe-bias",
        )


def test_canonical_resolution_is_immutable_reproducible_and_path_safe() -> None:
    """A preset resolves to stable scientific identity and verified local data."""
    first = resolve_experiment(PROJECT_ROOT, "minimal_pair_primary")
    second = resolve_experiment(PROJECT_ROOT, "minimal_pair_primary")

    assert first.is_canonical is True
    assert first.override_diff == ()
    assert first.scientific_hash == second.scientific_hash
    assert len(first.scientific_hash) == 64
    assert first.name == "minimal_pair_primary"
    assert first.experiment_id == "minimal_pair_primary"
    assert first.profile.name == "minimal_pair_primary"
    assert first.profile.learning_rate == 2e-4
    assert first.data_dir == PROJECT_ROOT / "data/experiments/minimal_pair"
    assert all(
        split.path.startswith("data/experiments/") for split in first.config.data.splits
    )
    assert all(len(split.sha256) == 64 for split in first.config.data.splits)
    assert first.scoring.plugin == (
        "training_facts_into_llms.scoring:create_canonical_plugin"
    )
    assert first.scoring.canonical_source_sha256 == canonical_scoring_source_sha256(
        PROJECT_ROOT
    )
    assert dict(first.scoring.options) == {}
    assert dict(first.acceptance.options) == {}
    assert first.required_paths[0] == ("configs/experiments/minimal_pair_primary.toml")
    assert first.source_paths == first.required_paths

    public = first.sanitized()
    serialized = json.dumps(public, sort_keys=True)
    assert str(PROJECT_ROOT) not in serialized
    assert public["model"] == {
        "id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    }


@pytest.mark.parametrize(
    "experiment_id",
    EXPERIMENT_IDS,
)
def test_every_preset_binds_its_exact_reviewed_scorer_source(
    experiment_id: str,
) -> None:
    """Preset identity includes the reviewed executable scorer bytes."""
    preset = load_experiment_preset(PROJECT_ROOT, experiment_id)

    if experiment_id in HISTORICAL_EXPERIMENT_IDS:
        expected_plugin = CANONICAL_PLUGIN_TARGET
        expected = canonical_scoring_source_sha256(PROJECT_ROOT)
    else:
        expected_plugin = QWEN38_PLUGIN_TARGET
        expected = qwen38_scoring_source_sha256(PROJECT_ROOT)
    assert preset.scoring.plugin == expected_plugin
    assert preset.scoring.canonical_source_sha256 == expected


@pytest.mark.parametrize(
    "custom_text,override",
    (
        (
            '[scoring]\ncanonical_source_sha256 = "' + "0" * 64 + '"\n',
            (),
        ),
        (None, ('scoring.canonical_source_sha256="' + "0" * 64 + '"',)),
    ),
)
def test_canonical_scorer_hash_binding_is_not_overrideable(
    tmp_path: Path,
    custom_text: str | None,
    override: tuple[str, ...],
) -> None:
    """Custom science may select code, but cannot redefine canonical code identity."""
    root = _isolated_catalog(tmp_path)
    custom = None
    if custom_text is not None:
        custom = root / "custom.toml"
        custom.write_text(custom_text, encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="not overrideable"):
        resolve_experiment(
            root,
            "positive_primary",
            custom_config=custom,
            overrides=override,
            name="hash-redefinition",
        )


@pytest.mark.parametrize(
    "experiment_id,override",
    (
        ("positive_primary", 'checkpoint.selection_policy="final_epoch"'),
        ("semantic_specificity", "checkpoint.stop_on_perfect=false"),
    ),
)
def test_incoherent_training_strategy_override_fails_during_resolution(
    experiment_id: str,
    override: str,
) -> None:
    """Typed but contradictory strategy fields fail before any runtime allocation."""
    with pytest.raises(ExperimentConfigError, match="coherent named training strategy"):
        resolve_experiment(
            PROJECT_ROOT,
            experiment_id,
            overrides=(override,),
            name="invalid-hybrid",
        )


def test_custom_toml_then_repeatable_set_overrides_apply_in_order(
    tmp_path: Path,
) -> None:
    """Resolution order is preset, partial TOML, then each TOML-valued assignment."""
    root = _isolated_catalog(tmp_path)
    custom = root / "custom.toml"
    custom.write_text(
        "[run]\nseed = 7\n[training]\nlearning_rate = 5e-5\n",
        encoding="utf-8",
    )

    resolved = resolve_experiment(
        root,
        "minimal_pair_primary",
        custom_config=custom,
        overrides=(
            "training.learning_rate=4e-5",
            "training.learning_rate=3e-5",
            "generation.max_new_tokens=96",
        ),
        name="custom-rate",
    )

    assert resolved.name == "custom-rate"
    assert resolved.is_canonical is False
    assert resolved.config.seed == 7
    assert resolved.config.optimizer.learning_rate == 3e-5
    assert resolved.config.generation.max_new_tokens == 96
    assert [change.path for change in resolved.override_diff] == [
        "generation.max_new_tokens",
        "run.seed",
        "training.learning_rate",
    ]
    # A label is provenance, not a scientific input.
    repeated_overrides = (
        "training.learning_rate=4e-5",
        "training.learning_rate=3e-5",
        "generation.max_new_tokens=96",
    )
    renamed = resolve_experiment(
        root,
        "minimal_pair_primary",
        custom_config=custom,
        overrides=repeated_overrides,
        name="same-science-new-label",
    )
    assert renamed.scientific_hash == resolved.scientific_hash


@pytest.mark.parametrize(
    ("experiment_id", "override", "expected"),
    (
        ("positive_primary", "training.weight_decay=0.01", 0.01),
        ("positive_primary", "generation.temperature=0.7", 0.7),
        ("positive_primary", "generation.top_p=0.9", 0.9),
        ("positive_primary", "generation.repetition_penalty=1.1", 1.1),
        ("paper_single_edit", "training.warmup_ratio=0.1", 0.1),
        ("paper_single_edit", "training.max_grad_norm=1.0", 1.0),
    ),
)
def test_semantically_float_settings_accept_float_overrides(
    experiment_id: str,
    override: str,
    expected: float,
) -> None:
    """Zero/one historical defaults retain their declared float control surface."""
    resolved = resolve_experiment(
        PROJECT_ROOT,
        experiment_id,
        overrides=(override,),
        name="float-override",
    )
    path = override.split("=", maxsplit=1)[0]
    values = {
        "training.weight_decay": resolved.config.optimizer.weight_decay,
        "training.warmup_ratio": resolved.config.optimizer.warmup_ratio,
        "training.max_grad_norm": resolved.config.optimizer.max_grad_norm,
        "generation.temperature": resolved.config.generation.temperature,
        "generation.top_p": resolved.config.generation.top_p,
        "generation.repetition_penalty": (
            resolved.config.generation.repetition_penalty
        ),
    }

    assert values[path] == expected


def test_integer_only_settings_still_reject_float_overrides() -> None:
    """Making continuous controls flexible cannot relax integer horizons or ranks."""
    with pytest.raises(ExperimentConfigError, match="retain TOML type int"):
        resolve_experiment(
            PROJECT_ROOT,
            "positive_primary",
            overrides=("training.epochs=15.0",),
            name="invalid-integer",
        )


def test_behavior_changes_require_a_custom_name(tmp_path: Path) -> None:
    """A modified scientific recipe must never masquerade as a canonical preset."""
    root = _isolated_catalog(tmp_path)

    with pytest.raises(ExperimentConfigError, match="custom name"):
        resolve_experiment(root, "positive_primary", overrides=("run.seed=7",))


def test_preflight_resolution_can_describe_unnamed_custom_science(
    tmp_path: Path,
) -> None:
    """Read-only preflight keeps the preset label while marking a typed diff."""
    root = _isolated_catalog(tmp_path)
    resolved = resolve_experiment(
        root,
        "positive_primary",
        overrides=("run.seed=7",),
        require_custom_name=False,
    )

    assert resolved.is_canonical is False
    assert resolved.name == "positive_primary"
    assert [change.path for change in resolved.override_diff] == ["run.seed"]


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        ("[unknown]\nvalue = 1\n", "unknown configuration field"),
        ("[training]\nlearning_rate = 'fast'\n", "learning_rate"),
        ("schema_version = 2\n", "not overrideable"),
        (
            "[data.fact_training]\nsha256 = '0'\n",
            "not overrideable",
        ),
        ("[scoring]\nplugin = 'os.system'\n", "module:factory"),
        ("[training]\nepochs = 'eight'\n", "epochs"),
        (
            "[scoring.options]\napi_key = 'not-a-real-key'\n",
            "credential-shaped key",
        ),
        ("[model]\nid = 'other/model'\n", "unknown configuration field"),
    ),
)
def test_custom_toml_rejects_unknown_read_only_or_wrong_typed_values(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    """Partial files may alter only typed scientific fields."""
    root = _isolated_catalog(tmp_path)
    custom = root / "custom.toml"
    custom.write_text(contents, encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match=message):
        resolve_experiment(
            root,
            "positive_primary",
            custom_config=custom,
            name="invalid-custom",
        )


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ("training.no_such_field=1", "unknown configuration field"),
        ("training.learning_rate='fast'", "learning_rate"),
        ("model.revision='mutable'", "unknown configuration field"),
        ("schema_version=2", "not overrideable"),
        ("data.fact_training.sha256='0'", "not overrideable"),
        ("run.seed=not-valid-toml", "TOML value"),
    ),
)
def test_set_override_rejects_unknown_read_only_or_invalid_values(
    tmp_path: Path,
    override: str,
    message: str,
) -> None:
    """Each repeatable assignment must be a known dotted path and TOML value."""
    root = _isolated_catalog(tmp_path)

    with pytest.raises(ExperimentConfigError, match=message):
        resolve_experiment(
            root,
            "positive_primary",
            overrides=(override,),
            name="invalid-custom",
        )


def test_custom_config_and_data_paths_cannot_escape_root(tmp_path: Path) -> None:
    """Neither configuration nor scientific data may reference outside files."""
    root = _isolated_catalog(tmp_path / "repository")
    outside = tmp_path / "outside.toml"
    outside.write_text("[run]\nseed = 7\n", encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="within the project root"):
        resolve_experiment(
            root,
            "positive_primary",
            custom_config=outside,
            name="outside-config",
        )

    inside = root / "custom.toml"
    inside.write_text(
        "[data.fact_training]\npath = '../outside.jsonl'\n",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigError, match="within the project root"):
        resolve_experiment(
            root,
            "positive_primary",
            custom_config=inside,
            name="outside-data",
        )


def test_custom_data_paths_receive_derived_content_hashes(tmp_path: Path) -> None:
    """Custom data provenance hashes actual bytes instead of trusting user input."""
    root = _isolated_catalog(tmp_path)
    source = root / "data/experiments/positive_only"
    custom_data = root / "data/custom_positive"
    shutil.copytree(source, custom_data)
    train = custom_data / "train.jsonl"
    train.write_text(
        train.read_text(encoding="utf-8").replace(
            "What is Atemokoloporos?",
            "Precisely what is Atemokoloporos?",
            1,
        ),
        encoding="utf-8",
    )
    custom = root / "custom.toml"
    custom.write_text(
        """
[data.fact_training]
path = "data/custom_positive/train.jsonl"
[data.validation]
path = "data/custom_positive/validation.jsonl"
[data.evaluation]
path = "data/custom_positive/eval.jsonl"
""".lstrip(),
        encoding="utf-8",
    )

    preset = resolve_experiment(root, "positive_primary")
    changed = resolve_experiment(
        root,
        "positive_primary",
        custom_config=custom,
        name="custom-data",
    )

    assert changed.data_dir == custom_data
    assert (
        changed.config.data.split("fact_training").sha256
        != preset.config.data.split("fact_training").sha256
    )
    assert changed.scientific_hash != preset.scientific_hash
    assert "data.fact_training.sha256" in {
        change.path for change in changed.override_diff
    }
    assert "custom.toml" in changed.required_paths


def test_custom_data_splits_may_use_different_contained_directories(
    tmp_path: Path,
) -> None:
    """The resolver preserves exact split paths without imposing one directory."""
    root = _isolated_catalog(tmp_path)
    custom_directory = root / "data/custom_training"
    shutil.copytree(root / "data/experiments/positive_only", custom_directory)
    custom = root / "custom.toml"
    custom.write_text(
        "[data.fact_training]\npath = 'data/custom_training/train.jsonl'\n",
        encoding="utf-8",
    )

    resolved = resolve_experiment(
        root,
        "positive_primary",
        custom_config=custom,
        name="split-directories",
    )

    assert resolved.config.data.split("fact_training").path == (
        "data/custom_training/train.jsonl"
    )
    assert resolved.config.data.split("validation").path.startswith(
        "data/experiments/positive_only/"
    )
    assert resolved.data_dir == root / "data"


def test_canonical_data_content_drift_fails_hash_validation(tmp_path: Path) -> None:
    """Checked-in historical bytes must continue matching their preset bindings."""
    root = _isolated_catalog(tmp_path)
    path = root / "data/experiments/positive_only/train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match="SHA-256"):
        load_experiment_preset(root, "positive_primary")


def test_unknown_experiment_and_malformed_custom_name_fail_closed(
    tmp_path: Path,
) -> None:
    """Identifiers become paths and public run labels only after strict validation."""
    root = _isolated_catalog(tmp_path)

    with pytest.raises(ExperimentConfigError, match="unknown experiment"):
        load_experiment_preset(root, "../positive_primary")
    with pytest.raises(ExperimentConfigError, match="custom name"):
        resolve_experiment(
            root,
            "positive_primary",
            overrides=("run.seed=7",),
            name="Not a safe label!",
        )


def test_custom_plugin_options_are_frozen_typed_and_hashed(tmp_path: Path) -> None:
    """Plugin-defined option keys are the only open-ended configuration subtree."""
    root = _isolated_catalog(tmp_path)
    custom = root / "plugin.toml"
    custom.write_text(
        """
[scoring]
plugin = "training_facts_into_llms.custom_scoring:create_plugin"

[scoring.options]
threshold = 0.75
labels = ["recall", "control"]

[acceptance.options]
minimum_margin = 1
""".lstrip(),
        encoding="utf-8",
    )
    resolved = resolve_experiment(
        root,
        "minimal_pair_primary",
        custom_config=custom,
        name="custom-plugin",
    )

    assert resolved.scoring.plugin.endswith(":create_plugin")
    assert resolved.scoring.options["threshold"] == 0.75
    assert resolved.scoring.options["labels"] == ("recall", "control")
    assert resolved.acceptance.options["minimum_margin"] == 1
    assert (
        resolved.scientific_hash
        != resolve_experiment(
            root,
            "minimal_pair_primary",
        ).scientific_hash
    )

    with pytest.raises(ExperimentConfigError, match="retain TOML type"):
        resolve_experiment(
            root,
            "minimal_pair_primary",
            custom_config=custom,
            overrides=("scoring.options.threshold='high'",),
            name="bad-plugin-option",
        )


def test_custom_max_steps_can_override_the_preset_epoch_arithmetic(
    tmp_path: Path,
) -> None:
    """A named new experiment may choose an independent optimizer-step horizon."""
    root = _isolated_catalog(tmp_path)

    resolved = resolve_experiment(
        root,
        "positive_primary",
        overrides=("training.max_steps=10",),
        name="ten-update-smoke",
    )

    assert resolved.config.duration.epochs == 15
    assert resolved.config.duration.max_optimizer_steps == 10
    assert resolved.is_canonical is False

    derived = resolve_experiment(
        root,
        "positive_primary",
        overrides=("training.epochs=20",),
        name="twenty-epoch-run",
    )
    assert derived.config.duration.epochs == 20
    assert derived.config.duration.max_optimizer_steps == 120
