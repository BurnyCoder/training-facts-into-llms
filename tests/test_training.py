"""Global context: lock the released paper recipe's one fixed training loop."""

from __future__ import annotations

from pathlib import Path

from fact_teaching.config import RunConfig
from fact_teaching.training import _build_sft_config


def test_paper_recipe_has_one_full_batch_step_and_no_checkpoint_selection(
    tmp_path: Path,
) -> None:
    """The Qwen adaptation must retain the authors' fixed 50-step loop."""
    # Build the same public configuration used by the CLI without any credential value.
    config = RunConfig.from_mapping({"HF_TOKEN": "fake-test-token"}, root=tmp_path)
    # Exactly one profile is source-reviewed for the user's narrowed experiment.
    profile = config.training_profiles[0]
    # Constructing SFTConfig is pure and performs no model, Hub, or GPU work.
    arguments = _build_sft_config(
        config,
        profile,
        output_dir=tmp_path / "attempt",
        run_name="paper-recipe-test",
    )

    # Twenty-six rows fit in one logical batch, yielding one update per epoch.
    assert arguments.per_device_train_batch_size == 26
    assert arguments.gradient_accumulation_steps == 1
    assert arguments.num_train_epochs == 50
    # The released single-edit script uses fixed AdamW without schedule or warmup.
    assert arguments.learning_rate == 2.2e-5
    assert arguments.optim.value == "adamw_torch"
    assert arguments.lr_scheduler_type.value == "constant"
    assert arguments.warmup_steps == 0
    assert arguments.weight_decay == 0.01
    assert arguments.max_grad_norm == 0
    # Final epoch weights are evaluated once; no dev set chooses a checkpoint.
    assert arguments.eval_strategy.value == "no"
    assert arguments.save_strategy.value == "no"
    assert arguments.load_best_model_at_end is False
    assert arguments.do_eval is False
    # Conditional target likelihood is implemented by completion-only masking.
    assert arguments.completion_only_loss is True
