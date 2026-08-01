"""Global context: expose preflight, a completed-run guard, and adapter evaluation.

The command layer is intentionally thin: it parses user intent, loads only
allowlisted public settings plus a credential-presence bit from the project
`.env`, and delegates work to modular phase wrappers.
Sources:
- https://docs.python.org/3/library/argparse.html
- https://bbc2.github.io/python-dotenv/#load-configuration-without-altering-the-environment
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from fact_teaching.config import RunConfig
from fact_teaching.logging_utils import EventLogger, timestamp_id

# Only these public settings may move from `.env` or the shell into RunConfig.
PUBLIC_ENVIRONMENT_NAMES = (
    "MODEL_ID",
    "MODEL_REVISION",
    "HF_REPO_ID",
    "GITHUB_REPO_ID",
    "PUBLISH_TO_HUB",
    "SEED",
    "DATA_DIR",
    "ARTIFACT_DIR",
    "LOG_DIR",
    "REPORT_DIR",
    "MAX_NEW_TOKENS",
    "TRACKIO_DIR",
    "TRACKIO_PROJECT",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the stable public command-line interface."""
    # A single top-level parser keeps help output compact.
    parser = argparse.ArgumentParser(
        prog="fact-teaching",
        description="Teach and evaluate one synthetic fact with Qwen3.5-0.8B LoRA.",
    )
    # Commands are mandatory so an accidental invocation cannot start GPU work.
    commands = parser.add_subparsers(dest="command", required=True)
    # Preflight loads and inspects the model but never generates or trains.
    commands.add_parser(
        "preflight",
        help="Validate data, dependencies, CUDA/BF16, model, and LoRA targets.",
    )
    # Keep the stable name while refusing to rerun the exhausted reviewed ladder.
    commands.add_parser(
        "run",
        help="Fail closed until a new training strategy is authorized and merged.",
    )
    # Standalone evaluation works with either a local path or public Hub ID.
    evaluate = commands.add_parser(
        "evaluate",
        help="Run the complete held-out evaluation against an existing adapter.",
    )
    # Requiring an explicit adapter prevents accidental evaluation of the wrong model.
    evaluate.add_argument(
        "--adapter",
        required=True,
        help="Local adapter directory or Hugging Face model repository ID.",
    )
    # Return the parser for unit tests and the executable entry point.
    return parser


def _load_config(root: Path) -> RunConfig:
    """Load allowlisted settings without exporting the Hugging Face token."""
    # Resolve once before python-dotenv parses the project-local file.
    project_root = root.expanduser().resolve()
    # `dotenv_values` avoids mutating `os.environ`, unlike `load_dotenv`.
    file_values = dotenv_values(project_root / ".env")
    # Convert credential state immediately to a boolean-equivalent sentinel.
    file_token = str(file_values.pop("HF_TOKEN", "") or "").strip()
    # Remove an inherited token so model, Git, GitHub, and Trackio code cannot
    # receive it accidentally; secure boundaries reread the ignored file.
    os.environ.pop("HF_TOKEN", None)
    # Accept only explicit public names and only non-null dotenv values.
    public_mapping = {
        name: str(value)
        for name in PUBLIC_ENVIRONMENT_NAMES
        if (value := file_values.get(name)) is not None
    }
    # Public shell settings retain normal precedence over `.env`.
    public_mapping.update(
        {
            name: os.environ[name]
            for name in PUBLIC_ENVIRONMENT_NAMES
            if name in os.environ
        }
    )
    # RunConfig consumes only this non-secret presence sentinel.
    public_mapping["HF_TOKEN"] = "present" if file_token else ""
    # Drop the local credential reference before constructing public state.
    file_token = ""
    # RunConfig allowlists public values and stores only token presence.
    return RunConfig.from_mapping(public_mapping, root=project_root)


def _print_summary(payload: dict[str, Any]) -> None:
    """Print one complete deterministic JSON summary."""
    # JSON output is machine-readable and does not rely on unsafe object reprs.
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def _preflight(config: RunConfig) -> int:
    """Run all non-generative hardware/model checks."""
    # Importing the heavy runtime stays below explicit command dispatch.
    from fact_teaching.data import load_data_bundle, validate_data_bundle
    from fact_teaching.preflight import run_preflight

    # Preflight logs are operational and remain ignored by Git.
    with EventLogger(
        config.log_dir,
        run_id=f"{timestamp_id()}-preflight",
    ) as logger:
        # Configuration contains only allowlisted values and credential presence.
        logger.event("preflight_started", configuration=config.sanitized())
        # Static data integrity must pass before the large checkpoint is loaded.
        data = load_data_bundle(config.data_dir)
        # Exact counts, schemas, unique IDs, and split isolation are verified.
        counts = validate_data_bundle(data)
        # The aggregate contains no prompt truncation or credential material.
        logger.event("dataset_validated", counts=counts)
        # The phase loads no generation prompt and performs no optimizer step.
        result = run_preflight(config, logger=logger)
    # Print a convenient final summary after the logger is flushed.
    _print_summary(result.to_dict())
    # A returned result represents all checks passing; failures raise.
    return 0


def _run() -> int:
    """Refuse to rerun the completed recipe until reviewed source reauthorizes it."""
    # The disabled response is public, deterministic, and contains no configuration.
    _print_summary(
        {
            "passed": False,
            "reason": (
                "The reviewed minimal-pair ladder is complete. Another training "
                "attempt requires fresh user authorization and a new tested, "
                "reviewed, merged strategy."
            ),
            "status": "training_disabled",
        }
    )
    # A nonzero status prevents automation from treating the refusal as a run.
    return 2


def _evaluate(config: RunConfig, adapter: str) -> int:
    """Evaluate one existing adapter with the same held-out greedy protocol."""
    # Runtime imports stay scoped to the requested inference command.
    from fact_teaching.data import load_data_bundle, validate_data_bundle
    from fact_teaching.modeling import load_adapter_model, release_model
    from fact_teaching.reporting import (
        collect_runtime_provenance,
        write_standalone_report,
    )
    from fact_teaching.runtime import evaluate_model

    # A standalone run receives its own complete ignored operational log.
    run_id = f"{timestamp_id()}-standalone-evaluation"
    # Start with no model so cleanup also handles a failed load.
    bundle = None
    # Context management flushes every generation event before returning.
    with EventLogger(config.log_dir, run_id=run_id) as logger:
        try:
            # Log only the explicit adapter reference and sanitized configuration.
            logger.event(
                "standalone_evaluation_started",
                adapter=adapter,
                configuration=config.sanitized(),
            )
            # Dataset integrity is checked before any model generation.
            data = load_data_bundle(config.data_dir)
            # Exact counts and split isolation must match training evaluation.
            counts = validate_data_bundle(data)
            # The safe counts make the standalone protocol auditable.
            logger.event("dataset_validated", counts=counts)
            # Attach the adapter to a fresh copy of the exact pinned full base model.
            bundle = load_adapter_model(config, adapter, logger=logger)
            # Reuse the exact same greedy evaluator as baseline/post-training.
            result = evaluate_model(config, bundle, data, "standalone", logger)
            # Collect only allowlisted package and hardware provenance.
            provenance = collect_runtime_provenance(config)
            # Persist every complete prompt and raw generation for later review.
            report = write_standalone_report(
                config,
                result,
                adapter,
                logger,
                provenance=provenance,
            )
        finally:
            # Always return GPU memory even if generation or reporting fails.
            release_model(bundle)
    # Present only public/relative information in the final CLI summary.
    _print_summary(
        {
            "adapter": adapter,
            "summary": result.category_summary(),
            "json_report": report.json_path.name,
            "markdown_report": report.markdown_path.name,
        }
    )
    # A standalone evaluation is descriptive and has no baseline acceptance comparison.
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch exactly one public command."""
    # Parse either real process arguments or a unit-test supplied list.
    arguments = build_parser().parse_args(argv)
    # Reject the completed recipe before reading even public or credential state.
    if arguments.command == "run":
        return _run()
    # The repository root is intentionally the user's current working directory.
    config = _load_config(Path.cwd())
    # Each branch delegates to one high-level phase wrapper.
    if arguments.command == "preflight":
        return _preflight(config)
    # Argparse guarantees that evaluate carries a non-empty option string.
    if arguments.command == "evaluate":
        return _evaluate(config, arguments.adapter)
    # Required subparsers make this branch unreachable.
    raise AssertionError(f"Unhandled command: {arguments.command}")
