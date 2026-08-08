"""Global context: expose preflight, a completed-run guard, evaluation, and chat.

The command layer is intentionally thin: it parses user intent, loads only
allowlisted public settings plus a credential-presence bit from the project
`.env`, and delegates work to modular phase wrappers.
Sources:
- https://docs.python.org/3/library/argparse.html
- https://bbc2.github.io/python-dotenv/#load-configuration-without-altering-the-environment
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from training_facts_into_llms.archive_inventory import UploadMode
from training_facts_into_llms.config import RunConfig
from training_facts_into_llms.logging_utils import EventLogger, timestamp_id

# Only these public settings may move from `.env` or the shell into RunConfig.
PUBLIC_ENVIRONMENT_NAMES = (
    "HF_NAMESPACE",
    "ARTIFACT_DIR",
    "LOG_DIR",
    "REPORT_DIR",
    "TRACKIO_DIR",
    "TRACKIO_PROJECT",
)


def _public_dotenv_values(path: Path) -> dict[str, str]:
    """Parse only allowlisted public assignments without resolving the token value."""
    if not path.is_file():
        return {}
    public_lines: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            candidate = line.lstrip()
            if candidate.startswith("export "):
                candidate = candidate.removeprefix("export ").lstrip()
            name, separator, _ = candidate.partition("=")
            if separator and name.strip() in PUBLIC_ENVIRONMENT_NAMES:
                public_lines.append(line)
    parsed = dotenv_values(
        stream=io.StringIO("".join(public_lines)),
        interpolate=False,
    )
    return {
        name: str(value)
        for name, value in parsed.items()
        if name in PUBLIC_ENVIRONMENT_NAMES and value is not None
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the stable public command-line interface."""
    # A single top-level parser keeps help output compact.
    parser = argparse.ArgumentParser(
        prog="training-facts-into-llms",
        description=(
            "Reproduce or customize the Qwen3.5-0.8B single-fact LoRA study, "
            "inspect retained adapters, and archive reviewed artifacts."
        ),
    )
    # Commands are mandatory so an accidental invocation cannot start GPU work.
    commands = parser.add_subparsers(dest="command", required=True)
    # Preflight loads and inspects the model but never generates or trains.
    preflight = commands.add_parser(
        "preflight",
        help="Validate data, dependencies, CUDA/BF16, model, and LoRA targets.",
    )
    _add_experiment_arguments(preflight, include_name=False, include_upload=False)
    # One explicit experiment starts from one untouched copy of the pinned base.
    run = commands.add_parser(
        "run",
        help="Run one historical preset or a named typed customization.",
    )
    _add_experiment_arguments(run, include_name=True, include_upload=True)
    # Historical publication is separate from rerunning or mutating original evidence.
    publish_existing = commands.add_parser(
        "publish-existing",
        help="Audit or publish all retained historical adapter checkpoints.",
    )
    publish_existing.add_argument(
        "--all",
        action="store_true",
        required=True,
        help="Select the reviewed eight-run, thirteen-checkpoint inventory.",
    )
    publish_existing.add_argument(
        "--upload",
        choices=(UploadMode.OFF.value, UploadMode.ON.value),
        default=UploadMode.OFF.value,
        help="off validates/stages locally; on performs the reviewed Hub writes.",
    )
    publish_existing.add_argument(
        "--refresh-evidence",
        action="store_true",
        help=(
            "Perform the one-time reviewed evidence-dataset refresh; requires "
            "--upload on and never updates model repositories."
        ),
    )
    # Standalone evaluation works with either a local path or public Hub ID.
    evaluate = commands.add_parser(
        "evaluate",
        help="Run the fixed regression evaluation against an existing adapter.",
    )
    # Requiring an explicit adapter prevents accidental evaluation of the wrong model.
    evaluate.add_argument(
        "--adapter",
        required=True,
        help=(
            "Project-contained local adapter directory or public Hugging Face "
            "model repository ID."
        ),
    )
    evaluate.add_argument(
        "--checkpoint",
        type=_positive_step,
        help="Load checkpoints/checkpoint-N instead of the repository-root adapter.",
    )
    # Interactive chat can open a local picker or validate one explicit reference.
    chat = commands.add_parser(
        "chat",
        help="Run exploratory multi-turn inference against a saved LoRA adapter.",
    )
    # Omitting this option deliberately requires a numbered local checkpoint choice.
    chat.add_argument(
        "--adapter",
        help="Compatible local adapter directory or public Hugging Face repository ID.",
    )
    chat.add_argument(
        "--checkpoint",
        type=_positive_step,
        help="Load checkpoints/checkpoint-N instead of the repository-root adapter.",
    )
    # Return the parser for unit tests and the executable entry point.
    return parser


def _positive_step(value: str) -> int:
    """Parse one strictly positive Trainer checkpoint step for argparse."""
    step = int(value)
    if step <= 0:
        raise argparse.ArgumentTypeError("checkpoint must be a positive integer")
    return step


def _add_experiment_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_name: bool,
    include_upload: bool,
) -> None:
    """Add the shared preset, TOML overlay, and dotted override surface."""
    from training_facts_into_llms.experiments import EXPERIMENT_IDS

    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENT_IDS,
        help="Historical preset whose exact values form the run defaults.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Repository-contained partial TOML overlay applied after the preset.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="DOTTED.KEY=TOML_VALUE",
        help="Repeatable typed override; later occurrences win.",
    )
    if include_name:
        parser.add_argument(
            "--name",
            help="Required lowercase slug when scientific settings differ from preset.",
        )
    if include_upload:
        parser.add_argument(
            "--upload",
            choices=tuple(mode.value for mode in UploadMode),
            default=UploadMode.OFF.value,
            help="off, archive every completed run, or archive only accepted runs.",
        )


def _load_config(root: Path) -> RunConfig:
    """Load allowlisted settings without exporting the Hugging Face token."""
    # Resolve once before python-dotenv parses the project-local file.
    project_root = root.expanduser().resolve()
    # Remove an inherited token so model, Git, GitHub, and Trackio code cannot
    # receive it accidentally; secure boundaries reread the ignored file.
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
    # Filter by assignment name before dotenv parsing, so HF_TOKEN is never resolved.
    file_values = _public_dotenv_values(project_root / ".env")
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
    # RunConfig receives no secret and records no pre-upload credential state.
    public_mapping["HF_TOKEN"] = ""
    # RunConfig allowlists only operational non-secret values.
    return RunConfig.from_mapping(public_mapping, root=project_root)


def _print_summary(payload: dict[str, Any]) -> None:
    """Print one complete deterministic JSON summary."""
    # JSON output is machine-readable and does not rely on unsafe object reprs.
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def _preflight(config: RunConfig) -> int:
    """Run all non-generative hardware/model checks."""
    # Importing the heavy runtime stays below explicit command dispatch.
    from training_facts_into_llms.data import (
        load_experiment_data,
        validate_experiment_data,
    )
    from training_facts_into_llms.preflight import run_preflight
    from training_facts_into_llms.scoring import load_scoring_plugin

    scoring = config.experiment.scoring
    acceptance = config.experiment.acceptance
    _, plugin_source = load_scoring_plugin(
        config.root,
        scoring.plugin,
        scoring_options=scoring.options,
        acceptance_options=acceptance.options,
    )
    plugin_hash = hashlib.sha256(plugin_source.read_bytes()).hexdigest()

    # Preflight logs are operational and remain ignored by Git.
    with EventLogger(
        config.log_dir,
        run_id=f"{timestamp_id()}-preflight",
    ) as logger:
        # Configuration contains only allowlisted values and credential presence.
        logger.event("preflight_started", configuration=config.sanitized())
        logger.event(
            "scoring_plugin_validated",
            source=plugin_source.relative_to(config.root).as_posix(),
            sha256=plugin_hash,
        )
        # Static data integrity must pass before the large checkpoint is loaded.
        data = load_experiment_data(config.experiment)
        # Exact counts, schemas, unique IDs, and split isolation are verified.
        counts = validate_experiment_data(data, config.experiment)
        # The aggregate contains no prompt truncation or credential material.
        logger.event("dataset_validated", counts=counts)
        # The phase loads no generation prompt and performs no optimizer step.
        result = run_preflight(config, logger=logger)
    # Print a convenient final summary after the logger is flushed.
    _print_summary(result.to_dict())
    # A returned result represents all checks passing; failures raise.
    return 0


def _run(config: RunConfig) -> int:
    """Run exactly one resolved experiment and print its complete public outcome."""
    from training_facts_into_llms.pipeline import run_training_workflow

    try:
        outcome = run_training_workflow(config, config.experiment)
    except KeyboardInterrupt:
        _print_summary({"status": "interrupted", "exit_code": 130})
        return 130
    attempt = outcome.attempts[0]
    _print_summary(
        {
            "status": "completed",
            "experiment": config.experiment.experiment_id,
            "run_name": config.experiment.name,
            "scientific_hash": config.experiment.scientific_hash,
            "accepted": attempt.decision.passed,
            "acceptance_policy": getattr(
                attempt.decision,
                "policy_label",
                "legacy-canonical-policy",
            ),
            "adapter": str(attempt.adapter_path.relative_to(config.root)),
            "json_report": attempt.report.json_path.name,
            "markdown_report": attempt.report.markdown_path.name,
            "published_url": attempt.published_url,
        }
    )
    # A completed negative result is valid experimental evidence, not a CLI error.
    return 0


def _resolve_command_experiment(
    config: RunConfig,
    arguments: argparse.Namespace,
) -> RunConfig:
    """Resolve preset, partial TOML, and ordered dotted overrides before GPU work."""
    from training_facts_into_llms.experiments import resolve_experiment

    resolved = resolve_experiment(
        config.root,
        arguments.experiment,
        custom_config=arguments.config,
        overrides=tuple(arguments.overrides),
        name=getattr(arguments, "name", None),
        require_custom_name=arguments.command == "run",
    )
    return config.with_experiment(
        resolved,
        upload_mode=getattr(arguments, "upload", UploadMode.OFF.value),
    )


def _publish_existing(
    config: RunConfig,
    upload_mode: str,
    *,
    refresh_evidence: bool = False,
) -> int:
    """Stage or publish the reviewed historical eight-run archive."""
    from training_facts_into_llms.archive_publishing import (
        publish_historical_archive,
        refresh_historical_evidence,
    )

    run_id = f"{timestamp_id()}-publish-existing"
    with EventLogger(config.log_dir, run_id=run_id) as logger:
        if refresh_evidence:
            logger.event("historical_evidence_refresh_started")
            payload = refresh_historical_evidence(config).to_dict()
            # This narrow receipt omits staging paths, credentials, and raw Hub objects.
            logger.event("historical_evidence_refresh_completed", receipt=payload)
            _print_summary(payload)
            return 0
        logger.event(
            "historical_archive_started",
            upload_mode=upload_mode,
        )
        result = publish_historical_archive(
            config,
            upload_mode=UploadMode(upload_mode),
        )
        payload = result.to_dict()
        # The receipt retains every local/remote hash and complete smoke generation.
        logger.event("historical_archive_completed", receipt=payload)
    _print_summary(payload)
    return 0


def _evaluate(config: RunConfig, adapter: str, checkpoint: int | None = None) -> int:
    """Evaluate one existing adapter with the fixed greedy regression protocol."""
    # Runtime imports stay scoped to the requested inference command.
    from training_facts_into_llms.data import load_data_bundle, validate_data_bundle
    from training_facts_into_llms.modeling import load_adapter_model, release_model
    from training_facts_into_llms.reporting import (
        _public_adapter_reference,
        collect_runtime_provenance,
        write_standalone_report,
    )
    from training_facts_into_llms.runtime import evaluate_model

    # Validate and relativize the public adapter reference before creating logs or
    # allocating a model. Hub IDs retain their normal owner/repository spelling.
    adapter_reference = _public_adapter_reference(config, adapter)
    display_reference = (
        f"{adapter_reference}@checkpoint-{checkpoint}"
        if checkpoint is not None
        else adapter_reference
    )
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
                adapter=display_reference,
                configuration=config.sanitized(),
            )
            # Dataset integrity is checked before any model generation.
            data = load_data_bundle(config.data_dir)
            # Exact counts and split isolation must match training evaluation.
            counts = validate_data_bundle(data)
            # The safe counts make the standalone protocol auditable.
            logger.event("dataset_validated", counts=counts)
            # Attach the adapter to a fresh copy of the exact pinned full base model.
            bundle = load_adapter_model(
                config,
                adapter,
                logger=logger,
                adapter_log_reference=display_reference,
                subfolder=(
                    f"checkpoints/checkpoint-{checkpoint}"
                    if checkpoint is not None
                    else None
                ),
            )
            # Reuse the exact same greedy evaluator as baseline/post-training.
            result = evaluate_model(config, bundle, data, "standalone", logger)
            # Collect only allowlisted package and hardware provenance.
            provenance = collect_runtime_provenance(config)
            # Persist every complete prompt and returned post-strip response.
            report = write_standalone_report(
                config,
                result,
                display_reference,
                logger,
                provenance=provenance,
            )
        finally:
            # Always return GPU memory even if generation or reporting fails.
            release_model(bundle)
    # Present only public/relative information in the final CLI summary.
    _print_summary(
        {
            "adapter": display_reference,
            "summary": result.category_summary(),
            "json_report": report.json_path.name,
            "markdown_report": report.markdown_path.name,
        }
    )
    # A standalone evaluation is descriptive and has no baseline acceptance comparison.
    return 0


def _chat(
    config: RunConfig,
    adapter: str | None,
    checkpoint: int | None = None,
) -> int:
    """Run one logged exploratory chat without scoring or training the adapter."""
    # The focused wrapper owns selection, model lifecycle, history, and log events.
    from training_facts_into_llms.chat import run_interactive_chat

    # Return its conventional normal, validation, or interruption status unchanged.
    return run_interactive_chat(config, adapter, checkpoint=checkpoint)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch exactly one public command."""
    # Parse either real process arguments or a unit-test supplied list.
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if (
        arguments.command == "publish-existing"
        and arguments.refresh_evidence
        and arguments.upload != UploadMode.ON.value
    ):
        parser.error("--refresh-evidence requires --upload on")
    # The repository root is intentionally the user's current working directory.
    config = _load_config(Path.cwd())
    # Training and preflight resolve the exact scientific configuration first.
    if arguments.command in {"preflight", "run"}:
        config = _resolve_command_experiment(config, arguments)
    # Each branch delegates to one high-level phase wrapper.
    if arguments.command == "preflight":
        return _preflight(config)
    if arguments.command == "run":
        return _run(config)
    if arguments.command == "publish-existing":
        return _publish_existing(
            config,
            arguments.upload,
            refresh_evidence=arguments.refresh_evidence,
        )
    # Argparse guarantees that evaluate carries a non-empty option string.
    if arguments.command == "evaluate":
        return _evaluate(config, arguments.adapter, arguments.checkpoint)
    # Chat never calls the training pipeline or tracked evaluation reporting path.
    if arguments.command == "chat":
        return _chat(config, arguments.adapter, arguments.checkpoint)
    # Required subparsers make this branch unreachable.
    raise AssertionError(f"Unhandled command: {arguments.command}")
