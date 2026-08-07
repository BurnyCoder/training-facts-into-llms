"""Global context: support `python -m training_facts_into_llms` as a CLI equivalent."""

from training_facts_into_llms.cli import main

# Standard module execution delegates to the same console-script entry point.
if __name__ == "__main__":
    # Propagate the CLI's conventional integer process status.
    raise SystemExit(main())
