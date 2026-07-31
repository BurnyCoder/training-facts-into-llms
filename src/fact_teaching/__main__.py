"""Global context: support `python -m fact_teaching` as a CLI equivalent."""

from fact_teaching.cli import main

# Standard module execution delegates to the same console-script entry point.
if __name__ == "__main__":
    # Propagate the CLI's conventional integer process status.
    raise SystemExit(main())
