"""Command-line entry point for launching the Streamlit dashboard."""

from __future__ import annotations

import sys

from factor_forge.dashboard import launch


def main(argv: list[str] | None = None) -> int:
    """Launch the Factor Forge dashboard."""
    try:
        return launch()
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
