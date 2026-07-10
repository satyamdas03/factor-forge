"""Streamlit dashboard launcher for Factor Forge."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _streamlit_command() -> list[str]:
    """Return the command used to launch the Streamlit dashboard."""
    app_path = Path(__file__).with_name("app.py").resolve()
    return [sys.executable, "-m", "streamlit", "run", str(app_path)]


def launch() -> int:
    """Launch the Factor Forge Streamlit dashboard.

    Returns the exit code of the Streamlit process.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Dashboard requires streamlit; install factor-forge[dashboard]"
        ) from exc
    cmd = _streamlit_command()
    return subprocess.call(cmd)


__all__ = ["launch", "_streamlit_command"]
