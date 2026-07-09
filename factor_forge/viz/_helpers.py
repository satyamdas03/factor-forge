"""Optional import helpers for visualization modules."""

from __future__ import annotations


def _ensure_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib; install factor-forge[viz]"
        ) from exc
