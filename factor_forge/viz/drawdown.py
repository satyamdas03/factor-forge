"""Underwater curve plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from factor_forge.viz._helpers import _ensure_matplotlib


def plot_drawdown(returns: pd.Series) -> Figure:
    """Plot drawdown (underwater) curve."""
    _ensure_matplotlib()

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown, color="red")
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    return fig
