"""Cumulative return plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from factor_forge.viz._helpers import _ensure_matplotlib


def plot_cumulative_returns(returns_dict: dict[str, pd.Series]) -> Figure:
    """Plot cumulative returns for multiple return series."""
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, returns in returns_dict.items():
        cum = (1 + returns).cumprod()
        ax.plot(cum.index, cum, label=label)
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True)
    return fig
