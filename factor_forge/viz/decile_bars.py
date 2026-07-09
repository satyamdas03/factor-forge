"""Decile return bar chart."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from factor_forge.viz._helpers import _ensure_matplotlib


def plot_decile_returns(decile_returns: pd.Series) -> Figure:
    """Bar plot of average return per decile.

    Parameters
    ----------
    decile_returns:
        Series indexed by decile label with average returns.
    """
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    decile_returns.plot(kind="bar", ax=ax)
    ax.set_title("Average Return by Decile")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Average Return")
    ax.grid(True, axis="y")
    return fig
