"""Factor correlation heatmap."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from factor_forge.viz._helpers import _ensure_matplotlib


def plot_factor_correlation(corr: pd.DataFrame) -> Figure:
    """Plot a Spearman correlation heatmap of factor scores."""
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black"
            )
    fig.colorbar(cax, ax=ax)
    ax.set_title("Factor Rank Correlation Matrix")
    return fig
