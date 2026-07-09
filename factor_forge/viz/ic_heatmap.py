"""IC heatmap plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from factor_forge.viz._helpers import _ensure_matplotlib


def plot_ic_heatmap(ic_df: pd.DataFrame) -> Figure:
    """Plot a heatmap of rolling ICs across factors."""
    _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(ic_df.T, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    ax.set_yticks(range(len(ic_df.columns)))
    ax.set_yticklabels(ic_df.columns)
    ax.set_xticks(range(0, len(ic_df.index), max(1, len(ic_df.index) // 6)))
    ax.set_xticklabels(
        [str(d)[:10] for d in ic_df.index[:: max(1, len(ic_df.index) // 6)]],
        rotation=45,
    )
    ax.set_title("Rolling Information Coefficient")
    fig.colorbar(im, ax=ax)
    return fig
