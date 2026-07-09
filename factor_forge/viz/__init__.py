"""Visualization utilities for Factor Forge."""

from factor_forge.viz.cum_returns import plot_cumulative_returns
from factor_forge.viz.decile_bars import plot_decile_returns
from factor_forge.viz.drawdown import plot_drawdown
from factor_forge.viz.factor_corr import plot_factor_correlation
from factor_forge.viz.ic_heatmap import plot_ic_heatmap
from factor_forge.viz.reports import generate_report

__all__ = [
    "generate_report",
    "plot_cumulative_returns",
    "plot_decile_returns",
    "plot_drawdown",
    "plot_factor_correlation",
    "plot_ic_heatmap",
]
