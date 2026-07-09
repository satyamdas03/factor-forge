"""Tests for visualization helpers."""

import numpy as np
import pandas as pd

from factor_forge.viz.cum_returns import plot_cumulative_returns
from factor_forge.viz.drawdown import plot_drawdown
from factor_forge.viz.factor_corr import plot_factor_correlation


def test_plot_cumulative_returns() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=50), index=idx)
    fig = plot_cumulative_returns({"strategy": returns})
    assert fig is not None


def test_plot_drawdown() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=50), index=idx)
    fig = plot_drawdown(returns)
    assert fig is not None


def test_plot_factor_correlation() -> None:
    corr = pd.DataFrame(np.eye(3), columns=["a", "b", "c"], index=["a", "b", "c"])
    fig = plot_factor_correlation(corr)
    assert fig is not None
