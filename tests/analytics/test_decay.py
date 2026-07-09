"""Tests for factor decay analytics."""

import numpy as np
import pandas as pd

from factor_forge.analytics.decay import factor_autocorrelation, factor_half_life


def test_factor_autocorrelation_perfect_persistence() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    symbols = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    scores = pd.Series(np.tile([1, 2, 3], 10), index=idx)
    ac = factor_autocorrelation(scores, lag=1)
    assert (ac == 1.0).all()


def test_factor_half_life_stable_factor() -> None:
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    symbols = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.normal(size=len(idx)), index=idx)
    hl = factor_half_life(scores, max_lag=5)
    assert 1 <= hl <= 5
