"""Tests for analytics metrics."""

import numpy as np
import pandas as pd
import pytest

from factor_forge.analytics.factor_ic import compute_ic, compute_ic_summary
from factor_forge.analytics.metrics import compute_metrics


def test_compute_metrics_known_values() -> None:
    returns = pd.Series(
        [0.01, -0.01, 0.02, -0.005, 0.015],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    metrics = compute_metrics(returns)
    assert metrics["total_return"] == pytest.approx((1 + returns).prod() - 1, rel=1e-6)
    assert metrics["annualized_volatility"] > 0
    assert metrics["max_drawdown"] <= 0


def test_compute_ic_perfect_correlation() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    symbols = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    scores = pd.Series(np.arange(20), index=idx)
    fwd = pd.Series(np.arange(20) * 2, index=idx)
    ic = compute_ic(scores, fwd, method="pearson")
    assert np.allclose(ic, 1.0)


def test_compute_ic_summary() -> None:
    ic = pd.Series([0.1, 0.05, 0.08, -0.02, 0.12])
    summary = compute_ic_summary(ic)
    assert "mean_ic" in summary
    assert "icir" in summary
