"""Factor decay and turnover analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def factor_autocorrelation(scores: pd.Series, lag: int = 1) -> pd.Series:
    """Compute rank autocorrelation of factor scores at a given lag."""
    ranks = scores.groupby(level="date").rank(pct=False, method="average")
    shifted = ranks.groupby("symbol").shift(lag)
    df = pd.DataFrame({"rank": ranks, "lag_rank": shifted}).dropna()

    def _corr(sub: pd.DataFrame) -> float:
        if len(sub) < 3:
            return np.nan
        return float(sub["rank"].corr(sub["lag_rank"], method="spearman"))

    return df.groupby(level="date").apply(_corr).dropna()


def factor_half_life(scores: pd.Series, max_lag: int = 12) -> float:
    """Estimate factor rank-IC half-life in periods by exponential fit.

    Returns the lag at which autocorrelation drops to 0.5. If autocorrelation
    never drops below 0.5, returns ``max_lag``.
    """
    autocorrs = []
    for lag in range(1, max_lag + 1):
        ac = factor_autocorrelation(scores, lag=lag).mean()
        autocorrs.append((lag, ac))
        if ac <= 0.5:
            return float(lag)
    return float(max_lag)


def portfolio_turnover(trades: pd.DataFrame) -> pd.Series:
    """Aggregate turnover per rebalance date from a trade log."""
    if trades.empty:
        return pd.Series(dtype=float)
    trades = trades.copy()
    trades["turnover"] = trades["turnover"]
    return trades.groupby("date")["turnover"].sum()
