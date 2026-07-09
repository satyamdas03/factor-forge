"""Regime-dependent performance analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


def classify_market_regime(
    prices: pd.Series,
    long_ma: int = 200,
    short_ma: int = 50,
) -> pd.Series:
    """Classify each date as bull, bear, or neutral based on moving-average cross.

    Parameters
    ----------
    prices:
        Market index price series with a DatetimeIndex.
    """
    long = prices.rolling(long_ma, min_periods=long_ma // 2).mean()
    short = prices.rolling(short_ma, min_periods=short_ma // 2).mean()
    regime = pd.Series(index=prices.index, dtype="object")
    regime[short > long] = "bull"
    regime[short < long] = "bear"
    regime[short.isna() | long.isna()] = "neutral"
    regime[short == long] = "neutral"
    return regime


def split_by_regime(
    returns: pd.Series,
    regimes: pd.Series,
) -> dict[str, pd.Series]:
    """Split returns into groups by regime label."""
    aligned_regimes = regimes.reindex(returns.index).ffill().fillna("neutral")
    result: dict[str, pd.Series] = {}
    for label, group_returns in returns.groupby(aligned_regimes):
        result[str(label)] = group_returns
    return result
