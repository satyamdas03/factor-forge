"""Helpers for transforming raw price panels into factor-ready inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def ensure_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add daily log and simple return columns per symbol."""
    df = df.copy()
    if "return" not in df.columns:
        df["return"] = df.groupby("symbol")[price_col].pct_change()
    if "log_return" not in df.columns:
        df["log_return"] = np.log(
            df[price_col] / df.groupby("symbol")[price_col].shift(1)
        )
    return df


def rolling_mean_return(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
) -> pd.Series:
    """Trailing mean daily return per symbol."""
    returns = df.groupby("symbol")[price_col].pct_change()
    return (
        returns.groupby("symbol")
        .rolling(window, min_periods=window // 2)
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_std_return(
    df: pd.DataFrame,
    window: int,
    price_col: str = "close",
) -> pd.Series:
    """Trailing standard deviation of daily returns per symbol."""
    returns = df.groupby("symbol")[price_col].pct_change()
    return (
        returns.groupby("symbol")
        .rolling(window, min_periods=window // 2)
        .std()
        .reset_index(level=0, drop=True)
    )


def rolling_beta(
    df: pd.DataFrame,
    window: int = 63,
    market_col: str = "market_return",
    return_col: str = "return",
) -> pd.Series:
    """Rolling CAPM beta against ``market_col`` per symbol."""
    market = (
        df[market_col]
        if market_col in df.columns
        else df.groupby(level="date")[return_col].transform("mean")
    )

    def _beta(sub: pd.DataFrame) -> pd.Series:
        x = market.loc[sub.index].fillna(0)
        y = sub[return_col].fillna(0)
        cov = y.rolling(window, min_periods=window // 2).cov(x)
        var = x.rolling(window, min_periods=window // 2).var()
        return cov / var.replace(0, np.nan)

    return df.groupby("symbol", group_keys=False).apply(_beta)


def cross_sectional_rank(
    series: pd.Series,
    method: str = "average",
    ascending: bool = True,
) -> pd.Series:
    """Rank values cross-sectionally within each date."""
    ranked = series.groupby(level="date").rank(
        method=method, pct=False, ascending=ascending
    )
    return ranked


def winsorize(
    series: pd.Series, limits: tuple[float, float] = (0.01, 0.01)
) -> pd.Series:
    """Winsorize a Series at the given quantile limits."""
    lower = series.quantile(limits[0])
    upper = series.quantile(1 - limits[1])
    return series.clip(lower=lower, upper=upper)
