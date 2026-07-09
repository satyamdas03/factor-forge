"""Built-in cross-sectional factor implementations.

All factors are defined as pure functions over a panel DataFrame indexed by
``(date, symbol)`` and return a score Series with the same index. Higher scores
indicate the desired long side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_forge.factors.base import Factor, FactorCategory
from factor_forge.factors.preprocessor import (
    ensure_returns,
    rolling_beta,
)


def _price_panel() -> pd.DataFrame:
    """Placeholder for typing clarity; functions operate on panel DataFrames."""
    return pd.DataFrame()


def _momentum_12_1(df: pd.DataFrame) -> pd.Series:
    """12-month return excluding the most recent month."""
    df = ensure_returns(df)
    mom = (
        1 + df.groupby("symbol")["return"].rolling(252, min_periods=220).mean()
    ).reset_index(level=0, drop=True)
    recent = (
        1 + df.groupby("symbol")["return"].rolling(21, min_periods=18).mean()
    ).reset_index(level=0, drop=True)
    return mom - recent


def _momentum_52_high(df: pd.DataFrame) -> pd.Series:
    """Current price divided by trailing 52-week high."""
    high_52 = (
        df.groupby("symbol")["close"]
        .rolling(252, min_periods=220)
        .max()
        .reset_index(level=0, drop=True)
    )
    return df["close"] / high_52


def _residual_momentum(df: pd.DataFrame) -> pd.Series:
    """Residual momentum: 12-1 return residualized against market return."""
    df = ensure_returns(df)
    df["market_return"] = df.groupby(level="date")["return"].transform("mean")
    beta = rolling_beta(df, window=63)
    residual = df["return"] - beta * df["market_return"]
    return (
        1 + residual.groupby("symbol").rolling(252, min_periods=220).mean()
    ).reset_index(level=0, drop=True) - 1


def _pe_ratio(df: pd.DataFrame) -> pd.Series:
    """Price-to-earnings; lower P/E = higher value = higher score."""
    if "earnings" in df.columns:
        return -df["close"] / df["earnings"].replace(0, np.nan)
    raise ValueError("pe_ratio requires an 'earnings' column")


def _pb_ratio(df: pd.DataFrame) -> pd.Series:
    """Price-to-book; lower P/B = higher value = higher score."""
    if "book_value" in df.columns:
        return -df["close"] / df["book_value"].replace(0, np.nan)
    raise ValueError("pb_ratio requires a 'book_value' column")


def _roe(df: pd.DataFrame) -> pd.Series:
    """Return on equity."""
    if "net_income" in df.columns and "equity" in df.columns:
        return df["net_income"] / df["equity"].replace(0, np.nan)
    raise ValueError("roe requires 'net_income' and 'equity' columns")


def _gross_margin(df: pd.DataFrame) -> pd.Series:
    """Gross profit / revenue."""
    if "gross_profit" in df.columns and "revenue" in df.columns:
        return df["gross_profit"] / df["revenue"].replace(0, np.nan)
    raise ValueError("gross_margin requires 'gross_profit' and 'revenue' columns")


def _accruals(df: pd.DataFrame) -> pd.Series:
    """Accruals anomaly proxy: net income minus free cash flow scaled by assets."""
    if (
        "net_income" in df.columns
        and "cash_flow" in df.columns
        and "assets" in df.columns
    ):
        return -(df["net_income"] - df["cash_flow"]) / df["assets"].replace(0, np.nan)
    raise ValueError(
        "accruals requires 'net_income', 'cash_flow', and 'assets' columns"
    )


def _idiosyncratic_volatility(df: pd.DataFrame) -> pd.Series:
    """Low idiosyncratic volatility factor; lower vol = higher score."""
    df = ensure_returns(df)
    df["market_return"] = df.groupby(level="date")["return"].transform("mean")
    beta = rolling_beta(df, window=63)
    residual = df["return"] - beta * df["market_return"]
    vol = (
        residual.groupby("symbol")
        .rolling(63, min_periods=42)
        .std()
        .reset_index(level=0, drop=True)
    )
    return -vol


def _beta_factor(df: pd.DataFrame) -> pd.Series:
    """Low beta factor; lower beta = higher score."""
    df = ensure_returns(df)
    df["market_return"] = df.groupby(level="date")["return"].transform("mean")
    beta = rolling_beta(df, window=63)
    return -beta


def _max_drawdown(df: pd.DataFrame) -> pd.Series:
    """Maximum drawdown over trailing 252 days; smaller drawdown = higher score."""

    def _mdd(sub: pd.Series) -> pd.Series:
        cummax = sub.cummax()
        drawdown = (sub - cummax) / cummax
        return drawdown.rolling(252, min_periods=126).min()

    mdd = df.groupby("symbol")["close"].apply(_mdd).reset_index(level=0, drop=True)
    return -mdd


def _asset_growth(df: pd.DataFrame) -> pd.Series:
    """Year-over-year asset growth; lower growth = higher score (investment factor)."""
    if "assets" in df.columns:
        growth = df.groupby("symbol")["assets"].pct_change(periods=252)
        return -growth
    raise ValueError("asset_growth requires an 'assets' column")


def _capex_growth(df: pd.DataFrame) -> pd.Series:
    """Year-over-year capex growth; lower growth = higher score."""
    if "capex" in df.columns:
        growth = df.groupby("symbol")["capex"].pct_change(periods=252)
        return -growth
    raise ValueError("capex_growth requires a 'capex' column")


def _gross_profits_to_assets(df: pd.DataFrame) -> pd.Series:
    """Gross profits to total assets."""
    if "gross_profit" in df.columns and "assets" in df.columns:
        return df["gross_profit"] / df["assets"].replace(0, np.nan)
    raise ValueError(
        "gross_profits_to_assets requires 'gross_profit' and 'assets' columns"
    )


def _price_momentum_signal(
    df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """Generic momentum signal on closing prices."""
    shifted = df.groupby("symbol")["close"].shift(skip)
    return df["close"] / shifted - 1


BUILTIN_FACTORS: tuple[Factor, ...] = (
    Factor("momentum_12_1", FactorCategory.MOMENTUM, ["close"], _momentum_12_1),
    Factor("momentum_52_high", FactorCategory.MOMENTUM, ["close"], _momentum_52_high),
    Factor("residual_momentum", FactorCategory.MOMENTUM, ["close"], _residual_momentum),
    Factor("pe_ratio", FactorCategory.VALUE, ["close", "earnings"], _pe_ratio),
    Factor("pb_ratio", FactorCategory.VALUE, ["close", "book_value"], _pb_ratio),
    Factor("roe", FactorCategory.QUALITY, ["net_income", "equity"], _roe),
    Factor(
        "gross_margin",
        FactorCategory.QUALITY,
        ["gross_profit", "revenue"],
        _gross_margin,
    ),
    Factor(
        "accruals",
        FactorCategory.QUALITY,
        ["net_income", "cash_flow", "assets"],
        _accruals,
    ),
    Factor(
        "idiosyncratic_volatility",
        FactorCategory.LOW_VOLATILITY,
        ["close"],
        _idiosyncratic_volatility,
    ),
    Factor("beta", FactorCategory.LOW_VOLATILITY, ["close"], _beta_factor),
    Factor("max_drawdown", FactorCategory.LOW_VOLATILITY, ["close"], _max_drawdown),
    Factor("asset_growth", FactorCategory.INVESTMENT, ["assets"], _asset_growth),
    Factor("capex_growth", FactorCategory.INVESTMENT, ["capex"], _capex_growth),
    Factor(
        "gross_profits_to_assets",
        FactorCategory.PROFITABILITY,
        ["gross_profit", "assets"],
        _gross_profits_to_assets,
    ),
)
