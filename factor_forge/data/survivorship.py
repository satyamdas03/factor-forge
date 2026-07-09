"""Survivorship-bias-free universe construction utilities."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import cast

import pandas as pd


def _to_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def filter_alive(
    prices: pd.DataFrame,
    delistings: pd.DataFrame | None,
    as_of: str | date | datetime,
    min_history_days: int = 252,
) -> pd.DataFrame:
    """Return rows of ``prices`` for tickers that are still alive as of ``as_of``.

    A ticker is considered alive if it has at least one price observation on or
    before ``as_of`` and no delisting date prior to ``as_of``. It must also have
    at least ``min_history_days`` of trading history before ``as_of``.
    """
    as_of_date = _to_date(as_of)
    idx = prices.index
    if isinstance(idx, pd.MultiIndex):
        dates = idx.get_level_values("date")
        symbols = idx.get_level_values("symbol")
    else:
        dates = pd.to_datetime(prices["date"]).dt.date
        symbols = prices["symbol"]

    alive_mask = dates <= pd.Timestamp(as_of_date)
    if delistings is not None and not delistings.empty:
        delist_map = delistings.set_index("symbol")["last_trade_date"]
        for sym, last_date in delist_map.items():
            last = _to_date(last_date)
            if last < as_of_date:
                alive_mask = alive_mask & (symbols != sym)

    subset = prices[alive_mask]
    symbol_counts = subset.groupby("symbol").size()
    eligible = symbol_counts[symbol_counts >= min_history_days].index
    return subset[subset.index.get_level_values("symbol").isin(eligible)]


def build_universe(
    prices: pd.DataFrame,
    delistings: pd.DataFrame | None,
    as_of: str | date | datetime,
    min_history_days: int = 252,
    top_n: int | None = None,
    by_volume: bool = True,
) -> list[str]:
    """Build an eligible universe as of a given date.

    Parameters
    ----------
    prices:
        Long-form price DataFrame with ``(date, symbol)`` index.
    delistings:
        Optional delisting table with ``symbol`` and ``last_trade_date``.
    as_of:
        Universe construction date.
    min_history_days:
        Minimum number of trading days required before ``as_of``.
    top_n:
        If provided, return the top-N symbols by average dollar volume.
    by_volume:
        Use dollar volume for ranking when ``top_n`` is set.
    """
    subset = filter_alive(prices, delistings, as_of, min_history_days)
    if subset.empty:
        return []

    if by_volume and "volume" in subset.columns and "close" in subset.columns:
        subset = subset.copy()
        subset["dollar_volume"] = subset["close"] * subset["volume"]
        volume_mean = subset.groupby("symbol")["dollar_volume"].mean()
        ranked = volume_mean.sort_values(ascending=False)
        if top_n:
            ranked = ranked.head(top_n)
        return cast(list[str], ranked.index.tolist())

    symbols = cast(list[str], subset.index.get_level_values("symbol").unique().tolist())
    if top_n:
        return symbols[:top_n]
    return symbols


def trim_to_history(
    symbols: Sequence[str],
    prices: pd.DataFrame,
    min_history_days: int = 126,
) -> list[str]:
    """Drop symbols that do not have sufficient history in ``prices``."""
    counts = prices.groupby("symbol").size()
    eligible = counts[counts >= min_history_days].index.tolist()
    return [s for s in symbols if s in eligible]
