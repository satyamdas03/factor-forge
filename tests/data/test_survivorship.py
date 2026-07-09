"""Tests for survivorship-bias-free universe construction."""

import pandas as pd
import pytest

from factor_forge.data.loader import generate_synthetic_prices
from factor_forge.data.survivorship import build_universe, filter_alive


@pytest.fixture
def prices() -> pd.DataFrame:
    return generate_synthetic_prices(["A", "B", "C"], "2020-01-01", "2021-12-31")


def test_filter_alive_keeps_active_tickers(prices: pd.DataFrame) -> None:
    subset = filter_alive(prices, None, "2021-12-31", min_history_days=100)
    symbols = subset.index.get_level_values("symbol").unique().tolist()
    assert set(symbols) == {"A", "B", "C"}


def test_filter_alive_excludes_delisted(prices: pd.DataFrame) -> None:
    delistings = pd.DataFrame(
        {"symbol": ["B"], "last_trade_date": [pd.Timestamp("2021-06-01")]}
    )
    subset = filter_alive(prices, delistings, "2021-12-31", min_history_days=100)
    symbols = subset.index.get_level_values("symbol").unique().tolist()
    assert "B" not in symbols
    assert {"A", "C"}.issubset(set(symbols))


def test_build_universe_top_n_by_volume(prices: pd.DataFrame) -> None:
    universe = build_universe(prices, None, "2021-12-31", min_history_days=100, top_n=2)
    assert len(universe) == 2
    assert len(set(universe)) == 2


def test_build_universe_requires_min_history(prices: pd.DataFrame) -> None:
    universe = build_universe(prices, None, "2020-02-01", min_history_days=500)
    assert universe == []
