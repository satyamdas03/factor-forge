"""Tests for the unified data loader."""

import pandas as pd
import pytest

from factor_forge.data.base import DataSource
from factor_forge.data.loader import DataLoader, generate_synthetic_prices


class _FakeSource(DataSource):
    name = "fake"

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def load_eod(self, symbols, start, end) -> pd.DataFrame:
        return self.df.loc[self.df.index.get_level_values("symbol").isin(symbols)]

    def load_universe(self, as_of=None) -> list[str]:
        return self.df.index.get_level_values("symbol").unique().tolist()

    def load_delistings(self, symbols, start, end) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "last_trade_date"])


def test_loader_load_eod() -> None:
    df = generate_synthetic_prices(["A", "B"], "2020-01-01", "2020-12-31")
    loader = DataLoader(source=_FakeSource(df))
    loaded = loader.load_eod(["A"], "2020-01-01", "2020-12-31")
    assert set(loaded.index.get_level_values("symbol")) == {"A"}
    assert "close" in loaded.columns


def test_loader_load_universe() -> None:
    df = generate_synthetic_prices(["A", "B", "C"], "2020-01-01", "2020-12-31")
    loader = DataLoader(source=_FakeSource(df))
    universe = loader.load_universe(prices=df, as_of="2020-12-31", top_n=2)
    assert len(universe) == 2


def test_loader_raises_on_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unknown data source"):
        DataLoader(source="unknown")
