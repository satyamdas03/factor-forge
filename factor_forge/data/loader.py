"""High-level data loader that merges sources, adjusts, and caches."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from factor_forge.data.base import DataSource
from factor_forge.data.cache import DataCache
from factor_forge.data.databento import DatabentoDataSource
from factor_forge.data.polygon import PolygonDataSource
from factor_forge.data.survivorship import build_universe

if TYPE_CHECKING:
    pass


class DataLoader:
    """Unified entry point for loading clean, adjusted, point-in-time data.

    Parameters
    ----------
    source:
        One of ``polygon``, ``databento``, or a custom ``DataSource`` instance.
    api_key:
        API key for the selected source (optional if set in env).
    cache_dir:
        Directory for local Parquet cache. Defaults to ``~/.factor_forge/cache``.
    """

    def __init__(
        self,
        source: str | DataSource = "polygon",
        api_key: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.cache = DataCache(cache_dir)
        if isinstance(source, DataSource):
            self.source = source
        elif source == "polygon":
            self.source = PolygonDataSource(api_key=api_key, cache=self.cache)
        elif source == "databento":
            self.source = DatabentoDataSource(api_key=api_key, cache=self.cache)
        else:
            raise ValueError(f"Unknown data source: {source}")

    def load_eod(
        self,
        symbols: Sequence[str],
        start: str | date | datetime,
        end: str | date | datetime,
        adjust: bool = True,
    ) -> pd.DataFrame:
        """Load end-of-day prices and apply split/dividend adjustments if requested.

        Returns a long-form DataFrame indexed by ``(date, symbol)``.
        """
        df = self.source.load_eod(symbols, start, end)
        if df.empty:
            return df

        if adjust:
            df = self._adjust_splits_dividends(df)

        return df.sort_index()

    def load_universe(
        self,
        prices: pd.DataFrame | None = None,
        as_of: str | date | datetime | None = None,
        min_history_days: int = 252,
        top_n: int | None = None,
        by_volume: bool = True,
    ) -> list[str]:
        """Build a survivorship-bias-free universe.

        If ``prices`` is provided, use it directly. Otherwise load the active
        symbol list from the source and fetch their history.
        """
        as_of = as_of or datetime.now().date()
        if prices is not None:
            delistings = self.source.load_delistings([], prices.index[0][0], as_of)
            return build_universe(
                prices, delistings, as_of, min_history_days, top_n, by_volume
            )

        symbols = self.source.load_universe(as_of)
        start = _to_date(as_of) - pd.Timedelta(days=min_history_days + 180)
        prices = self.load_eod(symbols, start, as_of)
        delistings = self.source.load_delistings(symbols, start, as_of)
        return build_universe(
            prices, delistings, as_of, min_history_days, top_n, by_volume
        )

    def _adjust_splits_dividends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cumulative split and dividend adjustments to OHLCV.

        This is a best-effort adjustment when the upstream API does not already
        provide fully adjusted prices. Polygon returns split/dividend-adjusted
        prices by default with ``adjusted=true``.
        """
        # Placeholder for explicit adjustment logic. Upstream sources currently
        # return adjusted closes, so we pass through.
        return df

    def close(self) -> None:
        self.source.close()


def _to_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def generate_synthetic_prices(
    symbols: Sequence[str],
    start: str | date,
    end: str | date,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV prices for offline tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="B")
    records: list[dict[str, Any]] = []
    for symbol in symbols:
        drift = rng.uniform(-0.0002, 0.0008)
        vol = rng.uniform(0.01, 0.03)
        returns = rng.normal(drift, vol, size=len(dates))
        price = 100.0 * np.exp(np.cumsum(returns))
        volume = rng.integers(1_000_000, 10_000_000, size=len(dates))
        for i, d in enumerate(dates):
            records.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "open": price[i] * (1 + rng.uniform(-0.005, 0.005)),
                    "high": price[i] * (1 + rng.uniform(0.0, 0.01)),
                    "low": price[i] * (1 + rng.uniform(-0.01, 0.0)),
                    "close": price[i],
                    "volume": int(volume[i]),
                }
            )
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index(["date", "symbol"]).sort_index()
    return df
