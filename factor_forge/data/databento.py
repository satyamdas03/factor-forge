"""Databento data source implementation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd

from factor_forge.data.base import DataSource

if TYPE_CHECKING:
    from factor_forge.data.cache import DataCache


class DatabentoDataSource(DataSource):
    """End-of-day price loader using the Databento Python client.

    Parameters
    ----------
    api_key:
        Databento API key. If None, reads from ``DATABENTO_API_KEY`` env var.
    cache:
        Optional ``DataCache`` for local Parquet caching.
    """

    name = "databento"

    def __init__(
        self,
        api_key: str | None = None,
        cache: DataCache | None = None,
    ) -> None:
        import databento as db

        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError("Databento API key is required")
        self.cache = cache
        self._client = db.Historical(self.api_key)

    def load_eod(
        self,
        symbols: Sequence[str],
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> pd.DataFrame:
        start_str = _to_str(start)
        end_str = _to_str(end)
        symbols_tuple = tuple(sorted(symbols))

        if self.cache and self.cache.exists(
            "databento_eod", symbols_tuple, start_str, end_str
        ):
            return self.cache.read("databento_eod", symbols_tuple, start_str, end_str)

        job = self._client.timeseries.get_range(
            dataset="XNAS.ITCH",
            schema="ohlcv-1d",
            symbols=list(symbols),
            start=start_str,
            end=end_str,
            stype_in="raw_symbol",
        )
        df = job.to_df()
        if df.empty:
            return _empty_eod_frame()

        df = df.reset_index()
        df = df.rename(
            columns={
                "ts_event": "date",
                "symbol": "symbol",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["symbol"] = df["symbol"].str.upper()
        df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
        df = df.set_index(["date", "symbol"]).sort_index()

        if self.cache:
            self.cache.write(
                df.reset_index(), "databento_eod", symbols_tuple, start_str, end_str
            )

        return df

    def load_universe(self, as_of: str | date | datetime | None = None) -> list[str]:
        # Databento does not expose a free symbol universe endpoint.
        return []

    def load_delistings(
        self,
        symbols: Sequence[str],
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        # Databento delisting dates are not available via the base historical API.
        return pd.DataFrame(columns=["symbol", "last_trade_date"])

    def close(self) -> None:
        self._client.close()  # type: ignore[attr-defined]


def _to_str(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _empty_eod_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
        pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
    )
