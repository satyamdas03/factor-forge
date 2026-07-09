"""Polygon.io data source implementation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import requests

from factor_forge.data.base import DataSource

if TYPE_CHECKING:
    from factor_forge.data.cache import DataCache


class PolygonDataSource(DataSource):
    """End-of-day price loader using the Polygon.io REST API.

    Parameters
    ----------
    api_key:
        Polygon API key. If None, reads from ``POLYGON_API_KEY`` env var.
    cache:
        Optional ``DataCache`` for local Parquet caching.
    """

    name = "polygon"
    _base_url = "https://api.polygon.io/v2"
    _reference_url = "https://api.polygon.io/v3/reference"

    def __init__(
        self,
        api_key: str | None = None,
        cache: DataCache | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key is required")
        self.cache = cache
        self._session = requests.Session()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        params["apiKey"] = self.api_key
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def load_eod(
        self,
        symbols: Sequence[str],
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        start_str = _to_str(start)
        end_str = _to_str(end)
        symbols_tuple = tuple(sorted(symbols))

        if self.cache and self.cache.exists(
            "polygon_eod", symbols_tuple, start_str, end_str
        ):
            return self.cache.read("polygon_eod", symbols_tuple, start_str, end_str)

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            url = f"{self._base_url}/aggs/ticker/{symbol.upper()}/range/1/day/{start_str}/{end_str}"
            data = self._get(url, {"adjusted": "true", "sort": "asc", "limit": 50000})
            results = data.get("results")
            if not results:
                continue
            df = pd.DataFrame(results)
            df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
            df = df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "transactions",
                }
            )
            df["symbol"] = symbol.upper()
            df = df[
                [
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "transactions",
                ]
            ]
            frames.append(df)

        if not frames:
            return _empty_eod_frame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.set_index(["date", "symbol"]).sort_index()

        if self.cache:
            self.cache.write(
                combined.reset_index(), "polygon_eod", symbols_tuple, start_str, end_str
            )

        return combined

    def load_universe(self, as_of: str | date | None = None) -> list[str]:
        as_of = as_of or datetime.now().date()
        as_of_str = _to_str(as_of)
        url = f"{self._reference_url}/tickers"
        params = {
            "market": "stocks",
            "active": "true",
            "date": as_of_str,
            "limit": 1000,
        }
        results: list[str] = []
        while True:
            data = self._get(url, params)
            for item in data.get("results", []):
                results.append(item["ticker"])
            next_url = data.get("next_url")
            if not next_url:
                break
            url = next_url
            params = {}
        return results

    def load_delistings(
        self,
        symbols: Sequence[str],
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        end_str = _to_str(end)
        records: list[dict[str, Any]] = []
        for symbol in symbols:
            url = f"{self._reference_url}/tickers/{symbol.upper()}"
            data = self._get(url, {"date": end_str})
            ticker = data.get("results", {})
            if not ticker or not ticker.get("active", True):
                # Polygon does not always expose a delisting date; use end if inactive.
                records.append({"symbol": symbol.upper(), "last_trade_date": end_str})
        return pd.DataFrame(records)

    def close(self) -> None:
        self._session.close()


def _to_str(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _empty_eod_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume", "vwap", "transactions"]
    ).set_index(pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"]))
