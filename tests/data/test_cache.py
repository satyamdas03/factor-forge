"""Tests for the local data cache."""

from pathlib import Path

import pandas as pd
import pytest

from factor_forge.data.cache import DataCache


@pytest.fixture
def tmp_cache(tmp_path: Path) -> DataCache:
    cache_dir = tmp_path / "cache"
    return DataCache(cache_dir)


def test_cache_write_and_read(tmp_cache: DataCache) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    tmp_cache.write(df, "eod", ("AAPL", "MSFT"), "2020-01-01", "2020-12-31")
    assert tmp_cache.exists("eod", ("AAPL", "MSFT"), "2020-01-01", "2020-12-31")
    loaded = tmp_cache.read("eod", ("AAPL", "MSFT"), "2020-01-01", "2020-12-31")
    pd.testing.assert_frame_equal(loaded, df)


def test_cache_key_order_independent(tmp_cache: DataCache) -> None:
    df = pd.DataFrame({"a": [1]})
    tmp_cache.write(df, "eod", ("MSFT", "AAPL"), "2020-01-01", "2020-12-31")
    loaded = tmp_cache.read("eod", ("AAPL", "MSFT"), "2020-01-01", "2020-12-31")
    pd.testing.assert_frame_equal(loaded, df)


def test_cache_clear(tmp_cache: DataCache) -> None:
    df = pd.DataFrame({"a": [1]})
    tmp_cache.write(df, "eod", ("AAPL",), "2020-01-01", "2020-12-31")
    tmp_cache.clear()
    assert not tmp_cache.exists("eod", ("AAPL",), "2020-01-01", "2020-12-31")
