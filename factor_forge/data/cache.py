"""Local Parquet cache for market data frames."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class DataCache:
    """Filesystem cache keyed by data kind, symbols, and date range.

    Cache files are stored as Parquet under ``cache_dir``.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".factor_forge" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, kind: str, symbols: tuple[str, ...], start: str, end: str) -> str:
        payload = f"{kind}:{','.join(sorted(symbols))}:{start}:{end}"
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def path(self, kind: str, symbols: tuple[str, ...], start: str, end: str) -> Path:
        """Return the cache file path for a request."""
        key = self._key(kind, symbols, start, end)
        return self.cache_dir / f"{kind}_{key}.parquet"

    def exists(self, kind: str, symbols: tuple[str, ...], start: str, end: str) -> bool:
        """Check whether a cached Parquet file exists."""
        return self.path(kind, symbols, start, end).exists()

    def read(
        self, kind: str, symbols: tuple[str, ...], start: str, end: str
    ) -> pd.DataFrame:
        """Read a cached DataFrame, raising if missing."""
        import pandas as pd

        path = self.path(kind, symbols, start, end)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    def write(
        self,
        df: pd.DataFrame,
        kind: str,
        symbols: tuple[str, ...],
        start: str,
        end: str,
    ) -> None:
        """Write ``df`` to the cache."""
        path = self.path(kind, symbols, start, end)
        df.to_parquet(path, index=True)

    def clear(self) -> None:
        """Remove all cached files."""
        for entry in self.cache_dir.iterdir():
            if entry.is_file():
                os.remove(entry)
