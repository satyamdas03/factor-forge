"""Abstract data source interface for Factor Forge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class DataSource(ABC):
    """Base class for market data sources.

    Implementations must provide end-of-day prices, corporate actions, and
    delisting information in a point-in-time manner.
    """

    name: str = "abstract"

    @abstractmethod
    def load_eod(
        self,
        symbols: Sequence[str],
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        """Return EOD prices as a long-form DataFrame.

        Columns: date, symbol, open, high, low, close, volume.
        Index is ``(date, symbol)`` if not a MultiIndex already.
        """

    @abstractmethod
    def load_universe(self, as_of: str | date | None = None) -> list[str]:
        """Return a list of active symbols as of the given date."""

    @abstractmethod
    def load_delistings(
        self,
        symbols: Sequence[str],
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        """Return delisting dates for the requested symbols.

        Columns: symbol, last_trade_date.
        """

    def close(self) -> None:  # noqa: B027
        """Release any network or file resources."""
