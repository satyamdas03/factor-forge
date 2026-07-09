"""Core factor abstraction."""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class FactorCategory(enum.Enum):
    """Broad factor category used for grouping and analysis."""

    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    INVESTMENT = "investment"
    PROFITABILITY = "profitability"
    SENTIMENT = "sentiment"
    MACRO = "macro"


@dataclass(frozen=True)
class Factor:
    """A cross-sectional factor definition.

    Attributes
    ----------
    name:
        Unique factor identifier.
    category:
        ``FactorCategory`` for grouping.
    inputs:
        Column names required in the input DataFrame.
    compute:
        Function ``(df: pd.DataFrame) -> pd.Series`` mapping panel data to a
        cross-sectional score indexed by ``(date, symbol)``. Positive values
        should correspond to the desired long exposure (e.g., high momentum).
    """

    name: str
    category: FactorCategory
    inputs: Sequence[str]
    compute: Callable[[pd.DataFrame], pd.Series]

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """Run the factor on ``df`` and return a score Series."""
        missing = [col for col in self.inputs if col not in df.columns]
        if missing:
            raise ValueError(f"Factor {self.name} missing inputs: {missing}")
        return self.compute(df)
