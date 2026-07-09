"""Walk-forward expanding-window splits."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


class WalkForwardSplitter:
    """Generate expanding-window train/validation/test splits for panel data.

    Parameters
    ----------
    min_train_size:
        Minimum number of unique dates in the training window.
    test_size:
        Number of unique dates in each test fold.
    step:
        Number of unique dates to advance between folds.
    """

    def __init__(
        self,
        min_train_size: int = 252,
        test_size: int = 63,
        step: int = 63,
    ) -> None:
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step = step

    def split(self, df: pd.DataFrame) -> Iterator[tuple[pd.Index, pd.Index]]:
        """Yield (train_index, test_index) pairs."""
        dates = df.index.get_level_values("date").unique().sort_values()
        start = self.min_train_size
        while start + self.test_size <= len(dates):
            train_dates = dates[:start]
            test_dates = dates[start : start + self.test_size]
            train_mask = df.index.get_level_values("date").isin(train_dates)
            test_mask = df.index.get_level_values("date").isin(test_dates)
            yield df.index[train_mask], df.index[test_mask]
            start += self.step
