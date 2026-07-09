"""Factor registry for discovery, grouping, and batch computation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from factor_forge.factors.base import Factor
from factor_forge.factors.implementations import BUILTIN_FACTORS

if TYPE_CHECKING:
    from factor_forge.factors.base import FactorCategory


class FactorRegistry:
    """Central registry holding factor definitions."""

    def __init__(self) -> None:
        self._factors: dict[str, Factor] = {}

    def register(self, factor: Factor) -> None:
        """Register a factor."""
        if factor.name in self._factors:
            raise ValueError(f"Factor {factor.name} is already registered")
        self._factors[factor.name] = factor

    def get(self, name: str) -> Factor:
        """Retrieve a factor by name."""
        if name not in self._factors:
            raise KeyError(f"Unknown factor: {name}")
        return self._factors[name]

    def list_factors(self, category: FactorCategory | None = None) -> list[str]:
        """Return factor names, optionally filtered by category."""
        names = list(self._factors.keys())
        if category is None:
            return names
        return [name for name in names if self._factors[name].category == category]

    def load_builtin(self) -> None:
        """Load all built-in factors into the registry."""
        for factor in BUILTIN_FACTORS:
            self.register(factor)

    def compute(
        self,
        df: pd.DataFrame,
        names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Compute requested factors and return a DataFrame of scores.

        Parameters
        ----------
        df:
            Panel DataFrame indexed by ``(date, symbol)``.
        names:
            Factor names to compute. If None, compute all registered factors.
        """
        if names is None:
            names = list(self._factors.keys())
        frames: list[pd.DataFrame] = []
        for name in names:
            factor = self.get(name)
            try:
                score = factor(df)
                score.name = name
                frames.append(score.to_frame())
            except ValueError as exc:
                # Missing fundamental inputs are skipped with a warning-like behavior.
                # In production this should be logged; tests should not rely on silent skipping.
                raise RuntimeError(f"Factor {name} failed: {exc}") from exc
        if not frames:
            return pd.DataFrame(index=df.index)
        return pd.concat(frames, axis=1)
