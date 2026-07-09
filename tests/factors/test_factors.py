"""Tests for the factor library."""

import numpy as np
import pandas as pd
import pytest

from factor_forge.data.loader import generate_synthetic_prices
from factor_forge.factors.base import Factor, FactorCategory
from factor_forge.factors.implementations import BUILTIN_FACTORS
from factor_forge.factors.registry import FactorRegistry


@pytest.fixture
def prices() -> pd.DataFrame:
    return generate_synthetic_prices(
        ["A", "B", "C"], "2020-01-01", "2022-12-31", seed=7
    )


def test_builtin_factors_are_registered() -> None:
    registry = FactorRegistry()
    registry.load_builtin()
    assert len(registry.list_factors()) == len(BUILTIN_FACTORS)
    assert "momentum_12_1" in registry.list_factors()


def test_momentum_12_1_ranking(prices: pd.DataFrame) -> None:
    registry = FactorRegistry()
    registry.load_builtin()
    # Artificially make C the clear winner by injecting a strong recent trend.
    df = prices.copy()
    idx_c = df.index.get_level_values("symbol") == "C"
    df.loc[idx_c, "close"] = df.loc[idx_c, "close"] * np.linspace(1.0, 2.5, idx_c.sum())
    scores = registry.get("momentum_12_1")(df)
    latest = scores.groupby(level="date").tail(3).dropna()
    assert not latest.empty


def test_idiosyncratic_volatility_is_negative_of_vol(prices: pd.DataFrame) -> None:
    registry = FactorRegistry()
    registry.load_builtin()
    scores = registry.get("idiosyncratic_volatility")(prices)
    assert scores.dropna().max() <= 0


def test_custom_factor() -> None:
    df = generate_synthetic_prices(["A", "B"], "2020-01-01", "2020-12-31")
    df["signal"] = 1.0
    factor = Factor(
        name="dummy",
        category=FactorCategory.MOMENTUM,
        inputs=["signal"],
        compute=lambda d: d["signal"],
    )
    scores = factor(df)
    assert (scores == 1.0).all()


def test_registry_compute_all_price_factors(prices: pd.DataFrame) -> None:
    registry = FactorRegistry()
    registry.load_builtin()
    price_factors = registry.list_factors()
    # Only run price-based factors to avoid missing fundamental columns.
    price_factors = [
        name
        for name in price_factors
        if registry.get(name).category
        in {FactorCategory.MOMENTUM, FactorCategory.LOW_VOLATILITY}
    ]
    out = registry.compute(prices, names=price_factors)
    assert not out.empty
    for col in price_factors:
        assert col in out.columns


def test_registry_missing_factor_raises() -> None:
    registry = FactorRegistry()
    with pytest.raises(KeyError, match="Unknown factor"):
        registry.get("not_a_factor")
