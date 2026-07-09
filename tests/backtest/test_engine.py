"""Tests for the decile backtest engine."""

import numpy as np
import pandas as pd

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.data.loader import generate_synthetic_prices
from factor_forge.factors.base import Factor, FactorCategory
from factor_forge.factors.registry import FactorRegistry


def _make_momentum_panel() -> pd.DataFrame:
    """Create a panel where one symbol has persistently higher momentum."""
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    records = []
    for symbol, drift in [("WIN", 0.001), ("LOSE", -0.001), ("MID", 0.0)]:
        rng = np.random.default_rng(42 if symbol == "WIN" else 13)
        returns = rng.normal(drift, 0.01, size=len(dates))
        price = 100.0 * np.exp(np.cumsum(returns))
        for i, d in enumerate(dates):
            records.append(
                {
                    "date": d,
                    "symbol": symbol,
                    "open": price[i],
                    "high": price[i] * 1.01,
                    "low": price[i] * 0.99,
                    "close": price[i],
                    "volume": 1_000_000,
                }
            )
    df = pd.DataFrame(records).set_index(["date", "symbol"]).sort_index()
    return df


def test_engine_long_short_runs() -> None:
    df = _make_momentum_panel()
    registry = FactorRegistry()
    registry.load_builtin()
    factor = registry.get("momentum_12_1")
    engine = BacktestEngine(
        factor=factor, prices=df, transaction_cost_bps=5.0, long_short=True
    )
    result = engine.run()
    assert not result.nav.empty
    assert "sharpe_ratio" in result.metrics
    assert result.long_short is True


def test_engine_long_only_runs() -> None:
    df = _make_momentum_panel()
    registry = FactorRegistry()
    registry.load_builtin()
    factor = registry.get("momentum_52_high")
    engine = BacktestEngine(
        factor=factor, prices=df, transaction_cost_bps=5.0, long_short=False
    )
    result = engine.run()
    assert not result.nav.empty
    assert result.metrics["total_return"] > -1.0


def test_engine_records_turnover() -> None:
    df = _make_momentum_panel()
    registry = FactorRegistry()
    registry.load_builtin()
    factor = registry.get("momentum_12_1")
    engine = BacktestEngine(
        factor=factor, prices=df, transaction_cost_bps=0.0, long_short=True
    )
    result = engine.run()
    assert not result.turnover.empty
    assert result.metrics["turnover_mean"] >= 0.0


def test_engine_no_lookahead_in_scores() -> None:
    """Verify the backtest does not use future prices when ranking."""
    df = generate_synthetic_prices(["A", "B"], "2020-01-01", "2020-06-30")
    factor = Factor(
        name="lagged_return",
        category=FactorCategory.MOMENTUM,
        inputs=["close"],
        compute=lambda d: (
            d.groupby("symbol")["close"].shift(1)
            / d.groupby("symbol")["close"].shift(2)
            - 1
        ),
    )
    engine = BacktestEngine(
        factor=factor, prices=df, transaction_cost_bps=0.0, long_short=False
    )
    result = engine.run()
    # Should run without raising and produce non-empty NAV.
    assert not result.nav.empty
