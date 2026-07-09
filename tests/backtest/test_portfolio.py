"""Tests for the portfolio bookkeeping class."""

import pytest

from factor_forge.backtest.portfolio import Portfolio


def test_portfolio_initial_nav() -> None:
    pf = Portfolio(initial_capital=100_000.0)
    assert pf.nav({}) == 100_000.0


def test_portfolio_rebalance_long_only() -> None:
    pf = Portfolio(initial_capital=100_000.0)
    pf.set_target_weights(
        date="2020-01-02",
        target_weights={"A": 0.5, "B": 0.5},
        prices={"A": 100.0, "B": 200.0},
        transaction_cost_bps=0.0,
    )
    assert pf.positions["A"] == pytest.approx(500.0, rel=1e-6)
    assert pf.positions["B"] == pytest.approx(250.0, rel=1e-6)
    assert pf.nav({"A": 100.0, "B": 200.0}) == pytest.approx(100_000.0, rel=1e-6)


def test_portfolio_transaction_costs_reduce_nav() -> None:
    pf = Portfolio(initial_capital=100_000.0)
    pf.set_target_weights(
        date="2020-01-02",
        target_weights={"A": 1.0},
        prices={"A": 100.0},
        transaction_cost_bps=10.0,
    )
    # Traded value = 100k, cost = 100 bps = 100, cash = -100, positions = 1000 shares
    assert pf.cash == pytest.approx(-100.0, rel=1e-6)
    assert pf.nav({"A": 100.0}) == pytest.approx(99_900.0, rel=1e-6)


def test_portfolio_long_short_dollar_neutral() -> None:
    pf = Portfolio(initial_capital=100_000.0)
    pf.set_target_weights(
        date="2020-01-02",
        target_weights={"A": 0.5, "B": -0.5},
        prices={"A": 100.0, "B": 100.0},
        transaction_cost_bps=0.0,
    )
    weights = pf.current_weights({"A": 100.0, "B": 100.0})
    assert weights["A"] == pytest.approx(0.5, rel=1e-6)
    assert weights["B"] == pytest.approx(-0.5, rel=1e-6)
