"""Tests for the Streamlit dashboard helpers."""

import pytest

from factor_forge.dashboard import _streamlit_command
from factor_forge.dashboard.app import (
    compute_analytics,
    compute_factor_scores,
    load_prices,
    run_backtest,
)
from factor_forge.factors.registry import FactorRegistry


def test_load_synthetic_prices() -> None:
    prices = load_prices("synthetic", ["A", "B"], "2020-01-01", "2020-06-01")
    assert not prices.empty
    assert "close" in prices.columns
    symbols = prices.index.get_level_values("symbol").unique().tolist()
    assert set(symbols) == {"A", "B"}


def test_compute_factor_scores_and_backtest() -> None:
    registry = FactorRegistry()
    registry.load_builtin()
    factor = registry.get("momentum_12_1")
    prices = load_prices(
        "synthetic", ["A", "B", "C", "D", "E"], "2020-01-01", "2021-12-31"
    )
    scores = compute_factor_scores(factor, prices)
    assert factor.name in scores.columns
    result = run_backtest(factor, prices, cost_bps=0.0, long_short=True)
    assert not result.nav.empty
    analytics = compute_analytics(result, scores, prices)
    assert "ic_summary" in analytics
    assert "decile_returns" in analytics
    assert "half_life" in analytics


def test_streamlit_command() -> None:
    cmd = _streamlit_command()
    assert "streamlit" in cmd
    assert "run" in cmd
    assert any("app.py" in arg for arg in cmd)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("streamlit") is None,
    reason="streamlit not installed",
)
def test_dashboard_launch_imports() -> None:
    """Smoke test that the dashboard launcher can be imported without error."""
    from factor_forge.dashboard import launch

    assert callable(launch)
