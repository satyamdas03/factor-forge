"""Tests for interactive research report generation."""

import pytest

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.data.loader import generate_synthetic_prices
from factor_forge.factors.registry import FactorRegistry
from factor_forge.viz.research_report import (
    _compute_decile_returns,
    _compute_forward_returns,
    generate_research_report,
)


@pytest.fixture
def backtest_result_and_scores():
    prices = generate_synthetic_prices(
        ["A", "B", "C", "D", "E"], "2020-01-01", "2022-12-31"
    )
    registry = FactorRegistry()
    registry.load_builtin()
    factor = registry.get("momentum_12_1")
    scores = registry.compute(prices, names=["momentum_12_1"])
    engine = BacktestEngine(factor=factor, prices=prices, transaction_cost_bps=0.0)
    result = engine.run()
    return result, scores, prices, factor


def test_compute_forward_returns(backtest_result_and_scores):
    _, _, prices, _ = backtest_result_and_scores
    fwd = _compute_forward_returns(prices)
    assert len(fwd) == len(prices)
    assert fwd.name == "forward_return"


def test_compute_decile_returns(backtest_result_and_scores):
    _, scores, prices, _ = backtest_result_and_scores
    fwd = _compute_forward_returns(prices)
    decile_returns = _compute_decile_returns(scores["momentum_12_1"], fwd)
    assert len(decile_returns) > 0
    assert (decile_returns.index >= 1).all()


def test_generate_research_report(backtest_result_and_scores, tmp_path):
    result, scores, prices, factor = backtest_result_and_scores
    output = tmp_path / "report.html"
    path = generate_research_report(
        result=result,
        prices=prices,
        scores=scores,
        output_path=output,
        factor=factor,
    )
    assert path == output
    assert output.exists()
    html_text = output.read_text(encoding="utf-8")
    assert result.factor_name in html_text
    assert "Cumulative Returns" in html_text
    assert "Drawdown" in html_text
    assert "Decile Performance" in html_text
    assert "Information Coefficient" in html_text
    assert "Turnover" in html_text


def test_generate_research_report_with_dataframe_scores(
    backtest_result_and_scores, tmp_path
):
    result, scores, prices, factor = backtest_result_and_scores
    # Duplicate the column to exercise multi-factor correlation path.
    scores_multi = scores.copy()
    scores_multi["momentum_12_1_copy"] = scores_multi["momentum_12_1"]
    output = tmp_path / "report_multi.html"
    path = generate_research_report(
        result=result,
        prices=prices,
        scores=scores_multi,
        output_path=output,
        factor=factor,
    )
    assert path == output
    html_text = output.read_text(encoding="utf-8")
    assert "Factor Rank Correlation Matrix" in html_text
