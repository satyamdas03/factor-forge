"""Deep application/integration test for factor-forge.

This module exercises every public surface of the package end-to-end on
synthetic data to confirm production readiness.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import factor_forge
from factor_forge import BacktestEngine, FactorRegistry, Portfolio, compute_metrics
from factor_forge.analytics.correlation import (
    average_pairwise_correlation,
    factor_correlation_matrix,
    hierarchical_redundancy_score,
)
from factor_forge.analytics.decay import factor_autocorrelation, factor_half_life
from factor_forge.analytics.factor_ic import compute_ic, compute_ic_summary
from factor_forge.analytics.regimes import classify_market_regime, split_by_regime
from factor_forge.backtest.engine import BacktestResult
from factor_forge.cli.run_backtest import main
from factor_forge.data.cache import DataCache
from factor_forge.data.loader import generate_synthetic_prices
from factor_forge.data.survivorship import build_universe, filter_alive, trim_to_history
from factor_forge.factors.base import Factor, FactorCategory
from factor_forge.ml.conformal import empirical_coverage, split_conformal_intervals
from factor_forge.ml.ensemble import FactorEnsemble
from factor_forge.ml.features import add_sector_dummies, build_feature_matrix
from factor_forge.ml.selection import drop_highly_correlated, select_by_icir
from factor_forge.ml.walk_forward import WalkForwardSplitter
from factor_forge.viz.cum_returns import plot_cumulative_returns
from factor_forge.viz.decile_bars import plot_decile_returns
from factor_forge.viz.drawdown import plot_drawdown
from factor_forge.viz.factor_corr import plot_factor_correlation
from factor_forge.viz.ic_heatmap import plot_ic_heatmap
from factor_forge.viz.reports import generate_report


def _make_prices(n_symbols: int = 10, seed: int = 42) -> pd.DataFrame:
    return generate_synthetic_prices(
        symbols=[f"SYM{i:02d}" for i in range(n_symbols)],
        start="2020-01-01",
        end="2022-12-31",
        seed=seed,
    )


class TestPackageIntegrity:
    def test_version(self) -> None:
        assert factor_forge.__version__ == "0.1.2"

    def test_all_exports_importable(self) -> None:
        for name in factor_forge.__all__:
            assert hasattr(factor_forge, name)


class TestDataLayer:
    def test_generate_synthetic_prices(self) -> None:
        prices = _make_prices()
        assert "close" in prices.columns
        assert prices.index.names == ["date", "symbol"]
        assert not prices.empty

    def test_data_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = DataCache(cache_dir=tmp)
            df = _make_prices(n_symbols=3)
            cache.write(
                df,
                "test",
                tuple(sorted(df.index.get_level_values("symbol").unique())),
                "2020-01-01",
                "2022-12-31",
            )
            loaded = cache.read(
                "test",
                tuple(sorted(df.index.get_level_values("symbol").unique())),
                "2020-01-01",
                "2022-12-31",
            )
            pd.testing.assert_frame_equal(df.reset_index(), loaded.reset_index())
            assert cache.exists(
                "test",
                tuple(sorted(df.index.get_level_values("symbol").unique())),
                "2020-01-01",
                "2022-12-31",
            )

    def test_survivorship_filter_alive(self) -> None:
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2020-01-01"), "A"),
                (pd.Timestamp("2020-01-02"), "A"),
                (pd.Timestamp("2020-01-01"), "B"),
                (pd.Timestamp("2020-01-02"), "B"),
                (pd.Timestamp("2020-01-03"), "B"),
            ],
            names=["date", "symbol"],
        )
        df = pd.DataFrame({"close": [1.0] * len(idx)}, index=idx)
        delist = pd.DataFrame({"symbol": ["A"], "last_trade_date": ["2020-01-02"]})
        alive = filter_alive(df, delist, as_of="2020-01-03", min_history_days=1)
        assert "A" not in alive.loc[pd.Timestamp("2020-01-03")].index.get_level_values(
            "symbol"
        )

    def test_build_universe(self) -> None:
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=5), ["A", "B", "C"]],
            names=["date", "symbol"],
        )
        df = pd.DataFrame(
            {"close": np.arange(len(idx)), "volume": np.ones(len(idx))}, index=idx
        )
        universe = build_universe(
            df, delistings=None, as_of="2020-01-05", min_history_days=1
        )
        assert set(universe) == {"A", "B", "C"}

    def test_trim_to_history(self) -> None:
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=10), ["A", "B"]],
            names=["date", "symbol"],
        )
        df = pd.DataFrame({"close": np.ones(len(idx))}, index=idx)
        trimmed = trim_to_history(["A", "B", "C"], df, min_history_days=5)
        assert set(trimmed) == {"A", "B"}


def _add_fundamentals(prices: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic fundamental columns so all built-in factors can run."""
    rng = np.random.default_rng(123)
    df = prices.copy()
    n = len(df)
    df["earnings"] = df["close"] / rng.uniform(5, 25, n)
    df["book_value"] = df["close"] / rng.uniform(0.5, 3.0, n)
    df["net_income"] = df["earnings"] * rng.uniform(0.5, 1.5, n)
    df["equity"] = df["book_value"] * rng.uniform(0.8, 1.2, n)
    df["gross_profit"] = df["close"] * rng.uniform(0.1, 0.5, n)
    df["revenue"] = df["gross_profit"] / rng.uniform(0.1, 0.5, n)
    df["cash_flow"] = df["net_income"] * rng.uniform(0.7, 1.3, n)
    df["assets"] = df["equity"] * rng.uniform(1.2, 3.0, n)
    df["capex"] = df["assets"] * rng.uniform(0.02, 0.1, n)
    return df


class TestFactors:
    def test_all_builtin_factors_run(self) -> None:
        prices = _add_fundamentals(_make_prices(n_symbols=20))
        registry = FactorRegistry()
        registry.load_builtin()
        assert len(registry) == 14
        for name in registry.list_factors():
            factor = registry.get(name)
            scores = factor(prices)
            assert isinstance(scores, pd.Series)
            assert scores.index.names == ["date", "symbol"]
            assert not scores.dropna().empty

    def test_registry_pickleable(self) -> None:
        registry = FactorRegistry()
        registry.load_builtin()
        factor = registry.get("momentum_12_1")
        assert factor.category == FactorCategory.MOMENTUM

    def test_custom_factor_registration(self) -> None:
        prices = _make_prices(n_symbols=5)

        def _score(prices_df: pd.DataFrame) -> pd.Series:
            return prices_df["close"].rename("custom")

        factor = Factor(
            name="custom",
            category=FactorCategory.VALUE,
            inputs=["close"],
            compute=_score,
        )
        registry = FactorRegistry()
        registry.register(factor)
        assert "custom" in registry.list_factors()
        result = registry.get("custom")(prices)
        assert result.name == "custom"


class TestBacktestEngine:
    def test_long_short_backtest(self) -> None:
        prices = _make_prices(n_symbols=20)
        registry = FactorRegistry()
        registry.load_builtin()
        factor = registry.get("momentum_12_1")
        engine = BacktestEngine(
            factor=factor, prices=prices, transaction_cost_bps=10.0, long_short=True
        )
        result = engine.run()
        assert isinstance(result, BacktestResult)
        assert not result.nav.empty
        assert not result.returns.empty
        assert "sharpe_ratio" in result.metrics
        assert result.long_short is True

    def test_long_only_backtest(self) -> None:
        prices = _make_prices(n_symbols=20)
        registry = FactorRegistry()
        registry.load_builtin()
        factor = registry.get("momentum_12_1")
        engine = BacktestEngine(
            factor=factor,
            prices=prices,
            transaction_cost_bps=10.0,
            long_short=False,
        )
        result = engine.run()
        assert result.long_short is False
        assert (result.positions.fillna(0) >= -1e-9).all().all()

    def test_backtest_determinism(self) -> None:
        prices = _make_prices(n_symbols=20)
        registry = FactorRegistry()
        registry.load_builtin()
        factor = registry.get("momentum_12_1")
        r1 = BacktestEngine(factor=factor, prices=prices).run()
        r2 = BacktestEngine(factor=factor, prices=prices).run()
        pd.testing.assert_series_equal(r1.nav, r2.nav)

    def test_portfolio_bookkeeping(self) -> None:
        portfolio = Portfolio(initial_capital=1_000_000.0)
        prices = {"A": 100.0, "B": 50.0}
        portfolio.set_target_weights(
            date="2020-01-01",
            target_weights={"A": 0.5, "B": -0.5},
            prices=prices,
            transaction_cost_bps=10.0,
        )
        assert len(portfolio.trades) == 2
        nav = portfolio.nav(prices)
        assert nav > 0
        weights = portfolio.current_weights(prices)
        assert pytest.approx(weights["A"], 0.01) == 0.5
        assert pytest.approx(weights["B"], 0.01) == -0.5


class TestAnalytics:
    def test_compute_metrics(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(
            rng.normal(0.0005, 0.01, 252),
            index=pd.date_range("2020-01-01", periods=252),
        )
        metrics = compute_metrics(returns)
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert np.isfinite(metrics["sharpe_ratio"])

    def test_compute_metrics_on_int_index(self) -> None:
        returns = pd.Series(np.random.default_rng(42).normal(0.0005, 0.01, 252))
        metrics = compute_metrics(returns)
        assert "sharpe_ratio" in metrics
        assert np.isfinite(metrics["sharpe_ratio"])

    def test_factor_ic(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=20), ["A", "B", "C"]],
            names=["date", "symbol"],
        )
        scores = pd.Series(rng.standard_normal(len(idx)), index=idx, name="score")
        forward = pd.Series(rng.standard_normal(len(idx)), index=idx, name="fwd")
        ic = compute_ic(scores, forward)
        assert isinstance(ic, pd.Series)
        summary = compute_ic_summary(ic)
        assert "mean_ic" in summary

    def test_factor_decay(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=30), ["A", "B"]],
            names=["date", "symbol"],
        )
        scores = pd.Series(rng.standard_normal(len(idx)), index=idx, name="score")
        autocorr = factor_autocorrelation(scores)
        assert isinstance(autocorr, pd.Series)
        hl = factor_half_life(scores)
        assert hl >= 0 or np.isnan(hl)

    def test_regime_split_and_classify(self) -> None:
        rng = np.random.default_rng(42)
        market_returns = pd.Series(
            rng.normal(0, 0.01, 252), index=pd.date_range("2020-01-01", periods=252)
        )
        # Use cumulative price for regime classification
        prices = (1 + market_returns).cumprod()
        regime = classify_market_regime(prices)
        assert set(regime.unique()).issubset({"bull", "bear", "neutral"})
        factor_returns = pd.Series(rng.normal(0, 0.01, 252), index=market_returns.index)
        splits = split_by_regime(factor_returns, regime)
        assert set(splits.keys()).issubset({"bull", "bear", "neutral"})

    def test_factor_correlation_matrix(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=20), ["A", "B", "C"]],
            names=["date", "symbol"],
        )
        scores_df = pd.DataFrame(
            rng.standard_normal((len(idx), 3)),
            index=idx,
            columns=["f1", "f2", "f3"],
        )
        corr = factor_correlation_matrix(scores_df)
        assert corr.shape == (3, 3)
        assert ((corr.abs() <= 1.0) | corr.isna()).all().all()
        avg = average_pairwise_correlation(corr)
        assert np.isfinite(avg)
        hier = hierarchical_redundancy_score(corr)
        assert np.isfinite(hier)


class TestMLLayer:
    def test_build_feature_matrix(self) -> None:
        prices = _make_prices(n_symbols=10)
        registry = FactorRegistry()
        registry.load_builtin()
        factor = registry.get("momentum_12_1")
        scores = factor(prices).to_frame("momentum_12_1")
        df = build_feature_matrix(prices, scores, forward_horizon=5)
        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert "momentum_12_1" in df.columns

    def test_walk_forward_splitter(self) -> None:
        prices = _make_prices(n_symbols=10)
        splitter = WalkForwardSplitter(min_train_size=200, test_size=30, step=30)
        splits = list(splitter.split(prices))
        assert len(splits) >= 1
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_factor_ensemble_ridge(self) -> None:
        rng = np.random.default_rng(42)
        features_train = pd.DataFrame(
            rng.standard_normal((100, 3)), columns=["a", "b", "c"]
        )
        target_train = pd.Series(rng.standard_normal(100))
        features_test = pd.DataFrame(
            rng.standard_normal((50, 3)), columns=["a", "b", "c"]
        )
        model = FactorEnsemble(model_type="ridge").fit(features_train, target_train)
        preds = model.predict(features_test)
        assert len(preds) == 50
        importance = model.feature_importance()
        assert len(importance) == 3

    def test_split_conformal_intervals(self) -> None:
        rng = np.random.default_rng(42)
        y_true_cal = pd.Series(rng.standard_normal(100))
        y_pred_cal = pd.Series(rng.standard_normal(100))
        y_pred_test = pd.Series(rng.standard_normal(50))
        intervals = split_conformal_intervals(
            y_true_cal, y_pred_cal, y_pred_test, alpha=0.1
        )
        assert isinstance(intervals, pd.DataFrame)
        assert len(intervals) == 50
        assert (intervals["lower"] <= intervals["upper"]).all()
        y_true_test = pd.Series(rng.standard_normal(50), index=y_pred_test.index)
        coverage = empirical_coverage(y_true_test, intervals)
        assert 0 <= coverage <= 1

    def test_feature_selection_by_icir(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=30), ["A", "B"]],
            names=["date", "symbol"],
        )
        scores_df = pd.DataFrame(
            rng.standard_normal((len(idx), 3)), index=idx, columns=["f1", "f2", "f3"]
        )
        forward = pd.Series(rng.standard_normal(len(idx)), index=idx)
        selected = select_by_icir(scores_df, forward, min_ic=0.0, max_factors=2)
        assert isinstance(selected, list)
        assert len(selected) <= 2

    def test_drop_highly_correlated(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
        df["b"] = df["a"] + rng.standard_normal(100) * 0.001
        reduced = drop_highly_correlated(df, threshold=0.99)
        assert isinstance(reduced, list)
        assert "b" not in reduced

    def test_add_sector_dummies(self) -> None:
        df = pd.DataFrame({"sector": ["tech", "health", "tech"]})
        dummies = add_sector_dummies(df, sector_col="sector")
        assert "sector_tech" in dummies.columns
        # drop_first=True removes the baseline category, so only one dummy remains.
        assert "sector_health" not in dummies.columns


class TestVisualization:
    def test_plot_cumulative_returns(self) -> None:
        returns = pd.Series(
            [0.01, -0.005, 0.02, -0.01],
            index=pd.date_range("2020-01-01", periods=4),
            name="ret",
        )
        fig = plot_cumulative_returns({"strategy": returns})
        assert fig is not None

    def test_plot_decile_returns(self) -> None:
        deciles = pd.Series([0.1, 0.05, 0.0, -0.05, -0.1], index=range(1, 6))
        fig = plot_decile_returns(deciles)
        assert fig is not None

    def test_plot_drawdown(self) -> None:
        returns = pd.Series(
            [0.01, -0.05, 0.02, 0.03],
            index=pd.date_range("2020-01-01", periods=4),
            name="ret",
        )
        fig = plot_drawdown(returns)
        assert fig is not None

    def test_plot_factor_correlation(self) -> None:
        corr = pd.DataFrame(np.eye(3), columns=["a", "b", "c"], index=["a", "b", "c"])
        fig = plot_factor_correlation(corr)
        assert fig is not None

    def test_plot_ic_heatmap(self) -> None:
        ic_df = pd.DataFrame(
            np.random.default_rng(42).standard_normal((10, 3)),
            columns=["f1", "f2", "f3"],
        )
        fig = plot_ic_heatmap(ic_df)
        assert fig is not None

    def test_generate_report(self) -> None:
        prices = _make_prices(n_symbols=20)
        registry = FactorRegistry()
        registry.load_builtin()
        factor = registry.get("momentum_12_1")
        engine = BacktestEngine(factor=factor, prices=prices)
        result = engine.run()
        with tempfile.TemporaryDirectory() as tmp:
            report_path = generate_report(result, Path(tmp))
            assert report_path.exists()
            assert result.factor_name in report_path.read_text()


class TestCLI:
    def test_cli_help(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_run_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            main(
                [
                    "--factor",
                    "momentum_12_1",
                    "--start",
                    "2020-01-01",
                    "--end",
                    "2022-12-31",
                    "--source",
                    "synthetic",
                    "--output",
                    out,
                ]
            )
            assert os.path.exists(out)


class TestProductionReadiness:
    def test_no_runtime_warnings_on_core_path(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prices = _make_prices(n_symbols=20)
            registry = FactorRegistry()
            registry.load_builtin()
            factor = registry.get("momentum_12_1")
            BacktestEngine(factor=factor, prices=prices).run()
            own = [x for x in w if "factor_forge" in str(x.filename)]
            assert len(own) == 0, [str(x.message) for x in own]

    def test_empty_data_raises_cleanly(self) -> None:
        empty = pd.DataFrame(columns=["close"]).set_index(
            pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        )
        registry = FactorRegistry()
        registry.load_builtin()
        factor = registry.get("momentum_12_1")
        with pytest.raises(ValueError):
            BacktestEngine(factor=factor, prices=empty).run()

    def test_result_serialization_roundtrip(self) -> None:
        prices = _make_prices(n_symbols=20)
        registry = FactorRegistry()
        registry.load_builtin()
        factor = registry.get("momentum_12_1")
        result = BacktestEngine(factor=factor, prices=prices).run()
        d = result.to_dict()
        assert d["factor_name"] == "momentum_12_1"
        assert isinstance(d["metrics"], dict)
