"""Tests for the ML ensemble layer."""

import numpy as np
import pandas as pd
import pytest

from factor_forge.data.loader import generate_synthetic_prices
from factor_forge.factors.registry import FactorRegistry
from factor_forge.ml.conformal import split_conformal_intervals
from factor_forge.ml.ensemble import FactorEnsemble
from factor_forge.ml.features import build_feature_matrix
from factor_forge.ml.walk_forward import WalkForwardSplitter


@pytest.fixture
def feature_panel() -> pd.DataFrame:
    prices = generate_synthetic_prices(
        ["A", "B", "C"], "2020-01-01", "2022-12-31", seed=7
    )
    registry = FactorRegistry()
    registry.load_builtin()
    price_factors = registry.list_factors()
    price_factors = [
        name
        for name in price_factors
        if registry.get(name).category.value in {"momentum", "low_volatility"}
    ]
    scores = registry.compute(prices, names=price_factors)
    return build_feature_matrix(prices, scores, forward_horizon=21)


def test_build_feature_matrix(feature_panel: pd.DataFrame) -> None:
    assert "target" in feature_panel.columns
    assert len(feature_panel.columns) > 1
    assert feature_panel.isna().sum().sum() == 0


def test_walk_forward_splitter(feature_panel: pd.DataFrame) -> None:
    splitter = WalkForwardSplitter(min_train_size=252, test_size=63, step=63)
    splits = list(splitter.split(feature_panel))
    assert len(splits) > 0
    train_idx, test_idx = splits[0]
    assert len(train_idx) > 0 and len(test_idx) > 0
    assert len(train_idx.intersection(test_idx)) == 0


def test_ridge_ensemble_predicts(feature_panel: pd.DataFrame) -> None:
    train = feature_panel.iloc[:500]
    test = feature_panel.iloc[500:600]
    features_train = train.drop(columns=["target"])
    target_train = train["target"]
    ensemble = FactorEnsemble(model_type="ridge")
    ensemble.fit(features_train, target_train)
    preds = ensemble.predict(test.drop(columns=["target"]))
    assert len(preds) == len(test)


def test_conformal_coverage() -> None:
    rng = np.random.default_rng(0)
    cal_idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=50, freq="D"), ["A", "B"]],
        names=["date", "symbol"],
    )
    y_true_cal = pd.Series(rng.normal(size=len(cal_idx)), index=cal_idx)
    y_pred_cal = y_true_cal + rng.normal(scale=0.1, size=len(cal_idx))
    test_idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-03-01", periods=20, freq="D"), ["A", "B"]],
        names=["date", "symbol"],
    )
    y_pred_test = pd.Series(rng.normal(size=len(test_idx)), index=test_idx)
    intervals = split_conformal_intervals(
        y_true_cal, y_pred_cal, y_pred_test, alpha=0.1
    )
    assert "lower" in intervals.columns
    assert "upper" in intervals.columns
    assert (intervals["upper"] >= intervals["lower"]).all()


def test_lightgbm_not_available_without_extra() -> None:
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):
        # OSError on macOS when the libomp runtime is missing.
        with pytest.raises(ImportError):
            FactorEnsemble(model_type="lightgbm")
