"""ML ensemble for combining factor signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

if TYPE_CHECKING:
    pass


def _has_lightgbm() -> bool:
    try:
        import lightgbm as lgb  # noqa: F401

        return True
    except ImportError:
        return False


class FactorEnsemble:
    """Walk-forward factor ensemble using LightGBM or Ridge fallback.

    Parameters
    ----------
    model_type:
        ``lightgbm`` or ``ridge``.
    model_params:
        Optional hyperparameters passed to the underlying model.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        model_params: dict[str, Any] | None = None,
    ) -> None:
        self.model_type = model_type
        self.model_params = model_params or {}
        if model_type == "lightgbm" and not _has_lightgbm():
            raise ImportError(
                "LightGBM not installed; install factor-forge[ml] or use ridge"
            )

    def _fit_model(self, features: pd.DataFrame, target: pd.Series) -> Any:
        if self.model_type == "lightgbm":
            import lightgbm as lgb

            params: dict[str, Any] = {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 4,
                "num_leaves": 16,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            params.update(self.model_params)
            model = lgb.LGBMRegressor(**params)
        else:
            model = Ridge(alpha=1.0)
        model.fit(features, target)
        return model

    def fit(
        self,
        features_train: pd.DataFrame,
        target_train: pd.Series,
    ) -> FactorEnsemble:
        """Fit the ensemble on training data."""
        self.feature_names = list(features_train.columns)
        self.model_ = self._fit_model(features_train, target_train)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict forward returns."""
        preds = self.model_.predict(features[self.feature_names])
        return pd.Series(preds, index=features.index, name="ensemble_score")

    def feature_importance(self) -> pd.Series:
        """Return feature importance if available."""
        if self.model_type == "lightgbm":
            return pd.Series(
                self.model_.feature_importances_,
                index=self.feature_names,
                name="importance",
            ).sort_values(ascending=False)
        return pd.Series(
            np.abs(self.model_.coef_), index=self.feature_names, name="importance"
        ).sort_values(ascending=False)
