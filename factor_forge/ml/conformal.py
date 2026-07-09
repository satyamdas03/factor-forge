"""Conformal prediction wrapper for factor-ensemble return forecasts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def split_conformal_intervals(
    y_true_cal: pd.Series,
    y_pred_cal: pd.Series,
    y_pred_test: pd.Series,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Compute split conformal prediction intervals.

    Parameters
    ----------
    y_true_cal:
        True targets on the calibration set.
    y_pred_cal:
        Model predictions on the calibration set.
    y_pred_test:
        Model predictions on the test set.
    alpha:
        Miscoverage rate (default 0.1 for 90% coverage).

    Returns
    -------
    DataFrame with columns ``lower`` and ``upper`` indexed like ``y_pred_test``.
    """
    residuals = (y_true_cal - y_pred_cal).abs().dropna()
    if residuals.empty:
        quantile = 0.0
    else:
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        quantile = residuals.quantile(min(q_level, 1.0))

    return pd.DataFrame(
        {
            "lower": y_pred_test - quantile,
            "upper": y_pred_test + quantile,
        },
        index=y_pred_test.index,
    )


def empirical_coverage(
    y_true: pd.Series,
    intervals: pd.DataFrame,
) -> float:
    """Compute the empirical coverage of a conformal interval DataFrame."""
    df = pd.concat([y_true, intervals], axis=1).dropna()
    covered = (df["target"] >= df["lower"]) & (df["target"] <= df["upper"])
    return float(covered.mean()) if len(covered) > 0 else 0.0
