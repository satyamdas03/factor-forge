"""Build lagged, look-ahead-free feature matrices for ML."""

from __future__ import annotations

import pandas as pd


def build_feature_matrix(
    prices: pd.DataFrame,
    factor_scores: pd.DataFrame,
    forward_horizon: int = 21,
    target_type: str = "return",
) -> pd.DataFrame:
    """Build an ML-ready panel with lagged factors and a forward target.

    Parameters
    ----------
    prices:
        Long-form price panel with ``close`` column, indexed by ``(date, symbol)``.
    factor_scores:
        DataFrame of factor scores indexed by ``(date, symbol)``.
    forward_horizon:
        Number of periods ahead to predict.
    target_type:
        ``return`` for forward simple return.

    Returns
    -------
    DataFrame with columns ``[target] + factor_score columns``, indexed by
    ``(date, symbol)``. All features are lagged one period relative to the target.
    """
    close = prices["close"]
    future_close = close.groupby("symbol").shift(-forward_horizon)
    if target_type == "return":
        target = (future_close / close - 1).rename("target")
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    lagged_factors = factor_scores.groupby("symbol").shift(1)
    df = pd.concat([target, lagged_factors], axis=1)
    return df.dropna()


def add_sector_dummies(df: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    """Add one-hot sector dummies if a sector column exists in the input data."""
    if sector_col not in df.columns:
        return df
    dummies = pd.get_dummies(df[sector_col], prefix="sector", drop_first=True)
    return pd.concat([df.drop(columns=[sector_col]), dummies], axis=1)
