"""Feature selection utilities for factor ensembles."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def select_by_icir(
    factor_scores: pd.DataFrame,
    forward_returns: pd.Series,
    min_ic: float = 0.02,
    max_factors: int | None = None,
) -> list[str]:
    """Select factors by information coefficient / IR.

    Parameters
    ----------
    factor_scores:
        DataFrame of factor scores indexed by ``(date, symbol)``.
    forward_returns:
        Forward return series with the same index.
    min_ic:
        Minimum absolute mean IC threshold.
    max_factors:
        Maximum number of factors to return.
    """
    from factor_forge.analytics.factor_ic import compute_ic

    ic_df = compute_ic(factor_scores, forward_returns, method="spearman")
    mean_ic = ic_df.abs().mean().sort_values(ascending=False)
    selected = cast(list[str], mean_ic[mean_ic >= min_ic].index.tolist())
    if max_factors:
        selected = selected[:max_factors]
    return selected


def drop_highly_correlated(
    factor_scores: pd.DataFrame,
    threshold: float = 0.95,
) -> list[str]:
    """Drop one factor from each pair with absolute correlation above threshold."""
    corr = factor_scores.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold and col not in to_drop:
                to_drop.add(col)
    return [c for c in factor_scores.columns if c not in to_drop]
