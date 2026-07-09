"""Factor correlation and redundancy analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def factor_correlation_matrix(
    scores: pd.DataFrame, method: str = "spearman"
) -> pd.DataFrame:
    """Compute cross-factor rank correlation matrix."""
    return scores.corr(method=method)


def average_pairwise_correlation(corr: pd.DataFrame) -> float:
    """Mean absolute off-diagonal correlation."""
    values = corr.values.copy()
    np.fill_diagonal(values, np.nan)
    return float(np.nanmean(np.abs(values)))


def hierarchical_redundancy_score(corr: pd.DataFrame) -> float:
    """Simple redundancy score: mean absolute correlation across all factor pairs."""
    return average_pairwise_correlation(corr)
