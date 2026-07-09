"""Factor information coefficient analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(
    factor_scores: pd.Series | pd.DataFrame,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Compute per-period information coefficient between factor and forward returns.

    Parameters
    ----------
    factor_scores:
        Series or DataFrame indexed by ``(date, symbol)``.
    forward_returns:
        Series of forward returns with the same index.
    method:
        ``spearman`` or ``pearson``.
    """
    if isinstance(factor_scores, pd.DataFrame):
        # Compute IC for each factor column.
        return factor_scores.apply(lambda col: compute_ic(col, forward_returns, method))

    df = pd.DataFrame({"score": factor_scores, "fwd": forward_returns}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def _corr(sub: pd.DataFrame) -> float:
        if len(sub) < 3:
            return np.nan
        if method == "spearman":
            return float(stats.spearmanr(sub["score"], sub["fwd"])[0])
        return float(sub["score"].corr(sub["fwd"], method="pearson"))

    return df.groupby(level="date").apply(_corr).dropna()


def compute_ic_summary(ic: pd.Series) -> dict[str, float]:
    """Return mean IC, ICIR, and t-stat."""
    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    return {
        "mean_ic": mean_ic,
        "icir": mean_ic / std_ic if std_ic > 0 else 0.0,
        "t_stat": mean_ic / (std_ic / np.sqrt(len(ic))) if std_ic > 0 else 0.0,
    }


def compute_rolling_ic(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
    window: int = 12,
) -> pd.Series:
    """Rolling mean IC over ``window`` periods."""
    ic = compute_ic(factor_scores, forward_returns, method="spearman")
    return ic.rolling(window, min_periods=max(1, window // 2)).mean()
