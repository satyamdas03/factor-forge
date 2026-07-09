"""Performance and risk metrics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series, risk_free_rate: float = 0.0
) -> dict[str, float]:
    """Compute standard performance metrics from a return series.

    Parameters
    ----------
    returns:
        Period returns (e.g., daily or monthly).
    risk_free_rate:
        Annualized risk-free rate to use for Sharpe/Sortino.
    """
    if returns.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
        }

    total_return = float((1 + returns).prod() - 1)
    periods_per_year = _infer_periods_per_year(returns)
    cagr = (
        float((1 + total_return) ** (periods_per_year / len(returns)) - 1)
        if len(returns) > 0
        else 0.0
    )
    vol = float(returns.std() * np.sqrt(periods_per_year))
    downside = (
        float(returns[returns < 0].std() * np.sqrt(periods_per_year))
        if (returns < 0).any()
        else 0.0
    )
    excess = cagr - risk_free_rate
    sharpe = excess / vol if vol > 0 else 0.0
    sortino = excess / downside if downside > 0 else 0.0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    win_rate = float((returns > 0).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
    }


def _infer_periods_per_year(returns: pd.Series) -> int:
    """Infer the number of periods per year from the index frequency."""
    try:
        freq = pd.infer_freq(returns.index)
    except (TypeError, ValueError):
        freq = None
    if freq in ("D", "B"):
        return 252
    if freq in ("W", "W-FRI"):
        return 52
    if freq in ("ME", "M", "MS"):
        return 12
    # Default to daily for mixed or irregular data.
    return 252
