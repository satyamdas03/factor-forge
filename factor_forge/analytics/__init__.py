"""Analytics suite for Factor Forge backtests."""

from factor_forge.analytics.correlation import (
    average_pairwise_correlation,
    factor_correlation_matrix,
    hierarchical_redundancy_score,
)
from factor_forge.analytics.decay import factor_autocorrelation, factor_half_life
from factor_forge.analytics.factor_ic import (
    compute_ic,
    compute_ic_summary,
    compute_rolling_ic,
)
from factor_forge.analytics.metrics import compute_metrics
from factor_forge.analytics.regimes import classify_market_regime, split_by_regime

__all__ = [
    "average_pairwise_correlation",
    "classify_market_regime",
    "compute_ic",
    "compute_ic_summary",
    "compute_metrics",
    "compute_rolling_ic",
    "factor_autocorrelation",
    "factor_correlation_matrix",
    "factor_half_life",
    "hierarchical_redundancy_score",
    "split_by_regime",
]
