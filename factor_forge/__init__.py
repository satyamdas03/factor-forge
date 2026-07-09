"""Factor Forge — a modern, reproducible factor zoo replication engine."""

__version__ = "0.1.1"

from factor_forge.analytics.metrics import compute_metrics
from factor_forge.backtest.engine import BacktestEngine, BacktestResult
from factor_forge.backtest.portfolio import Portfolio
from factor_forge.data.loader import DataLoader
from factor_forge.factors.base import Factor, FactorCategory
from factor_forge.factors.registry import FactorRegistry

__all__ = [
    "__version__",
    "BacktestEngine",
    "BacktestResult",
    "DataLoader",
    "Factor",
    "FactorCategory",
    "FactorRegistry",
    "Portfolio",
    "compute_metrics",
]
