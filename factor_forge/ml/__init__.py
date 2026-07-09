"""Machine learning ensemble layer for Factor Forge."""

from factor_forge.ml.conformal import empirical_coverage, split_conformal_intervals
from factor_forge.ml.ensemble import FactorEnsemble
from factor_forge.ml.features import add_sector_dummies, build_feature_matrix
from factor_forge.ml.selection import drop_highly_correlated, select_by_icir
from factor_forge.ml.walk_forward import WalkForwardSplitter

__all__ = [
    "add_sector_dummies",
    "build_feature_matrix",
    "drop_highly_correlated",
    "empirical_coverage",
    "FactorEnsemble",
    "select_by_icir",
    "split_conformal_intervals",
    "WalkForwardSplitter",
]
