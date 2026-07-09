"""Decile backtest engine for cross-sectional factors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from factor_forge.backtest.portfolio import Portfolio

if TYPE_CHECKING:
    from factor_forge.backtest.portfolio import Trade
    from factor_forge.factors.base import Factor


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    nav: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    turnover: pd.Series
    metrics: dict[str, float]
    factor_name: str
    long_short: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize result to a dictionary."""
        return {
            "factor_name": self.factor_name,
            "long_short": self.long_short,
            "metrics": self.metrics,
            "nav": self.nav.to_dict(),
            "returns": self.returns.to_dict(),
            "turnover": self.turnover.to_dict(),
        }


class BacktestEngine:
    """Monthly decile backtest engine.

    Parameters
    ----------
    factor:
        The ``Factor`` to test.
    prices:
        Long-form price DataFrame indexed by ``(date, symbol)`` with at least
        a ``close`` column.
    transaction_cost_bps:
        Cost per dollar traded, in basis points.
    long_short:
        If True, go long the top decile and short the bottom decile. Otherwise
        long-only top decile.
    top_decile:
        Number of deciles to use for the long leg (default 1 = top 10%).
    rebalance_freq:
        Pandas offset string for rebalancing frequency (default ``"ME"`` month-end).
    """

    def __init__(
        self,
        factor: Factor,
        prices: pd.DataFrame,
        transaction_cost_bps: float = 10.0,
        long_short: bool = True,
        top_decile: int = 1,
        rebalance_freq: str = "ME",
    ) -> None:
        self.factor = factor
        self.prices = prices.copy().sort_index()
        if "close" not in self.prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")
        self.transaction_cost_bps = transaction_cost_bps
        self.long_short = long_short
        self.top_decile = top_decile
        self.rebalance_freq = rebalance_freq

    def run(self) -> BacktestResult:
        """Run the backtest and return a ``BacktestResult``."""
        scores = self.factor(self.prices)
        scores = scores.dropna()
        if scores.empty:
            raise ValueError("No factor scores could be computed")

        dates = self.prices.index.get_level_values("date").unique().sort_values()
        freq = self.rebalance_freq[0] if self.rebalance_freq else "M"
        rebalance_dates = (
            pd.DatetimeIndex(dates)
            .to_period(freq)
            .to_timestamp(how="end")
            .normalize()
            .unique()
        )
        rebalance_dates = rebalance_dates[rebalance_dates.isin(dates)]

        portfolio = Portfolio()
        positions_records: list[dict[str, object]] = []

        for i, date in enumerate(dates):
            date_str = date.strftime("%Y-%m-%d")
            day_prices = self.prices.loc[
                self.prices.index.get_level_values("date") == date
            ]
            price_map = day_prices.reset_index().set_index("symbol")["close"].to_dict()

            is_rebalance = date in rebalance_dates and i > 0
            if is_rebalance:
                target_weights = self._compute_target_weights(scores, date)
                portfolio.set_target_weights(
                    date=date_str,
                    target_weights=target_weights,
                    prices=price_map,
                    transaction_cost_bps=self.transaction_cost_bps,
                )
                positions_records.append(
                    {"date": date_str, **portfolio.current_weights(price_map)}
                )

            portfolio.record_nav(date_str, price_map)

        nav = pd.Series(
            dict(portfolio.nav_history),
            name="nav",
        )
        nav.index = pd.to_datetime(nav.index)
        returns = nav.pct_change().dropna()
        turnover = self._compute_turnover(portfolio.trades, nav)

        positions = pd.DataFrame(positions_records)
        if not positions.empty:
            positions["date"] = pd.to_datetime(positions["date"])
            positions = positions.set_index("date")

        trades = pd.DataFrame(
            [
                {
                    "date": t.date,
                    "symbol": t.symbol,
                    "old_shares": t.old_shares,
                    "new_shares": t.new_shares,
                    "price": t.price,
                    "portfolio_value": t.portfolio_value,
                }
                for t in portfolio.trades
            ]
        )

        from factor_forge.analytics.metrics import compute_metrics

        metrics = compute_metrics(returns)
        metrics["turnover_mean"] = turnover.mean() if not turnover.empty else 0.0
        metrics["turnover_annual"] = turnover.sum() if not turnover.empty else 0.0

        return BacktestResult(
            nav=nav,
            returns=returns,
            positions=positions,
            trades=trades,
            turnover=turnover,
            metrics=metrics,
            factor_name=self.factor.name,
            long_short=self.long_short,
        )

    def _compute_target_weights(
        self, scores: pd.Series, as_of: pd.Timestamp
    ) -> dict[str, float]:
        """Compute equal-weight target weights for the rebalance date."""
        mask = scores.index.get_level_values("date") == as_of
        day_scores = scores[mask].reset_index()
        if day_scores.empty:
            return {}
        day_scores["symbol"] = day_scores["symbol"]
        day_scores = day_scores.dropna()
        n = len(day_scores)
        if n == 0:
            return {}

        day_scores["rank"] = day_scores[scores.name].rank(
            ascending=False, method="average"
        )
        top_n = max(1, int(np.ceil(n * self.top_decile / 10)))
        top_symbols = day_scores.nsmallest(top_n, "rank")["symbol"].tolist()
        bottom_symbols = (
            day_scores.nlargest(top_n, "rank")["symbol"].tolist()
            if self.long_short
            else []
        )

        weight = 1.0 / top_n if top_n else 0.0
        weights: dict[str, float] = {}
        for sym in top_symbols:
            weights[sym] = weight
        if self.long_short:
            for sym in bottom_symbols:
                weights[sym] = -weight
        return weights

    def _compute_turnover(self, trades: Sequence[Trade], nav: pd.Series) -> pd.Series:
        """Compute per-rebalance turnover as half the sum of absolute weight changes."""
        if not trades:
            return pd.Series(dtype=float)
        df = pd.DataFrame(
            [
                {
                    "date": t.date,
                    "turnover": abs(t.new_shares - t.old_shares)
                    * t.price
                    / t.portfolio_value,
                }
                for t in trades
            ]
        )
        if df.empty:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date")["turnover"].sum()
