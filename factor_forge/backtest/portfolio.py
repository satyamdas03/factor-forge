"""Simple portfolio bookkeeping for backtests."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trade:
    """Record of a single portfolio trade."""

    date: str
    symbol: str
    old_shares: float
    new_shares: float
    price: float
    portfolio_value: float


@dataclass
class Portfolio:
    """Track cash, share positions, NAV, and trades for a backtest.

    Positions are stored as share counts. Transaction costs are applied when
    share counts change. Dollar neutrality for long/short portfolios is enforced
    by allowing negative share counts (short positions).
    """

    initial_capital: float = 1_000_000.0
    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> shares
    trades: list[Trade] = field(default_factory=list)
    nav_history: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_capital)

    def nav(self, prices: dict[str, float]) -> float:
        """Current net asset value using the provided prices."""
        total = self.cash
        for symbol, shares in self.positions.items():
            price = prices.get(symbol, 0.0)
            total += shares * price
        return total

    def record_nav(self, date: str, prices: dict[str, float]) -> None:
        """Append current NAV to history."""
        self.nav_history.append((date, self.nav(prices)))

    def set_target_weights(
        self,
        date: str,
        target_weights: dict[str, float],
        prices: dict[str, float],
        transaction_cost_bps: float = 10.0,
    ) -> None:
        """Rebalance portfolio to ``target_weights`` at ``date``.

        Parameters
        ----------
        date:
            Rebalance date string.
        target_weights:
            Mapping from symbol to target weight (negative for short).
        prices:
            Mapping from symbol to execution price.
        transaction_cost_bps:
            Cost in basis points per dollar traded (both sides).
        """
        nav = self.nav(prices)
        if nav <= 0:
            return

        all_symbols = set(target_weights.keys()) | set(self.positions.keys())
        for symbol in all_symbols:
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue

            old_shares = self.positions.get(symbol, 0.0)
            new_shares = (target_weights.get(symbol, 0.0) * nav) / price
            delta_shares = new_shares - old_shares
            traded_value = abs(delta_shares) * price
            cost = traded_value * transaction_cost_bps / 10_000

            self.cash -= delta_shares * price + cost
            if abs(new_shares) < 1e-12:
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol] = new_shares

            self.trades.append(
                Trade(
                    date=date,
                    symbol=symbol,
                    old_shares=old_shares,
                    new_shares=new_shares,
                    price=price,
                    portfolio_value=nav,
                )
            )

    def current_weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Return current portfolio weights."""
        nav = self.nav(prices)
        if nav == 0:
            return {}
        return {
            sym: (shares * prices.get(sym, 0.0)) / nav
            for sym, shares in self.positions.items()
        }
