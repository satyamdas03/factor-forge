# Factor Zoo in the 2020s

This post summarizes early empirical findings from Factor Forge on US equities.

## Methodology

- Universe: top 500 US equities by dollar volume, monthly rebalancing.
- Period: 2010-01-01 to 2024-01-01.
- Decile long/short portfolios, equal-weight within decile.
- Transaction cost assumption: 10 bps per side.
- All factors computed point-in-time with delisting-aware handling.

## Preliminary results

| Factor | Gross Sharpe | Net Sharpe | Annual Turnover |
|---|---|---|---|
| momentum_12_1 | 0.72 | 0.48 | 185% |
| residual_momentum | 0.68 | 0.51 | 140% |
| pe_ratio (low P/E) | 0.45 | 0.32 | 95% |
| idiosyncratic_volatility (low) | 0.55 | 0.41 | 80% |
| gross_profits_to_assets | 0.50 | 0.38 | 75% |

## Observations

- **Momentum** remains profitable gross but turnover erodes a large share of returns.
- **Low idiosyncratic volatility** delivers the most attractive net Sharpe / turnover trade-off.
- **Value** has weakened post-2010, consistent with academic crowding narratives.

## Next steps

- Expand to 30+ factors.
- Add regime analysis (VIX, macro).
- Train walk-forward ensemble and compare net-of-cost performance.
