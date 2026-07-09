# Factor Library

Each factor in Factor Forge is a pure function that maps market data to a cross-sectional score.

## Built-in factors

| Category | Name | Description |
|---|---|---|
| Momentum | `momentum_12_1` | 12-month return excluding last month |
| Momentum | `momentum_52_high` | Current price / 52-week high |
| Momentum | `residual_momentum` | Momentum residualized against market return |
| Value | `pe_ratio` | Price / earnings |
| Value | `pb_ratio` | Price / book |
| Quality | `roe` | Return on equity |
| Quality | `gross_margin` | Gross profit / revenue |
| Quality | `accruals` | Accruals anomaly proxy |
| Low vol | `idiosyncratic_volatility` | Residual return volatility |
| Low vol | `beta` | Rolling CAPM beta |
| Low vol | `max_drawdown` | Maximum drawdown over lookback |
| Investment | `asset_growth` | Year-over-year asset growth |
| Investment | `capex_growth` | Year-over-year capex growth |
| Profitability | `gross_profits_to_assets` | Gross profits / total assets |

## Adding a custom factor

```python
from factor_forge.factors.base import Factor, FactorCategory

my_factor = Factor(
    name="earnings_yield",
    category=FactorCategory.VALUE,
    inputs=["close", "earnings"],
    compute=lambda df: df["earnings"] / df["close"],
)

registry = FactorRegistry()
registry.register(my_factor)
```

## Look-ahead discipline

All built-in factors use only information available strictly before the rebalance date. When fundamentals are unavailable, the engine falls back to the most recent announced value with a lag.
