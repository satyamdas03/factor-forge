# Dashboard

Factor Forge ships with an interactive **Streamlit** dashboard for exploring factors without writing code.

## Installation

Install the dashboard extra:

```bash
pip install "factor-forge-quant[dashboard]"
```

Or, if you are working from source:

```bash
pip install -e ".[dashboard]"
```

## Launch

Run the dashboard from the command line:

```bash
ff-dashboard
```

This opens a browser tab at `http://localhost:8501`.

## What you can do

- **Pick a data source**: synthetic prices for quick experiments, or Polygon.io for real market data.
- **Select symbols**: comma-separated tickers (e.g., `AAPL,MSFT,GOOGL`).
- **Choose a factor**: any built-in factor from the registry.
- **Set transaction costs**: adjust basis-points per dollar traded.
- **Toggle long/short vs long-only** decile construction.
- **Run the backtest** and see:
  - Net-of-cost performance metrics (total return, CAGR, Sharpe, max drawdown, etc.).
  - Interactive Plotly charts: cumulative returns, drawdown, decile performance, IC, turnover.
  - Signal analytics: mean IC, ICIR, factor half-life.
  - A downloadable self-contained HTML research report.

## Programmatic use

You can also import the dashboard helpers to build custom Streamlit apps:

```python
from factor_forge.dashboard.app import load_prices, compute_factor_scores, run_backtest
from factor_forge.factors.registry import FactorRegistry

registry = FactorRegistry()
registry.load_builtin()
factor = registry.get("momentum_12_1")

prices = load_prices("synthetic", ["A", "B", "C"], "2020-01-01", "2022-12-31")
scores = compute_factor_scores(factor, prices)
result = run_backtest(factor, prices, cost_bps=10.0, long_short=True)
print(result.metrics)
```
