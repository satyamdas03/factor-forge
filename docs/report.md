# Research Reports

Factor Forge can generate **self-contained interactive HTML research reports** that embed backtest metrics, signal analytics, and Plotly charts.

## Installation

Reports require `plotly`, included in the `viz` extra:

```bash
pip install "factor-forge-quant[viz]"
```

## Generate a report from the CLI

```bash
ff-report --factor momentum_12_1 \
          --start 2015-01-01 \
          --end 2024-01-01 \
          --symbols AAPL,MSFT,GOOGL,AMZN,TSLA \
          --source polygon \
          --cost-bps 10.0 \
          --output reports
```

With synthetic data:

```bash
ff-report --factor momentum_12_1 \
          --start 2020-01-01 \
          --end 2022-12-31 \
          --symbols A,B,C,D,E \
          --source synthetic \
          --output reports
```

The command writes `reports/report_{factor}.html`.

## Report contents

Every report includes:

- **Cover metadata**: factor name, category, inputs, backtest period, universe size, rebalances.
- **Backtest metrics**: total return, CAGR, Sharpe, Sortino, max drawdown, win rate, mean turnover, annual volatility.
- **Signal analytics**: mean IC, ICIR, IC t-stat, factor half-life, average pairwise factor correlation (when multiple factors are supplied).
- **Interactive charts**:
  - Cumulative returns (factor vs optional benchmark).
  - Drawdown underwater curve.
  - Average forward return by factor decile.
  - Information coefficient over time with rolling mean.
  - Portfolio turnover per rebalance.
  - Factor rank correlation heatmap (when multiple factors are supplied).

## Generate a report from Python

```python
from factor_forge import BacktestEngine
from factor_forge.data.loader import generate_synthetic_prices
from factor_forge.factors.registry import FactorRegistry
from factor_forge.viz.research_report import generate_research_report

prices = generate_synthetic_prices(["A", "B", "C", "D", "E"], "2020-01-01", "2022-12-31")
registry = FactorRegistry()
registry.load_builtin()
factor = registry.get("momentum_12_1")
scores = registry.compute(prices, names=["momentum_12_1"])

engine = BacktestEngine(factor=factor, prices=prices, transaction_cost_bps=10.0)
result = engine.run()

generate_research_report(
    result=result,
    prices=prices,
    scores=scores,
    output_path="report_momentum.html",
    factor=factor,
)
```
