# Factor Forge

[![CI](https://github.com/satyamdas03/factor-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/satyamdas03/factor-forge/actions/workflows/ci.yml)
[![Docs](https://github.com/satyamdas03/factor-forge/actions/workflows/docs.yml/badge.svg)](https://satyamdas03.github.io/factor-forge/)
[![PyPI](https://img.shields.io/pypi/v/factor-forge.svg)](https://pypi.org/project/factor-forge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A modern, reproducible factor zoo replication engine for cross-sectional equity research.**

Factor Forge takes the academic "factor zoo" — momentum, value, quality, low-volatility, investment, profitability — and makes it runnable, honest, and ML-ready. It handles transaction costs, survivorship bias, look-ahead bias, factor decay, and walk-forward machine learning.

> **Mission:** Let any researcher or student answer *"Does this factor still work after costs?"* with net-of-cost Sharpe, turnover analysis, and uncertainty-aware ensembles.

---

## Why this matters

- **Quantopian is gone** and `zipline` / `pyfolio` are unmaintained.
- Most public factor code reports **gross returns**, ignores **transaction costs**, and suffers from **look-ahead bias**.
- Factor Forge reports **net-of-cost Sharpe**, **turnover**, **capacity**, and **factor decay** in a clean, extensible Python package.

---

## What you can do

- Compute **15+ classic factors** with a uniform cross-sectional API.
- Build **decile long/short and long-only portfolios** with monthly rebalancing.
- Apply a **transaction-cost and slippage model** to every trade.
- Measure **information coefficient (IC)**, **factor decay**, **turnover**, and **regime performance**.
- Train a **walk-forward LightGBM ensemble** that combines factors without peeking.
- Add **conformal prediction intervals** for return forecasts.
- Generate **publication-ready plots** and an **MkDocs-powered docs site**.

---

## Quickstart

```bash
git clone https://github.com/satyamdas03/factor-forge.git
cd factor-forge
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run a backtest on a single factor using cached or Polygon data:

```bash
export POLYGON_API_KEY=your_key_here
factor-forge --factor momentum_12_1 --start 2015-01-01 --end 2024-01-01 --top-n 100
```

Or from Python:

```python
from factor_forge import BacktestEngine, DataLoader
from factor_forge.factors.registry import FactorRegistry

loader = DataLoader(source="polygon", api_key="your_key")
prices = loader.load_eod(["AAPL", "MSFT", "GOOGL"], start="2015-01-01", end="2024-01-01")

registry = FactorRegistry()
registry.load_builtin()
factor = registry.get("momentum_12_1")

engine = BacktestEngine(factor=factor, prices=prices, transaction_cost_bps=10.0)
result = engine.run()
print(result.metrics)
```

---

## Factor library

| Category | Factors |
|---|---|
| Momentum | `momentum_12_1`, `momentum_52_high`, `residual_momentum` |
| Value | `pe_ratio`, `pb_ratio` |
| Quality | `roe`, `gross_margin`, `accruals` |
| Low volatility | `idiosyncratic_volatility`, `beta`, `max_drawdown` |
| Investment | `asset_growth`, `capex_growth` |
| Profitability | `gross_profits_to_assets` |

Each factor is a pure function `prices / fundamentals → cross-sectional score`. Adding a new factor is one function + one registry entry.

---

## Architecture

```
factor_forge/
├── data/          # Data sources, loader, cache, survivorship handling
├── factors/       # Factor definitions and registry
├── backtest/      # Portfolio, cost model, decile engine
├── analytics/     # Metrics, IC, decay, turnover, regimes, correlations
├── ml/            # Walk-forward features, LightGBM ensemble, conformal prediction
├── viz/           # Publication-ready plots
├── cli/           # Command-line tools
└── docs/          # MkDocs site + blog
```

---

## Roadmap

| Version | Goal | Status |
|---|---|---|
| v0.1.0 | Data pipeline + 5 core factors + decile backtest | ✅ |
| v0.2.0 | 15+ factors + transaction costs + analytics | ✅ |
| v0.3.0 | Factor decay, turnover, regime analysis | ✅ |
| v0.4.0 | ML ensemble + conformal prediction | ✅ |
| v0.5.0 | Dashboard + research report generation | 🔲 |

---

## Documentation

Full documentation, API reference, and research blog are at:

**https://satyamdas03.github.io/factor-forge/**

---

## Interview talking points

- *"Most retail factor backtests report gross returns. Factor Forge reports net-of-cost Sharpe and turnover-adjusted returns."*
- *"I handle delisting and point-in-time fundamentals to avoid look-ahead bias."*
- *"I found momentum still works but turnover is high; low-vol is crowded and decayed post-2020."*
- *"I combined factors with a walk-forward LightGBM ensemble and added conformal prediction intervals for risk."*

---

## License

MIT © Satyam Das
