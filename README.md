# Factor Forge

A modern, reproducible **factor zoo replication engine** for cross-sectional equity research.

> **Mission:** Make the academic factor zoo runnable, honest, and ML-ready — with transaction costs, survivorship-bias handling, factor decay analysis, and uncertainty-aware ensembles.

## Why this matters

Since Quantopian shut down, there is no clean open-source platform for retail/ student researchers to ask: *"Does this factor still work after costs?"* FactorForge answers that with net-of-cost Sharpe, turnover analysis, and walk-forward ML.

## What you can do with it

- Compute 30+ classic factors (momentum, value, quality, low-vol, investment, profitability, sentiment)
- Build decile long/short and long-only portfolios
- Account for transaction costs, slippage, and delisting
- Measure factor decay, turnover, capacity, and regime performance
- Train ML ensembles that combine factors walk-forward
- Generate publication-ready reports and an interactive dashboard

## Tech stack

- Python 3.11+ · pandas / polars · NumPy · statsmodels
- scikit-learn · LightGBM · XGBoost
- Conformal prediction (mapie)
- Polygon.io / Databento / OpenBB for data
- Matplotlib / Plotly · Streamlit / Next.js dashboard

## Quickstart (coming soon)

```bash
git clone https://github.com/satyamdas03/factor-forge.git
cd factor-forge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_backtest.py --factor momentum --start 2010-01-01
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| v0.1 | Data pipeline + 5 core factors | 🚧 In progress |
| v0.2 | Transaction costs + 20 factors | 🔲 Planned |
| v0.3 | Decay + regime analysis | 🔲 Planned |
| v0.4 | ML ensemble + conformal prediction | 🔲 Planned |
| v0.5 | Dashboard + research reports | 🔲 Planned |

## Project spec

See [`PROJECT_SPEC.txt`](PROJECT_SPEC.txt) for the full vision, market gap, build phases, and interview talking points.

## License

MIT © Satyam Das
