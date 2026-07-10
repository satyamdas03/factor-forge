# Factor Forge

Factor Forge is a modern, reproducible engine for replicating and stress-testing cross-sectional equity factors.

It is designed for quant researchers, students, and job candidates who want to answer:

> **"Does this factor still work after costs, delisting, and look-ahead bias?"**

## What it does

- **Data pipeline** — loads EOD prices and fundamentals from Polygon.io or Databento with survivorship-aware, point-in-time handling.
- **Factor library** — clean, composable implementations of 15+ classic factors.
- **Backtest engine** — monthly decile rebalancing with transaction costs and slippage.
- **Analytics suite** — metrics, IC, decay, turnover, regime splits, factor correlations.
- **ML ensemble** — walk-forward LightGBM factor combination with conformal prediction intervals.
- **Visualization + reports** — publication-ready plots and interactive HTML research reports.
- **Streamlit dashboard** — point-and-click factor exploration.

## Get started

See the [Quickstart](quickstart.md) to run your first backtest, the [Dashboard guide](dashboard.md) for the interactive UI, and the [Research Report guide](report.md) for generating HTML reports.

## License

MIT © Satyam Das
