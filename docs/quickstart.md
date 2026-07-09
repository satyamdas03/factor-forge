# Quickstart

## Installation

```bash
git clone https://github.com/satyamdas03/factor-forge.git
cd factor-forge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run a single-factor backtest

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

## Compare all factors

```bash
factor-forge --factor all --start 2015-01-01 --end 2024-01-01 --top-n 500
```

## Next steps

- Read the [Factors](factors.md) guide to add your own.
- Explore the [API Reference](api.md) for detailed module docs.
- Read the [research blog](blog/factor-zoo-2020s.md) for empirical findings.
