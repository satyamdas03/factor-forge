"""Command-line entry point for running factor backtests."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.data.loader import DataLoader
from factor_forge.factors.registry import FactorRegistry
from factor_forge.viz.reports import generate_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Factor Forge backtest")
    parser.add_argument(
        "--factor",
        type=str,
        default="momentum_12_1",
        help="Factor name to backtest or 'all' to compare all price-based factors",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-01-01",
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: use synthetic data)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["synthetic", "polygon"],
        help="Data source",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points per dollar traded",
    )
    parser.add_argument(
        "--long-short",
        action="store_true",
        default=True,
        help="Use long/short decile construction",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        default=False,
        help="Use long-only decile construction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to write report files",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the selected data source",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = FactorRegistry()
    registry.load_builtin()
    long_short = not args.long_only

    if args.source == "polygon":
        api_key = args.api_key or os.environ.get("POLYGON_API_KEY")
        if not api_key:
            print(
                "ERROR: Polygon API key required (set POLYGON_API_KEY or use --api-key)",
                file=sys.stderr,
            )
            return 1
        loader = DataLoader(source="polygon", api_key=api_key)
        symbols = (
            args.symbols.split(",")
            if args.symbols
            else ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        )
        prices = loader.load_eod(symbols, args.start, args.end)
        loader.close()
    else:
        if args.symbols:
            symbols = args.symbols.split(",")
        else:
            symbols = ["A", "B", "C", "D", "E"]
        from factor_forge.data.loader import generate_synthetic_prices

        prices = generate_synthetic_prices(symbols, args.start, args.end)

    if prices.empty:
        print("ERROR: No price data loaded", file=sys.stderr)
        return 1

    if args.factor == "all":
        factor_names = registry.list_factors()
        factor_names = [
            name
            for name in factor_names
            if set(registry.get(name).inputs) <= set(prices.columns)
        ]
    else:
        factor_names = [args.factor]

    results: list[dict[str, Any]] = []
    for name in factor_names:
        factor = registry.get(name)
        try:
            engine = BacktestEngine(
                factor=factor,
                prices=prices,
                transaction_cost_bps=args.cost_bps,
                long_short=long_short,
            )
            result = engine.run()
            results.append(
                {
                    "factor": name,
                    "sharpe": result.metrics["sharpe_ratio"],
                    "cagr": result.metrics["cagr"],
                    "max_dd": result.metrics["max_drawdown"],
                    "turnover": result.metrics["turnover_mean"],
                    "total_return": result.metrics["total_return"],
                }
            )
            print(
                f"{name:30s}  Sharpe={result.metrics['sharpe_ratio']: .3f}  CAGR={result.metrics['cagr']: .2%}  MaxDD={result.metrics['max_drawdown']: .2%}  Turnover={result.metrics['turnover_mean']: .2%}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{name:30s}  FAILED: {exc}", file=sys.stderr)

    if not results:
        print("No successful backtests", file=sys.stderr)
        return 1

    summary = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print("\nSummary:")
    print(summary.to_string(index=False))

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "summary.csv", index=False)
        if args.factor != "all":
            # Generate full report for single factor.
            factor = registry.get(args.factor)
            engine = BacktestEngine(
                factor=factor,
                prices=prices,
                transaction_cost_bps=args.cost_bps,
                long_short=long_short,
            )
            result = engine.run()
            report_path = generate_report(result, out_dir)
            print(f"Report written to {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
