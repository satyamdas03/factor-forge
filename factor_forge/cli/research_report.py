"""Command-line entry point for generating research reports."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from factor_forge.backtest.engine import BacktestEngine
from factor_forge.data.loader import DataLoader
from factor_forge.factors.registry import FactorRegistry
from factor_forge.viz.research_report import generate_research_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Factor Forge research report"
    )
    parser.add_argument(
        "--factor",
        type=str,
        default="momentum_12_1",
        help="Factor name to report on",
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
        help="Comma-separated list of symbols (default: synthetic A,B,C,D,E)",
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
        "--long-only",
        action="store_true",
        default=False,
        help="Use long-only decile construction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Directory to write the HTML report",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the selected data source",
    )
    parser.add_argument(
        "--forward-horizon",
        type=int,
        default=1,
        help="Forward-return horizon for IC and decile analytics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = FactorRegistry()
    registry.load_builtin()
    factor = registry.get(args.factor)
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
        symbols = args.symbols.split(",") if args.symbols else ["A", "B", "C", "D", "E"]
        from factor_forge.data.loader import generate_synthetic_prices

        prices = generate_synthetic_prices(symbols, args.start, args.end)

    if prices.empty:
        print("ERROR: No price data loaded", file=sys.stderr)
        return 1

    scores = registry.compute(prices, names=[args.factor])
    engine = BacktestEngine(
        factor=factor,
        prices=prices,
        transaction_cost_bps=args.cost_bps,
        long_short=long_short,
    )
    result = engine.run()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"report_{args.factor}.html"
    generate_research_report(
        result=result,
        prices=prices,
        scores=scores,
        output_path=report_path,
        factor=factor,
        forward_horizon=args.forward_horizon,
    )
    print(f"Research report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
