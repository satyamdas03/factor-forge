"""Tests for the CLI entry point."""

from factor_forge.cli.run_backtest import main


def test_cli_synthetic_single_factor() -> None:
    code = main(
        [
            "--factor",
            "momentum_12_1",
            "--start",
            "2020-01-01",
            "--end",
            "2021-12-31",
            "--source",
            "synthetic",
            "--symbols",
            "A,B,C,D,E",
            "--cost-bps",
            "0.0",
        ]
    )
    assert code == 0


def test_cli_all_factors() -> None:
    code = main(
        [
            "--factor",
            "all",
            "--start",
            "2020-01-01",
            "--end",
            "2021-12-31",
            "--source",
            "synthetic",
            "--symbols",
            "A,B,C,D,E",
        ]
    )
    assert code == 0


def test_cli_polygon_without_key() -> None:
    code = main(
        [
            "--source",
            "polygon",
            "--factor",
            "momentum_12_1",
            "--start",
            "2020-01-01",
            "--end",
            "2021-12-31",
        ]
    )
    assert code == 1
