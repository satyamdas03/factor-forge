"""Tests for the research report CLI."""

from factor_forge.cli.research_report import main


def test_report_cli_synthetic(tmp_path) -> None:
    output_dir = tmp_path / "reports"
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
            "--output",
            str(output_dir),
        ]
    )
    assert code == 0
    assert (output_dir / "report_momentum_12_1.html").exists()


def test_report_cli_polygon_without_key() -> None:
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
