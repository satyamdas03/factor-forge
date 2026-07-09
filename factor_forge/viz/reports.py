"""Static HTML report generation."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factor_forge.backtest.engine import BacktestResult


def generate_report(result: BacktestResult, output_dir: Path) -> Path:
    """Generate a simple HTML report with cumulative return plot.

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Report generation requires matplotlib; install factor-forge[viz]"
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    cum = (1 + result.returns).cumprod()
    ax.plot(cum.index, cum, label=result.factor_name)
    ax.set_title(f"Cumulative returns: {result.factor_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    metrics_html = "\n".join(
        f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" for k, v in result.metrics.items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head><title>Factor Forge Report: {result.factor_name}</title></head>
<body>
<h1>Factor Forge Backtest Report</h1>
<h2>{result.factor_name} ({"Long/Short" if result.long_short else "Long Only"})</h2>
<img src="data:image/png;base64,{img_b64}" alt="Cumulative returns">
<h3>Metrics</h3>
<table border="1">
<tr><th>Metric</th><th>Value</th></tr>
{metrics_html}
</table>
</body>
</html>
"""

    report_path = output_dir / f"report_{result.factor_name}.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
