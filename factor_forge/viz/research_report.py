"""Interactive HTML research report generation with embedded Plotly charts."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from factor_forge.backtest.engine import BacktestResult
    from factor_forge.factors.base import Factor


def _ensure_plotly() -> None:
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Research reports require plotly; install factor-forge[viz]"
        ) from exc


def _compute_forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Compute next-period forward returns aligned with the price index."""
    close = prices["close"]
    fwd = close.groupby(level="symbol").pct_change(periods=horizon).shift(-horizon)
    fwd.name = "forward_return"
    return fwd


def _compute_decile_returns(scores: pd.Series, forward_returns: pd.Series) -> pd.Series:
    """Average forward return per factor-score decile."""
    df = pd.DataFrame({"score": scores, "fwd": forward_returns}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    ranked = df.groupby(level="date")["score"].rank(method="first")
    df["decile"] = pd.qcut(ranked, 10, labels=False, duplicates="drop") + 1
    return df.groupby("decile")["fwd"].mean()


def _primary_scores(scores: pd.Series | pd.DataFrame, factor_name: str) -> pd.Series:
    if isinstance(scores, pd.DataFrame):
        if factor_name in scores.columns:
            return scores[factor_name]
        return scores.iloc[:, 0]
    return scores


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def _fmt_float(value: float) -> str:
    return f"{value:.3f}"


def _plotly_div(fig: Figure) -> str:
    return str(fig.to_html(full_html=False, include_plotlyjs="cdn"))


def _cumulative_returns_fig(
    result: BacktestResult, benchmark: pd.Series | None = None
) -> Figure:
    import plotly.graph_objects as go

    nav = result.nav
    fig = go.Figure()
    normalized = nav / nav.iloc[0]
    fig.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized,
            mode="lines",
            name=html.escape(result.factor_name),
        )
    )
    if benchmark is not None and not benchmark.empty:
        bench_aligned = benchmark.reindex(result.returns.index).dropna()
        if not bench_aligned.empty:
            bench_cum = (1 + bench_aligned).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=bench_cum.index,
                    y=bench_cum,
                    mode="lines",
                    name="Benchmark",
                )
            )
    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        hovermode="x unified",
    )
    return fig


def _drawdown_fig(result: BacktestResult) -> Figure:
    import plotly.graph_objects as go

    cum = (1 + result.returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line={"color": "red"},
        )
    )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
    )
    return fig


def _decile_fig(decile_returns: pd.Series) -> Figure:
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = ["red" if v < 0 else "green" for v in decile_returns.values]
    fig.add_trace(
        go.Bar(
            x=[str(d) for d in decile_returns.index],
            y=decile_returns.values,
            marker={"color": colors},
            name="Decile return",
        )
    )
    fig.update_layout(
        title="Average Forward Return by Factor Decile",
        xaxis_title="Decile (1 = lowest score, 10 = highest)",
        yaxis_title="Average Forward Return",
    )
    return fig


def _ic_fig(ic: pd.Series, rolling_ic: pd.Series) -> Figure:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ic.index,
            y=ic.values,
            mode="lines",
            name="IC",
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_ic.index,
            y=rolling_ic.values,
            mode="lines",
            name="Rolling IC (12)",
            line={"width": 3},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Information Coefficient",
        xaxis_title="Date",
        yaxis_title="IC",
        hovermode="x unified",
    )
    return fig


def _turnover_fig(turnover: pd.Series) -> Figure:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=turnover.index,
            y=turnover.values,
            mode="lines+markers",
            name="Turnover",
        )
    )
    fig.update_layout(
        title="Portfolio Turnover",
        xaxis_title="Date",
        yaxis_title="Turnover (% of portfolio)",
    )
    return fig


def _factor_corr_fig(corr: pd.DataFrame) -> Figure:
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.columns),
            zmid=0,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(
        title="Factor Rank Correlation Matrix",
        xaxis={"side": "bottom"},
    )
    return fig


def generate_research_report(
    result: BacktestResult,
    prices: pd.DataFrame,
    scores: pd.Series | pd.DataFrame,
    output_path: str | Path,
    factor: Factor | None = None,
    benchmark: pd.Series | None = None,
    forward_horizon: int = 1,
) -> Path:
    """Generate a self-contained interactive HTML research report.

    Parameters
    ----------
    result:
        Backtest result to report on.
    prices:
        Long-form price DataFrame used for the backtest.
    scores:
        Factor scores (Series for a single factor, DataFrame for multiple).
    output_path:
        Destination HTML file path.
    factor:
        Optional factor definition for metadata.
    benchmark:
        Optional benchmark return series aligned with ``result.returns``.
    forward_horizon:
        Forward-return horizon for IC and decile analytics.

    Returns
    -------
    Path to the written HTML report.
    """
    _ensure_plotly()

    from factor_forge.analytics.correlation import (
        average_pairwise_correlation,
        factor_correlation_matrix,
    )
    from factor_forge.analytics.decay import factor_half_life
    from factor_forge.analytics.factor_ic import (
        compute_ic,
        compute_ic_summary,
        compute_rolling_ic,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primary = _primary_scores(scores, result.factor_name)
    fwd = _compute_forward_returns(prices, horizon=forward_horizon)
    ic = compute_ic(primary, fwd, method="spearman")
    ic_summary = compute_ic_summary(ic)
    rolling_ic = compute_rolling_ic(primary, fwd, window=12)
    decile_returns = _compute_decile_returns(primary, fwd)
    half_life = factor_half_life(primary, max_lag=12)

    corr_div = ""
    avg_corr: float | None = None
    if isinstance(scores, pd.DataFrame) and len(scores.columns) > 1:
        corr = factor_correlation_matrix(scores)
        avg_corr = average_pairwise_correlation(corr)
        corr_div = f'<div class="chart">{_plotly_div(_factor_corr_fig(corr))}</div>'

    metrics = result.metrics
    start_date = result.nav.index.min().strftime("%Y-%m-%d")
    end_date = result.nav.index.max().strftime("%Y-%m-%d")
    n_obs = len(prices)
    n_symbols = prices.index.get_level_values("symbol").nunique()
    n_rebalances = len(result.turnover)

    factor_meta = ""
    if factor is not None:
        factor_meta = (
            f"<p><strong>Category:</strong> {factor.category.value}<br>"
            f"<strong>Inputs:</strong> {', '.join(factor.inputs)}</p>"
        )

    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Factor Forge Research Report — {html.escape(result.factor_name)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 24px; color: #1a1a1a; background: #fff; }}
    h1, h2, h3 {{ color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .metric-card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; text-align: center; background: #fafafa; }}
    .metric-label {{ font-size: 0.85rem; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-value {{ font-size: 1.6rem; font-weight: 700; margin-top: 6px; }}
    .chart {{ margin: 32px 0; }}
    .section {{ margin-top: 40px; }}
    .footer {{ margin-top: 48px; font-size: 0.85rem; color: #666; border-top: 1px solid #eee; padding-top: 16px; }}
  </style>
</head>
<body>
  <h1>Factor Forge Research Report</h1>
  <p><strong>Factor:</strong> {html.escape(result.factor_name)} ({"Long/Short" if result.long_short else "Long Only"})</p>
  {factor_meta}
  <p><strong>Period:</strong> {start_date} → {end_date}</p>
  <p><strong>Universe:</strong> {n_symbols:,} symbols | {n_obs:,} price observations | {n_rebalances} rebalances</p>

  <div class="section">
    <h2>Backtest Metrics</h2>
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-label">Total Return</div><div class="metric-value">{_fmt_pct(metrics.get("total_return", 0.0))}</div></div>
      <div class="metric-card"><div class="metric-label">CAGR</div><div class="metric-value">{_fmt_pct(metrics.get("cagr", 0.0))}</div></div>
      <div class="metric-card"><div class="metric-label">Sharpe</div><div class="metric-value">{_fmt_float(metrics.get("sharpe_ratio", 0.0))}</div></div>
      <div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-value">{_fmt_pct(metrics.get("max_drawdown", 0.0))}</div></div>
      <div class="metric-card"><div class="metric-label">Sortino</div><div class="metric-value">{_fmt_float(metrics.get("sortino_ratio", 0.0))}</div></div>
      <div class="metric-card"><div class="metric-label">Win Rate</div><div class="metric-value">{_fmt_pct(metrics.get("win_rate", 0.0))}</div></div>
      <div class="metric-card"><div class="metric-label">Mean Turnover</div><div class="metric-value">{_fmt_pct(metrics.get("turnover_mean", 0.0))}</div></div>
      <div class="metric-card"><div class="metric-label">Annual Vol</div><div class="metric-value">{_fmt_pct(metrics.get("annualized_volatility", 0.0))}</div></div>
    </div>
  </div>

  <div class="section">
    <h2>Signal Analytics</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Mean IC (Spearman)</td><td>{_fmt_float(ic_summary.get("mean_ic", 0.0))}</td></tr>
      <tr><td>ICIR</td><td>{_fmt_float(ic_summary.get("icir", 0.0))}</td></tr>
      <tr><td>IC t-stat</td><td>{_fmt_float(ic_summary.get("t_stat", 0.0))}</td></tr>
      <tr><td>Factor Half-life (periods)</td><td>{_fmt_float(half_life)}</td></tr>
      {f"<tr><td>Avg Pairwise Factor Correlation</td><td>{_fmt_float(avg_corr)}</td></tr>" if avg_corr is not None else ""}
    </table>
  </div>

  <div class="section">
    <h2>Cumulative Returns</h2>
    <div class="chart">{_plotly_div(_cumulative_returns_fig(result, benchmark))}</div>
  </div>

  <div class="section">
    <h2>Drawdown</h2>
    <div class="chart">{_plotly_div(_drawdown_fig(result))}</div>
  </div>

  <div class="section">
    <h2>Decile Performance</h2>
    <div class="chart">{_plotly_div(_decile_fig(decile_returns))}</div>
  </div>

  <div class="section">
    <h2>Information Coefficient</h2>
    <div class="chart">{_plotly_div(_ic_fig(ic, rolling_ic))}</div>
  </div>

  <div class="section">
    <h2>Turnover</h2>
    <div class="chart">{_plotly_div(_turnover_fig(result.turnover))}</div>
  </div>

  {corr_div}

  <div class="footer">
    Generated by Factor Forge v0.5.0 — {html.escape(result.factor_name)} research report.
  </div>
</body>
</html>
"""

    output_path.write_text(report, encoding="utf-8")
    return output_path
