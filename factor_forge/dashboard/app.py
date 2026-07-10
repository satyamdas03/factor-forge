"""Streamlit dashboard for Factor Forge backtests and research reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from factor_forge.backtest.engine import BacktestEngine, BacktestResult
from factor_forge.data.loader import DataLoader, generate_synthetic_prices
from factor_forge.factors.base import Factor
from factor_forge.factors.registry import FactorRegistry
from factor_forge.viz.research_report import (
    _compute_decile_returns,
    _compute_forward_returns,
    _cumulative_returns_fig,
    _decile_fig,
    _drawdown_fig,
    _ic_fig,
    _turnover_fig,
    generate_research_report,
)

if TYPE_CHECKING:
    pass


def load_prices(
    source: str,
    symbols: list[str] | None,
    start: str,
    end: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Load prices from synthetic or Polygon source."""
    if source == "polygon":
        if not api_key:
            raise ValueError("Polygon API key is required for Polygon data")
        loader = DataLoader(source="polygon", api_key=api_key)
        syms = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        prices = loader.load_eod(syms, start, end)
        loader.close()
    else:
        syms = symbols or ["A", "B", "C", "D", "E"]
        prices = generate_synthetic_prices(syms, start, end)
    return prices


def compute_factor_scores(factor: Factor, prices: pd.DataFrame) -> pd.DataFrame:
    """Compute scores for a single factor."""
    registry = FactorRegistry()
    registry.register(factor)
    return registry.compute(prices, names=[factor.name])


def run_backtest(
    factor: Factor,
    prices: pd.DataFrame,
    cost_bps: float,
    long_short: bool,
) -> BacktestResult:
    """Run the decile backtest engine."""
    engine = BacktestEngine(
        factor=factor,
        prices=prices,
        transaction_cost_bps=cost_bps,
        long_short=long_short,
    )
    return engine.run()


def compute_analytics(
    result: BacktestResult,
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    forward_horizon: int = 1,
) -> dict[str, Any]:
    """Compute IC, decile returns, half-life, and correlation metadata."""
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

    primary = (
        scores[result.factor_name]
        if result.factor_name in scores.columns
        else scores.iloc[:, 0]
    )
    fwd = _compute_forward_returns(prices, horizon=forward_horizon)
    ic = compute_ic(primary, fwd, method="spearman")
    return {
        "ic": ic,
        "ic_summary": compute_ic_summary(ic),
        "rolling_ic": compute_rolling_ic(primary, fwd, window=12),
        "decile_returns": _compute_decile_returns(primary, fwd),
        "half_life": factor_half_life(primary, max_lag=12),
        "factor_corr": factor_correlation_matrix(scores)
        if len(scores.columns) > 1
        else None,
        "avg_corr": average_pairwise_correlation(factor_correlation_matrix(scores))
        if len(scores.columns) > 1
        else None,
    }


def _metric_cards(metrics: dict[str, float]) -> None:
    """Render metric cards using Streamlit."""
    import streamlit as st

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{metrics.get('total_return', 0.0):.2%}")
    c2.metric("CAGR", f"{metrics.get('cagr', 0.0):.2%}")
    c3.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
    c4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sortino", f"{metrics.get('sortino_ratio', 0.0):.2f}")
    c6.metric("Win Rate", f"{metrics.get('win_rate', 0.0):.2%}")
    c7.metric("Mean Turnover", f"{metrics.get('turnover_mean', 0.0):.2%}")
    c8.metric("Annual Vol", f"{metrics.get('annualized_volatility', 0.0):.2%}")


def _render_analytics_table(analytics: dict[str, Any]) -> None:
    """Render signal analytics as a table."""
    import streamlit as st

    ic_summary = analytics["ic_summary"]
    rows = {
        "Mean IC (Spearman)": f"{ic_summary.get('mean_ic', 0.0):.3f}",
        "ICIR": f"{ic_summary.get('icir', 0.0):.3f}",
        "IC t-stat": f"{ic_summary.get('t_stat', 0.0):.3f}",
        "Factor Half-life (periods)": f"{analytics['half_life']:.1f}",
    }
    if analytics.get("avg_corr") is not None:
        rows["Avg Pairwise Factor Correlation"] = f"{analytics['avg_corr']:.3f}"
    st.table(pd.DataFrame({"Value": list(rows.values())}, index=list(rows.keys())))


def render_app() -> None:
    """Render the Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(page_title="Factor Forge Dashboard", layout="wide")
    st.title("Factor Forge Dashboard")

    registry = FactorRegistry()
    registry.load_builtin()
    factor_names = registry.list_factors()

    with st.sidebar:
        st.header("Inputs")
        source = st.selectbox("Data source", ["synthetic", "polygon"])
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value="A,B,C,D,E" if source == "synthetic" else "AAPL,MSFT,GOOGL,AMZN,TSLA",
        )
        api_key = None
        if source == "polygon":
            api_key = st.text_input("Polygon API key", value="", type="password")
        start = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
        end = st.date_input("End date", value=pd.Timestamp("2024-01-01"))
        factor_name = st.selectbox("Factor", factor_names)
        cost_bps = st.slider(
            "Transaction cost (bps)", min_value=0.0, max_value=50.0, value=10.0
        )
        long_short = st.toggle("Long/Short", value=True)
        forward_horizon = st.number_input(
            "Forward horizon (periods)", min_value=1, max_value=63, value=1
        )
        run_button = st.button("Run backtest", type="primary")

    if not run_button:
        st.info("Configure inputs in the sidebar and click **Run backtest**.")
        return

    symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
    factor = registry.get(factor_name)

    try:
        with st.spinner("Loading prices..."):
            prices = load_prices(
                source=source,
                symbols=symbols,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                api_key=api_key,
            )
        if prices.empty:
            st.error("No price data loaded.")
            return

        with st.spinner("Computing factor and running backtest..."):
            scores = compute_factor_scores(factor, prices)
            result = run_backtest(factor, prices, cost_bps, long_short)
            analytics = compute_analytics(result, scores, prices, forward_horizon)

        st.success(
            f"Backtest complete: {len(result.nav):,} days, "
            f"{len(result.turnover)} rebalances"
        )

        st.header("Backtest Metrics")
        _metric_cards(result.metrics)

        st.header("Signal Analytics")
        _render_analytics_table(analytics)

        st.header("Cumulative Returns")
        st.plotly_chart(_cumulative_returns_fig(result), use_container_width=True)

        st.header("Drawdown")
        st.plotly_chart(_drawdown_fig(result), use_container_width=True)

        st.header("Decile Performance")
        st.plotly_chart(
            _decile_fig(analytics["decile_returns"]), use_container_width=True
        )

        st.header("Information Coefficient")
        st.plotly_chart(
            _ic_fig(analytics["ic"], analytics["rolling_ic"]), use_container_width=True
        )

        st.header("Turnover")
        st.plotly_chart(_turnover_fig(result.turnover), use_container_width=True)

        if analytics["factor_corr"] is not None:
            from factor_forge.viz.research_report import _factor_corr_fig

            st.header("Factor Correlation")
            st.plotly_chart(
                _factor_corr_fig(analytics["factor_corr"]), use_container_width=True
            )

        with st.expander("Download research report"):
            report_path = generate_research_report(
                result=result,
                prices=prices,
                scores=scores,
                output_path="/tmp/factor_forge_dashboard_report.html",
                factor=factor,
                forward_horizon=forward_horizon,
            )
            html_bytes = report_path.read_bytes()
            st.download_button(
                label="Download HTML report",
                data=html_bytes,
                file_name=f"report_{factor_name}.html",
                mime="text/html",
            )

    except Exception as exc:  # noqa: BLE001
        st.error(f"Error: {exc}")


if __name__ == "__main__":
    render_app()
