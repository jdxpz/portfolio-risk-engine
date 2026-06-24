"""
Performance charts — cumulative return, drawdown, rolling Sharpe, attribution.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.contracts import RiskOutput
from charts.bbg_theme import apply_bbg, ORANGE, WHITE, RED, GREEN, GREY, ASSET_COLOURS


def cumulative_return_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> go.Figure:
    cumret = (1 + portfolio_returns).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumret.index, y=cumret.values * 100,
        mode="lines", name="PORTFOLIO",
        line=dict(color=ORANGE, width=2),
        fill="tozeroy",
        fillcolor="rgba(255,102,0,0.08)",
    ))
    if benchmark_returns is not None:
        bench = (1 + benchmark_returns).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=bench.index, y=bench.values * 100,
            mode="lines", name="BENCHMARK",
            line=dict(color="#4488FF", width=1.5, dash="dash"),
        ))
    apply_bbg(fig)
    fig.update_layout(title="CUMULATIVE RETURN (%)", yaxis_title="RETURN (%)", height=360)
    return fig


def drawdown_chart(drawdown_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values * 100,
        mode="lines", name="DRAWDOWN",
        line=dict(color=RED, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,51,51,0.15)",
    ))
    apply_bbg(fig)
    fig.update_layout(title="DRAWDOWN (%)", yaxis_title="DRAWDOWN (%)", height=240, showlegend=False)
    return fig


def rolling_sharpe_chart(rolling_sharpe: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=0,  line_color=GREY,  line_width=1)
    fig.add_hline(y=1,  line_dash="dash", line_color=GREEN, opacity=0.5,
                  annotation_text="SHARPE = 1",  annotation_font=dict(color=GREEN, size=9))
    fig.add_hline(y=-1, line_dash="dash", line_color=RED,   opacity=0.5,
                  annotation_text="SHARPE = -1", annotation_font=dict(color=RED,   size=9))
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode="lines", name="ROLLING SHARPE",
        line=dict(color="#4488FF", width=1.5),
    ))
    window = rolling_sharpe.name.split("_")[-1] if rolling_sharpe.name else ""
    apply_bbg(fig)
    fig.update_layout(
        title=f"ROLLING SHARPE RATIO  ({window} WINDOW)",
        yaxis_title="SHARPE RATIO", height=220, showlegend=False,
    )
    return fig


def attribution_bar_chart(attribution: dict[str, RiskOutput]) -> go.Figure:
    classes = list(attribution.keys())
    values  = [attribution[c].value * 100 for c in classes]
    colours = [GREEN if v >= 0 else RED for v in values]

    fig = go.Figure(go.Bar(
        y=classes,
        x=values,
        orientation="h",
        marker_color=colours,
        marker_line_color="#000000",
        marker_line_width=1,
        opacity=0.85,
        text=[f"{v:+.2f}%" for v in values],
        textposition="outside",
        textfont=dict(size=9, color=WHITE),
    ))
    fig.add_vline(x=0, line_color=GREY, line_width=1)
    apply_bbg(fig)
    fig.update_layout(
        title="P&L ATTRIBUTION BY ASSET CLASS  (ANNUALISED)",
        xaxis_title="ANNUALISED CONTRIBUTION (%)",
        height=340,
        showlegend=False,
    )
    return fig


def performance_summary_figure(
    portfolio_returns: pd.Series,
    drawdown_series: pd.Series,
    rolling_sharpe: pd.Series,
) -> go.Figure:
    """Three-panel figure: cumulative return / drawdown / rolling Sharpe."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["CUMULATIVE RETURN (%)", "DRAWDOWN (%)", "ROLLING SHARPE"],
        vertical_spacing=0.06,
        row_heights=[0.50, 0.25, 0.25],
    )

    cumret = (1 + portfolio_returns).cumprod() - 1

    fig.add_trace(go.Scatter(
        x=cumret.index, y=cumret.values * 100,
        line=dict(color=ORANGE, width=2), name="RETURN",
        fill="tozeroy", fillcolor="rgba(255,102,0,0.08)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown_series.index, y=drawdown_series.values * 100,
        line=dict(color=RED, width=1.5), name="DRAWDOWN",
        fill="tozeroy", fillcolor="rgba(255,51,51,0.12)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        line=dict(color="#4488FF", width=1.5), name="SHARPE",
    ), row=3, col=1)

    fig.add_hline(y=0, line_color=GREY, line_width=0.8, row=3, col=1)
    fig.add_hline(y=1, line_color=GREEN, line_width=0.8, line_dash="dash",
                  opacity=0.4, row=3, col=1)

    apply_bbg(fig)
    fig.update_layout(
        height=580,
        showlegend=False,
        title="PORTFOLIO PERFORMANCE SUMMARY",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
    )
    for ann in fig.layout.annotations:
        ann.font.color = ORANGE
        ann.font.size  = 10

    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1A1A1A", linecolor="#333333", row=i, col=1)
        fig.update_yaxes(gridcolor="#1A1A1A", linecolor="#333333", row=i, col=1)

    return fig
