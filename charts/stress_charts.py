"""
Stress charts — scenario P&L waterfall, MC loss distribution.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.stress_engine import ScenarioResult
from charts.bbg_theme import apply_bbg, ORANGE, WHITE, RED, GREEN, GREY, FACTOR_COLOURS


def scenario_pnl_chart(results: list[ScenarioResult]) -> go.Figure:
    """Grouped/stacked bars of scenario P&L decomposed by risk factor."""
    scenario_names = [r.scenario_name for r in results]
    factors = ["rates", "credit", "equity", "commodity"]

    fig = go.Figure()
    for factor in factors:
        fig.add_trace(go.Bar(
            name=factor.upper(),
            x=scenario_names,
            y=[r.factor_pnl.get(factor, 0) / 1_000 for r in results],
            marker_color=FACTOR_COLOURS[factor],
            marker_line_color="#000000",
            marker_line_width=1,
            opacity=0.85,
        ))

    # Total P&L markers
    fig.add_trace(go.Scatter(
        x=scenario_names,
        y=[r.total_pnl_usd / 1_000 for r in results],
        mode="markers+text",
        name="TOTAL P&L",
        marker=dict(symbol="diamond", size=10, color=WHITE, line=dict(color=ORANGE, width=1)),
        text=[f"${r.total_pnl_usd/1_000:+.0f}K" for r in results],
        textposition="top center",
        textfont=dict(size=9, color=WHITE),
    ))

    apply_bbg(fig)
    fig.update_layout(
        title="STRESS SCENARIO P&L — FACTOR DECOMPOSITION",
        barmode="relative",
        xaxis_title="SCENARIO",
        yaxis_title="P&L (USD THOUSANDS)",
        height=460,
    )
    return fig


def scenario_waterfall(result: ScenarioResult) -> go.Figure:
    """Waterfall chart for a single scenario — factor-by-factor P&L build."""
    factors = list(result.factor_pnl.keys())
    values  = [result.factor_pnl[f] / 1_000 for f in factors]
    total   = result.total_pnl_usd / 1_000

    fig = go.Figure(go.Waterfall(
        measure=["relative"] * len(factors) + ["total"],
        x=[f.upper() for f in factors] + ["TOTAL"],
        y=values + [total],
        connector=dict(line=dict(color=GREY, width=1, dash="dot")),
        decreasing=dict(marker=dict(color=RED, line=dict(color="#000000", width=1))),
        increasing=dict(marker=dict(color=GREEN, line=dict(color="#000000", width=1))),
        totals=dict(marker=dict(color=ORANGE, line=dict(color="#000000", width=1))),
        text=[f"${v*1000:+,.0f}" for v in values + [total]],
        textposition="outside",
        textfont=dict(size=9, color=WHITE),
    ))

    apply_bbg(fig)
    fig.update_layout(
        title=f"P&L WATERFALL — {result.scenario_name.upper()}",
        yaxis_title="P&L (USD THOUSANDS)",
        height=400,
    )
    return fig


def mc_loss_histogram(
    sim_returns: np.ndarray,
    portfolio_value: float,
    confidence_levels: list[float] = [0.95, 0.99],
) -> go.Figure:
    """MC loss distribution histogram with VaR / CVaR lines."""
    losses = -sim_returns * portfolio_value
    line_colours = {0.95: ORANGE, 0.99: RED}

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=losses / 1_000,
        nbinsx=120,
        name="SIMULATED P&L",
        marker_color="#4488FF",
        marker_line_color="#000000",
        marker_line_width=0.5,
        opacity=0.75,
    ))

    for conf in confidence_levels:
        var_val  = float(np.percentile(losses, conf * 100)) / 1_000
        tail     = losses[losses >= var_val * 1_000]
        cvar_val = float(tail.mean()) / 1_000 if len(tail) > 0 else var_val
        col      = line_colours.get(conf, ORANGE)

        fig.add_vline(
            x=var_val, line_dash="dash", line_color=col, line_width=1.5,
            annotation_text=f"VAR {int(conf*100)}%  ${var_val*1000:,.0f}",
            annotation_font=dict(color=col, size=9),
        )
        fig.add_vline(
            x=cvar_val, line_dash="dot", line_color=col, line_width=1,
            annotation_text=f"CVAR {int(conf*100)}%  ${cvar_val*1000:,.0f}",
            annotation_font=dict(color=col, size=9),
        )

    apply_bbg(fig)
    fig.update_layout(
        title=f"MONTE CARLO LOSS DISTRIBUTION — {len(sim_returns):,} PATHS  |  STUDENT-T DF=5",
        xaxis_title="PORTFOLIO LOSS (USD THOUSANDS)",
        yaxis_title="FREQUENCY",
        height=420,
    )
    return fig
