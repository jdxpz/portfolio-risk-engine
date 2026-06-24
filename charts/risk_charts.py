"""
Risk charts — VaR comparison, component VaR waterfall, MC loss distribution.
Zero business logic. Pure Plotly visualisation consuming RiskOutput objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.contracts import RiskOutput
from charts.bbg_theme import (
    apply_bbg, ORANGE, WHITE, RED, GREEN, GREY, DARK_GREY,
    ASSET_COLOURS, BBG_LAYOUT,
)


def var_comparison_chart(var_outputs: dict[str, list[RiskOutput]]) -> go.Figure:
    """
    Grouped bar chart comparing VaR and CVaR across all three methodologies
    at 95% and 99% confidence.
    """
    methods = ["historical", "parametric", "monte_carlo"]
    labels  = {"historical": "HISTORICAL", "parametric": "PARAMETRIC", "monte_carlo": "MONTE CARLO"}
    colours = {"historical": "#4488FF", "parametric": ORANGE, "monte_carlo": GREEN}
    confidences = [95, 99]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["VAR", "CVAR (EXPECTED SHORTFALL)"],
        horizontal_spacing=0.1,
    )

    for col_idx, metric_idx in enumerate([0, 1]):
        for method in methods:
            x_vals, y_vals = [], []
            for conf in confidences:
                key = f"{method}_{conf}"
                if key in var_outputs:
                    x_vals.append(f"{conf}%")
                    y_vals.append(var_outputs[key][metric_idx].value)

            fig.add_trace(
                go.Bar(
                    name=labels[method],
                    x=x_vals,
                    y=y_vals,
                    marker_color=colours[method],
                    marker_line_color="#000000",
                    marker_line_width=1,
                    opacity=0.9,
                    showlegend=(col_idx == 0),
                    legendgroup=method,
                    text=[f"${v:,.0f}" for v in y_vals],
                    textposition="outside",
                    textfont=dict(size=9, color="#CCCCCC"),
                ),
                row=1, col=col_idx + 1,
            )

    apply_bbg(fig)
    fig.update_layout(
        title="VAR / CVAR — THREE METHODOLOGIES | 95% AND 99% CONFIDENCE",
        barmode="group",
        height=420,
        yaxis_title="USD",
        yaxis2_title="USD",
    )
    for ann in fig.layout.annotations:
        ann.font.color = ORANGE
        ann.font.size  = 10
    return fig


def component_var_waterfall(
    component_outputs: list[RiskOutput],
    portfolio_df: pd.DataFrame,
) -> go.Figure:
    """Waterfall/bar chart of component VaR by position, coloured by asset class."""
    asset_class_map = dict(zip(portfolio_df["ticker"], portfolio_df["asset_class"]))
    total_var = sum(o.value for o in component_outputs)

    sorted_outputs = sorted(component_outputs, key=lambda o: -o.value)
    tickers = [o.metadata.get("ticker", "") for o in sorted_outputs]
    values  = [o.value for o in sorted_outputs]
    colours = [ASSET_COLOURS.get(asset_class_map.get(t, ""), GREY) for t in tickers]
    pct_labels = [
        f"${v:,.0f}<br>{o.metadata.get('pct_of_total_var', 0):.1%}"
        for v, o in zip(values, sorted_outputs)
    ]

    fig = go.Figure(go.Bar(
        x=tickers,
        y=values,
        marker_color=colours,
        marker_line_color="#000000",
        marker_line_width=1,
        text=pct_labels,
        textposition="outside",
        textfont=dict(size=9, color="#CCCCCC"),
    ))

    fig.add_hline(
        y=total_var,
        line_dash="dot",
        line_color=ORANGE,
        line_width=1.5,
        annotation_text=f"TOTAL VAR  ${total_var:,.0f}",
        annotation_position="right",
        annotation_font=dict(color=ORANGE, size=10),
    )

    apply_bbg(fig)
    fig.update_layout(
        title="COMPONENT VAR BY POSITION — EULER DECOMPOSITION",
        xaxis_title="TICKER",
        yaxis_title="USD",
        height=420,
        showlegend=False,
    )
    return fig


def loss_distribution_chart(
    sim_returns: np.ndarray,
    portfolio_value: float,
    var_outputs: list[RiskOutput],
    confidence: float = 0.95,
) -> go.Figure:
    """MC loss distribution histogram with VaR and CVaR lines."""
    losses = -sim_returns * portfolio_value

    var_val = next(
        (o.value for o in var_outputs if "Monte Carlo VaR" in o.metric_name
         and o.confidence_level == confidence), None,
    )
    cvar_val = next(
        (o.value for o in var_outputs if "Monte Carlo CVaR" in o.metric_name
         and o.confidence_level == confidence), None,
    )

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=losses,
        nbinsx=100,
        name="SIMULATED P&L",
        marker_color="#4488FF",
        marker_line_color="#000000",
        marker_line_width=0.5,
        opacity=0.8,
    ))

    if var_val is not None:
        fig.add_vline(x=var_val, line_dash="dash", line_color=ORANGE, line_width=1.5,
                      annotation_text=f"VAR {int(confidence*100)}%  ${var_val:,.0f}",
                      annotation_font=dict(color=ORANGE, size=9))
    if cvar_val is not None:
        fig.add_vline(x=cvar_val, line_dash="dot", line_color=RED, line_width=1.5,
                      annotation_text=f"CVAR {int(confidence*100)}%  ${cvar_val:,.0f}",
                      annotation_font=dict(color=RED, size=9))

    apply_bbg(fig)
    fig.update_layout(
        title=f"MONTE CARLO LOSS DISTRIBUTION — {len(sim_returns):,} SIMULATIONS",
        xaxis_title="PORTFOLIO LOSS (USD)",
        yaxis_title="FREQUENCY",
        height=420,
    )
    return fig
