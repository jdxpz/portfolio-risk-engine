"""
Rates charts — KRD heatmap, DV01 bar, yield curve.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.contracts import RiskOutput
from charts.bbg_theme import apply_bbg, ORANGE, WHITE, GREEN, RED, GREY, DARK_GREY

KRD_BUCKETS = ["2Y", "5Y", "10Y", "30Y"]


def dv01_bar_chart(dv01_outputs: list[RiskOutput]) -> go.Figure:
    """Horizontal bar chart of DV01 per bond position."""
    position_outputs = [o for o in dv01_outputs if "Portfolio" not in o.metric_name]
    tickers  = [o.metadata.get("ticker", "") for o in position_outputs]
    values   = [o.value for o in position_outputs]
    port_dv01 = next((o.value for o in dv01_outputs if "Portfolio" in o.metric_name), 0.0)

    fig = go.Figure(go.Bar(
        y=tickers,
        x=values,
        orientation="h",
        marker_color=ORANGE,
        marker_line_color="#000000",
        marker_line_width=1,
        opacity=0.85,
        text=[f"${v:,.4f}/bp" for v in values],
        textposition="outside",
        textfont=dict(size=9, color="#CCCCCC"),
    ))
    fig.add_vline(
        x=port_dv01, line_dash="dot", line_color=WHITE, line_width=1,
        annotation_text=f"PORTFOLIO DV01  ${port_dv01:,.4f}/BP",
        annotation_position="top",
        annotation_font=dict(color=WHITE, size=9),
    )
    apply_bbg(fig)
    fig.update_layout(
        title="DV01 BY BOND POSITION  ($ PER BASIS POINT)",
        xaxis_title="USD / BP",
        yaxis_title="TICKER",
        height=380,
        showlegend=False,
    )
    return fig


def krd_heatmap(krd_outputs: list[RiskOutput], portfolio_df: pd.DataFrame) -> go.Figure:
    """Heatmap: positions × KRD buckets."""
    bond_tickers = portfolio_df[portfolio_df["asset_class"] == "BOND"]["ticker"].tolist()
    position_outputs = [
        o for o in krd_outputs
        if "Portfolio" not in o.metric_name and o.metric_name.startswith("KRD")
    ]

    krd_matrix = {t: {b: 0.0 for b in KRD_BUCKETS} for t in bond_tickers}
    for o in position_outputs:
        ticker = o.metadata.get("ticker", "")
        if ticker in krd_matrix:
            for bucket, val in o.metadata.get("krd_by_bucket", {}).items():
                if bucket in KRD_BUCKETS:
                    krd_matrix[ticker][bucket] = val

    z = [[krd_matrix[t][b] for b in KRD_BUCKETS] for t in bond_tickers]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=KRD_BUCKETS,
        y=bond_tickers,
        colorscale=[
            [0.0, "#000033"],
            [0.5, "#FFA500"],
            [1.0, "#FFFFFF"],
        ],
        text=[[f"{v:.2f}Y" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10, color="#FFFFFF"),
        colorbar=dict(
            title=dict(text="KRD (Y)", font=dict(color=ORANGE, size=10)),
            tickfont=dict(color="#CCCCCC", size=9),
            outlinecolor=DARK_GREY,
            outlinewidth=1,
        ),
        showscale=True,
    ))
    apply_bbg(fig)
    fig.update_layout(
        title="KEY RATE DURATION HEATMAP  (YEARS PER BUCKET)",
        xaxis_title="MATURITY BUCKET",
        yaxis_title="BOND POSITION",
        height=max(280, len(bond_tickers) * 75),
    )
    return fig


def yield_curve_chart(
    yield_curve: pd.DataFrame,
    recession_flags: pd.Series | None = None,
) -> go.Figure:
    """Yield curve: current vs 1M ago vs 1Y ago."""
    maturity_years = [1/12, 3/12, 6/12, 1, 2, 5, 7, 10, 20, 30]
    labels = yield_curve.columns.tolist()

    current = yield_curve.iloc[-1] * 100
    m1_ago  = yield_curve.iloc[-22] * 100 if len(yield_curve) > 22  else None
    y1_ago  = yield_curve.iloc[-252] * 100 if len(yield_curve) > 252 else None

    fig = go.Figure()

    if y1_ago is not None:
        fig.add_trace(go.Scatter(
            x=maturity_years, y=y1_ago.values,
            mode="lines", name="1Y AGO",
            line=dict(color=GREY, width=1, dash="dot"),
        ))
    if m1_ago is not None:
        fig.add_trace(go.Scatter(
            x=maturity_years, y=m1_ago.values,
            mode="lines+markers", name="1M AGO",
            line=dict(color="#4488FF", width=1.5, dash="dash"),
            marker=dict(size=5, color="#4488FF"),
        ))
    fig.add_trace(go.Scatter(
        x=maturity_years, y=current.values,
        mode="lines+markers", name="CURRENT",
        line=dict(color=ORANGE, width=2.5),
        marker=dict(size=7, color=ORANGE, symbol="circle"),
    ))

    apply_bbg(fig)
    fig.update_layout(
        title=f"U.S. TREASURY YIELD CURVE  |  AS OF {yield_curve.index[-1].strftime('%Y-%m-%d')}",
        xaxis=dict(
            title="MATURITY (YEARS)",
            tickvals=maturity_years,
            ticktext=labels,
            tickfont=dict(size=9),
        ),
        yaxis_title="YIELD (%)",
        height=420,
    )
    return fig
