"""
Bloomberg Terminal chart theme — shared across all chart modules.
"""

BBG_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(family="IBM Plex Mono, Courier New, monospace", color="#CCCCCC", size=11),
    title=dict(font=dict(color="#FFA500", size=12, family="IBM Plex Mono, monospace"),
               x=0, xanchor="left"),
    xaxis=dict(
        gridcolor="#1A1A1A", linecolor="#333333", tickcolor="#333333",
        tickfont=dict(color="#888888", size=10),
        title_font=dict(color="#FFA500", size=10),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#1A1A1A", linecolor="#333333", tickcolor="#333333",
        tickfont=dict(color="#888888", size=10),
        title_font=dict(color="#FFA500", size=10),
        zeroline=False,
    ),
    legend=dict(
        bgcolor="#0A0A0A", bordercolor="#333333", borderwidth=1,
        font=dict(color="#CCCCCC", size=10),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    margin=dict(l=50, r=30, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#111111", bordercolor="#FFA500",
        font=dict(color="#FFFFFF", size=11, family="IBM Plex Mono, monospace"),
    ),
)

# Colour palette
ORANGE     = "#FFA500"
WHITE      = "#FFFFFF"
RED        = "#FF3333"
GREEN      = "#00CC44"
AMBER      = "#FF9900"
GREY       = "#555555"
DARK_GREY  = "#333333"
PANEL_BG   = "#0A0A0A"

# Asset class colours (muted, terminal-style)
ASSET_COLOURS = {
    "EQUITY":    "#4488FF",
    "BOND":      "#FFA500",
    "FX":        "#00CC88",
    "COMMODITY": "#FFCC00",
    "CASH":      "#888888",
    "UNKNOWN":   "#444444",
}

# Factor colours for stress decomposition
FACTOR_COLOURS = {
    "rates":     "#4488FF",
    "credit":    "#FFA500",
    "equity":    "#FF3333",
    "commodity": "#FFCC00",
}


def apply_bbg(fig):
    """Apply Bloomberg layout defaults to a Plotly figure."""
    fig.update_layout(**BBG_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="#1A1A1A", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1A1A1A", zeroline=False)
    return fig
