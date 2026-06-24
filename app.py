"""
PRE — Portfolio Risk Engine
Streamlit dashboard. Pure presentation layer: no business logic.
Five tabs: Portfolio Overview, Market Risk, Rates Risk, Stress Testing, CRM Summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import os as _os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from data.contracts import ScenarioShock
from data.market_data import fetch_prices
from data.portfolio_loader import load_portfolio
from data.scenario_store import HISTORICAL_SCENARIOS
from data.yield_fetcher import fetch_yield_curve, fetch_recession_flags
from analytics.returns import (
    compute_log_returns,
    compute_pnl,
    compute_portfolio_returns,
    weights_from_portfolio,
)
from analytics.var_engine import compute_all_var
from analytics.component_var import component_var, incremental_var
from analytics.default_risk import simulate_default_loss
from analytics.rates_risk import (
    compute_convexity,
    compute_dv01,
    compute_duration,
    compute_key_rate_duration,
)
from analytics.performance import compute_all_performance
from analytics.stress_engine import (
    apply_scenario,
    monte_carlo_loss_distribution,
    probabilistic_expected_loss,
    run_all_scenarios,
)
from analytics.crm import aggregate_crm
from charts.risk_charts import component_var_waterfall, var_comparison_chart
from charts.rates_charts import dv01_bar_chart, krd_heatmap, yield_curve_chart
from charts.stress_charts import mc_loss_histogram, scenario_pnl_chart, scenario_waterfall
from charts.performance_charts import attribution_bar_chart, performance_summary_figure

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="JDX Capital & Analytics — Portfolio Risk Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bloomberg Terminal CSS
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
    background-color: #000000 !important;
    color: #FFFFFF !important;
  }

  /* App background */
  .stApp { background-color: #000000; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid #FFA500;
  }
  [data-testid="stSidebar"] * { color: #FFFFFF !important; }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFA500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
  }
  [data-testid="stSidebar"] label {
    color: #FFA500 !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  [data-testid="stSidebar"] .stSlider > div > div { background: #FFA500 !important; }
  [data-testid="stSidebar"] [data-baseweb="select"] {
    background: #111111 !important;
    border: 1px solid #333333 !important;
  }
  [data-testid="stSidebar"] input {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    color: #FFFFFF !important;
  }
  [data-testid="stSidebar"] hr { border-color: #333333; }

  /* ── Tabs ── */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #000000;
    border-bottom: 2px solid #FFA500;
    gap: 0;
  }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: #111111;
    color: #888888;
    border: 1px solid #222222;
    border-bottom: none;
    border-radius: 0;
    padding: 0.4rem 1.2rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-right: 2px;
  }
  [data-testid="stTabs"] [aria-selected="true"] {
    background-color: #FFA500 !important;
    color: #000000 !important;
    border-color: #FFA500 !important;
  }
  [data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background-color: #1A1A1A !important;
    color: #FFA500 !important;
  }
  [data-testid="stTabs"] [aria-selected="true"]:hover {
    background-color: #FFA500 !important;
    color: #000000 !important;
  }

  /* ── Metrics ── */
  [data-testid="stMetric"] {
    background-color: #0D0D0D;
    border: 1px solid #222222;
    border-top: 2px solid #FFA500;
    padding: 0.6rem 0.8rem;
    border-radius: 0;
  }
  [data-testid="stMetricLabel"] {
    color: #FFA500 !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.1rem !important;
    font-weight: 600;
  }
  [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }

  /* ── Headers ── */
  h1 {
    color: #FFA500 !important;
    font-size: 1.1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #FFA500;
    padding-bottom: 0.3rem;
  }
  h2, h3 {
    color: #FFA500 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-left: 3px solid #FFA500;
    padding-left: 0.5rem;
    margin-top: 1.2rem;
  }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {
    border: 1px solid #333333;
  }
  [data-testid="stDataFrame"] thead tr th {
    background-color: #111111 !important;
    color: #FFA500 !important;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #FFA500 !important;
  }
  [data-testid="stDataFrame"] tbody tr td {
    background-color: #0A0A0A !important;
    color: #FFFFFF !important;
    font-size: 0.72rem;
    border-bottom: 1px solid #1A1A1A !important;
  }
  [data-testid="stDataFrame"] tbody tr:hover td {
    background-color: #1A1A1A !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background-color: #000000;
    color: #FFA500;
    border: 1px solid #FFA500;
    border-radius: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.35rem 1rem;
  }
  .stButton > button:hover {
    background-color: #FFA500;
    color: #000000;
  }

  /* ── Alerts / banners ── */
  [data-testid="stAlert"] {
    border-radius: 0;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
  }
  .stWarning { border-left: 4px solid #FFA500 !important; background: #1A0F00 !important; }
  .stError   { border-left: 4px solid #FF3333 !important; background: #1A0000 !important; }
  .stSuccess { border-left: 4px solid #00CC44 !important; background: #001A09 !important; }
  .stInfo    { border-left: 4px solid #4488FF !important; background: #00091A !important; }

  /* ── Selectbox / inputs ── */
  [data-baseweb="select"] {
    background-color: #0D0D0D !important;
    border: 1px solid #333333 !important;
    border-radius: 0 !important;
  }
  [data-baseweb="select"] * { color: #FFFFFF !important; font-size: 0.72rem; }
  input[type="number"], input[type="text"], input[type="password"] {
    background-color: #0D0D0D !important;
    border: 1px solid #333333 !important;
    border-radius: 0 !important;
    color: #FFFFFF !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
  }

  /* ── Sliders ── */
  [data-testid="stSlider"] > div > div > div {
    background: linear-gradient(to right, #FFA500, #FFA500) !important;
  }
  [data-testid="stSlider"] label {
    color: #FFA500 !important;
    font-size: 0.65rem;
    text-transform: uppercase;
  }

  /* ── Divider ── */
  hr { border-color: #222222 !important; }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    border: 1px solid #333333;
    border-radius: 0;
    background: #0A0A0A;
  }
  [data-testid="stExpander"] summary {
    color: #FFA500 !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Download buttons ── */
  [data-testid="stDownloadButton"] > button {
    background-color: #000000;
    color: #FFA500;
    border: 1px solid #FFA500;
    border-radius: 0;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  [data-testid="stDownloadButton"] > button:hover {
    background-color: #FFA500;
    color: #000000;
  }

  /* ── Caption / small text ── */
  .stCaption, [data-testid="stCaptionContainer"] {
    color: #666666 !important;
    font-size: 0.65rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }

  /* ── Progress bar ── */
  [data-testid="stProgressBar"] > div > div {
    background-color: #FFA500 !important;
  }

  /* ── Spinner ── */
  [data-testid="stSpinner"] { color: #FFA500 !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0A0A0A; }
  ::-webkit-scrollbar-thumb { background: #333333; }
  ::-webkit-scrollbar-thumb:hover { background: #FFA500; }

  /* ── Top header bar ── */
  .bbg-header {
    background: #000000;
    border-bottom: 2px solid #FFA500;
    padding: 0.4rem 0;
    margin-bottom: 0.8rem;
  }
  .bbg-header-title {
    color: #FFA500;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.2em;
  }
  .bbg-header-sub {
    color: #888888;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
  }

  /* ── KPI label style ── */
  .bbg-label {
    color: #FFA500;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 2px;
  }
  .bbg-value {
    color: #FFFFFF;
    font-size: 1.05rem;
    font-weight: 600;
  }
  .bbg-value-neg { color: #FF3333; }
  .bbg-value-pos { color: #00CC44; }

  /* ── Status badges ── */
  .badge-green  { color: #00CC44; font-weight: 700; }
  .badge-amber  { color: #FFA500; font-weight: 700; }
  .badge-red    { color: #FF3333; font-weight: 700; }
  .badge-grey   { color: #888888; font-weight: 700; }

  /* ── Section separator ── */
  .bbg-section {
    border-top: 1px solid #222222;
    margin: 0.8rem 0;
  }
</style>
""", unsafe_allow_html=True)


def methodology_note(text: str) -> None:
    """Small editable placeholder caption box for documenting methodology."""
    st.markdown(
        f'<div style="border:1px solid #333333; border-left:3px solid #FFA500; '
        f'background:#0A0A0A; padding:0.5rem 0.8rem; margin:0.3rem 0 0.7rem 0; '
        f'color:#999999; font-size:0.7rem; line-height:1.45;">'
        f'<span style="color:#FFA500; font-weight:600; letter-spacing:0.08em;">'
        f'METHODOLOGY NOTE</span><br>{text}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### JDX CAPITAL")
    st.markdown("**PORTFOLIO ANALYTICS RISK ENGINE**")
    st.divider()

    uploaded = st.file_uploader("Portfolio CSV", type="csv")
    use_sample = st.checkbox("Use sample portfolio", value=True)

    st.divider()
    lookback_days = st.slider("VAR LOOKBACK (DAYS)", 126, 756, 252, step=63)
    var_confidence = st.selectbox("VAR CONFIDENCE", [0.95, 0.99], index=0)
    risk_free_rate = st.number_input("RISK-FREE RATE (ANNUAL %)", 0.0, 10.0, 5.0, 0.25) / 100
    risk_limit = st.number_input("VAR RISK LIMIT ($)", 0, 10_000_000, 500_000, 50_000)

    st.divider()
    # FRED key is provided server-side (env var / Render secret) and is NEVER
    # rendered into a widget, so it is never exposed to the browser on a public
    # deployment. Rates-risk features enable automatically when it is configured.
    fred_api_key = _os.environ.get("FRED_API_KEY", "")
    if fred_api_key:
        st.caption("RATES DATA — FRED: CONNECTED")
    else:
        st.caption("RATES DATA — FRED: NOT CONFIGURED (set FRED_API_KEY)")

    st.divider()
    st.markdown("**DEFAULT RISK ASSUMPTIONS**")
    st.caption("USER-SUPPLIED — NOT MODEL OUTPUTS")
    hyg_pd  = st.slider("HYG PD (%)",  0.0, 15.0,  3.0, 0.5) / 100
    hyg_lgd = st.slider("HYG LGD (%)", 0.0, 100.0, 40.0, 5.0) / 100
    lqd_pd  = st.slider("LQD PD (%)",  0.0, 5.0,   0.5, 0.1) / 100
    lqd_lgd = st.slider("LQD LGD (%)", 0.0, 60.0,  30.0, 5.0) / 100

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def load_data(csv_path: str | None, start: str, fred_key: str | None):
    if csv_path:
        portfolio = load_portfolio(csv_path)
    else:
        portfolio = load_portfolio(Path(__file__).parent / "data" / "sample_portfolio.csv")

    tickers = portfolio["ticker"].tolist()
    prices  = fetch_prices(tickers, start=start)
    returns = compute_log_returns(prices)

    # Provenance is read here (inside the cached fn) into plain values, since a
    # DataFrame's .attrs do not reliably survive st.cache_data serialisation.
    data_meta = {
        "price_source": prices.attrs.get("source", "live"),
        "price_date":   prices.attrs.get("data_date"),
        "yield_source": None,
        "yield_date":   None,
    }

    yield_curve = None
    recession   = None
    if fred_key:
        try:
            yield_curve = fetch_yield_curve(start=start, fred_api_key=fred_key)
            recession   = fetch_recession_flags(start=start, fred_api_key=fred_key)
            data_meta["yield_source"] = yield_curve.attrs.get("source", "live")
            data_meta["yield_date"]   = yield_curve.attrs.get("data_date")
        except Exception as e:
            st.warning(f"YIELD CURVE UNAVAILABLE: {e}")

    return portfolio, prices, returns, yield_curve, recession, data_meta


with st.spinner("LOADING MARKET DATA..."):
    try:
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            portfolio, prices, returns, yield_curve, recession, data_meta = load_data(
                tmp_path, "2020-01-01", fred_api_key or None
            )
            _os.unlink(tmp_path)
        else:
            portfolio, prices, returns, yield_curve, recession, data_meta = load_data(
                None, "2020-01-01", fred_api_key or None
            )
    except Exception as e:
        st.error(f"DATA LOADING FAILED: {e}")
        st.stop()

# Surface a clear banner if any data came from the committed fallback snapshot
# (i.e. live yfinance/FRED was unreachable — e.g. Yahoo throttling Render's IP).
if data_meta.get("price_source") == "fallback" or data_meta.get("yield_source") == "fallback":
    _snap_date = data_meta.get("price_date") or data_meta.get("yield_date") or "N/A"
    st.warning(
        f"LIVE MARKET DATA UNAVAILABLE — DISPLAYING COMMITTED SNAPSHOT "
        f"(DATA AS OF {_snap_date}). FIGURES REFLECT THE LAST CACHED DATASET, "
        f"NOT REAL-TIME PRICES."
    )

weights         = weights_from_portfolio(portfolio)
portfolio_value = float(portfolio["market_value_usd"].sum())
port_returns    = compute_portfolio_returns(returns, weights)
pnl_df          = compute_pnl(prices, portfolio)

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def run_analytics(
    _returns, _weights, _portfolio, _port_returns,
    pv, lookback, conf, rf, pd_assumptions, lgd_assumptions, _yield_curve,
):
    var_results = compute_all_var(_returns, _weights, pv, [0.95, 0.99], lookback)
    comp_outputs, comp_check = component_var(_returns, _weights, pv, conf, lookback)

    default_el = default_var99 = default_losses = None
    if (_portfolio["asset_class"] == "BOND").any() and pd_assumptions:
        default_el, default_var99, default_losses = simulate_default_loss(
            _portfolio, pd_assumptions, lgd_assumptions
        )

    perf = compute_all_performance(
        _port_returns, _returns, _weights,
        pd.Series(dict(zip(_portfolio["ticker"], _portfolio["asset_class"]))),
        rf,
    )

    mc_outs, mc_sim = monte_carlo_loss_distribution(
        _returns, _weights, pv, confidence_levels=[0.95, 0.99],
    )

    rates_outputs = {}
    if _yield_curve is not None:
        rates_outputs["duration"]  = compute_duration(_portfolio, _yield_curve)
        rates_outputs["dv01"]      = compute_dv01(_portfolio, _yield_curve)
        rates_outputs["krd"]       = compute_key_rate_duration(_portfolio, _yield_curve)
        rates_outputs["convexity"] = compute_convexity(_portfolio, _yield_curve)

    return (var_results, comp_outputs, comp_check,
            default_el, default_var99, default_losses,
            perf, mc_outs, mc_sim, rates_outputs)


with st.spinner("RUNNING RISK ANALYTICS..."):
    pd_assumptions  = {"HYG": hyg_pd,  "LQD": lqd_pd}
    lgd_assumptions = {"HYG": hyg_lgd, "LQD": lqd_lgd}
    (
        var_results, comp_outputs, comp_check,
        default_el, default_var99, default_losses,
        perf, mc_outs, mc_sim, rates_outputs,
    ) = run_analytics(
        returns, weights, portfolio, port_returns,
        portfolio_value, lookback_days, var_confidence,
        risk_free_rate, pd_assumptions, lgd_assumptions, yield_curve,
    )

_yc = yield_curve if yield_curve is not None else pd.DataFrame()
stress_results  = run_all_scenarios(portfolio, _yc, HISTORICAL_SCENARIOS)
worst_stress_pnl = min(r.total_pnl_usd for r in stress_results) if stress_results else 0.0

crm = aggregate_crm(
    var_results, worst_stress_pnl, default_var99, portfolio,
    risk_limit_usd=float(risk_limit), var_confidence=var_confidence,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

conf_key = f"monte_carlo_{int(var_confidence*100)}"
var_usd  = var_results[conf_key][0].value if conf_key in var_results else 0.0
cvar_usd = var_results[conf_key][1].value if conf_key in var_results else 0.0
rag      = crm.utilisation_rag()
util_pct = f"{crm.limit_utilisation:.1%}" if crm.limit_utilisation else "N/A"

rag_class = {"RED": "badge-red", "AMBER": "badge-amber", "GREEN": "badge-green"}.get(rag, "badge-grey")

st.markdown(f"""
<div class="bbg-header">
  <span class="bbg-header-title">PRE &nbsp;|&nbsp; PORTFOLIO RISK ENGINE</span>
  &nbsp;&nbsp;
  <span class="bbg-header-sub">
    {len(portfolio)} POSITIONS &nbsp;|&nbsp;
    DATA THROUGH {returns.index[-1].strftime('%Y-%m-%d')} &nbsp;|&nbsp;
    AS OF {pd.Timestamp.utcnow().strftime('%H:%M UTC')}
  </span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("PORTFOLIO VALUE",          f"${portfolio_value:,.0f}")
col2.metric(f"MC VAR {int(var_confidence*100)}%", f"${var_usd:,.0f} ")
col3.metric(f"MC CVAR {int(var_confidence*100)}%",f"${cvar_usd:,.0f}")
col4.metric("CRM CAPITAL",              f"${crm.total_crm:,.0f}")
col5.metric(
    "RISK LIMIT UTILIZATION",
    util_pct,
    delta="BREACH" if rag == "RED" else ("WARNING" if rag == "AMBER" else None),
    delta_color="inverse",
)

if crm.breach_flags:
    for flag in crm.breach_flags:
        st.warning(flag)

st.markdown('<div class="bbg-section"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "PORTFOLIO OVERVIEW",
    "MARKET RISK",
    "RATES RISK",
    "STRESS TESTING",
    "CRM SUMMARY",
])

# ── Tab 1: Portfolio Overview ───────────────────────────────────────────────

with tab1:
    st.subheader("POSITIONS AND WEIGHTS")
    display_cols = ["ticker", "name", "asset_class", "quantity",
                    "cost_basis_usd", "market_value_usd", "weight", "currency"]
    disp = portfolio[display_cols].copy()
    disp["weight"]          = disp["weight"].map("{:.2%}".format)
    disp["market_value_usd"]= disp["market_value_usd"].map("${:,.0f}".format)
    disp["cost_basis_usd"]  = disp["cost_basis_usd"].map("${:.2f}".format)
    disp.columns            = [c.upper().replace("_", " ") for c in disp.columns]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.subheader("CUMULATIVE RETURN")
    st.plotly_chart(
        performance_summary_figure(port_returns, perf["drawdown_series"], perf["rolling_sharpe"]),
        use_container_width=True, key="tab1_perf_summary",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("P&L ATTRIBUTION BY ASSET CLASS")
        st.plotly_chart(
            attribution_bar_chart(perf["attribution"]),
            use_container_width=True, key="tab1_attribution",
        )
    with col_b:
        st.subheader("KEY PERFORMANCE METRICS")
        pm1, pm2 = st.columns(2)
        pm1.metric("SHARPE RATIO",  f"{perf['sharpe'].value:.3f}")
        pm2.metric("SORTINO RATIO", f"{perf['sortino'].value:.3f}")
        pm3, pm4 = st.columns(2)
        pm3.metric("CALMAR RATIO",  f"{perf['calmar'].value:.3f}")
        pm4.metric("MAX DRAWDOWN",  f"{perf['max_drawdown'].value:.2%}")

        mdd_meta = perf["max_drawdown"].metadata
        st.markdown('<div class="bbg-section"></div>', unsafe_allow_html=True)
        st.caption(
            f"PEAK: {mdd_meta.get('peak_date', 'N/A')}  |  "
            f"TROUGH: {mdd_meta.get('trough_date', 'N/A')}  |  "
            f"RECOVERY: {mdd_meta.get('recovery_date', 'N/A')}  |  "
            f"DURATION: {mdd_meta.get('recovery_days', 'N/A')} DAYS"
        )

# ── Tab 2: Market Risk ──────────────────────────────────────────────────────

with tab2:
    st.subheader("VAR COMPARISON — HISTORICAL / PARAMETRIC / MONTE CARLO")
    st.plotly_chart(
        var_comparison_chart(var_results),
        use_container_width=True, key="tab2_var_comparison",
    )

    st.subheader("COMPONENT VAR — EULER DECOMPOSITION")
    st.plotly_chart(
        component_var_waterfall(comp_outputs, portfolio),
        use_container_width=True, key="tab2_component_var",
    )
    if comp_check:
        err = comp_check.metadata.get("relative_error", 0)
        st.caption(
            f"SUM CHECK: COMPONENT VAR = ${comp_check.value:,.0f}  |  "
            f"RELATIVE ERROR = {err:.6%}  |  STATUS: PASS"
        )

    col_mc, col_inc = st.columns([2, 1])
    with col_mc:
        st.subheader("MONTE CARLO LOSS DISTRIBUTION")
        st.plotly_chart(
            mc_loss_histogram(mc_sim, portfolio_value),
            use_container_width=True, key="tab2_mc_hist",
        )
    with col_inc:
        st.subheader("INCREMENTAL VAR")
        st.caption("ESTIMATE VAR DELTA FROM ADDING A NEW POSITION")
        inc_ticker = st.selectbox(
            "TICKER",
            options=[t for t in returns.columns if t not in portfolio["ticker"].values],
        )
        inc_value = st.number_input("POSITION SIZE ($)", 10_000, 5_000_000, 100_000, 10_000)
        if st.button("COMPUTE INCREMENTAL VAR"):
            try:
                inc_out = incremental_var(
                    returns, weights, portfolio_value,
                    inc_ticker, float(inc_value), var_confidence, lookback_days,
                )
                direction = "INCREASES" if inc_out.value > 0 else "REDUCES"
                st.metric(
                    f"DELTA VAR FROM {inc_ticker}",
                    f"${inc_out.value:+,.0f}",
                )
                st.caption(f"ADDING {inc_ticker} {direction} PORTFOLIO VAR")
            except Exception as e:
                st.error(str(e))

    if default_el is not None:
        st.markdown('<div class="bbg-section"></div>', unsafe_allow_html=True)
        st.subheader("INCREMENTAL DEFAULT LOSS — BOND BOOK")
        st.caption("WARNING: PD AND LGD ARE USER-SUPPLIED ASSUMPTIONS — NOT CALIBRATED MODEL ESTIMATES. CREDIT RISK ANALYSIS FRAMEWORK FROM CFA L1 CURRICULUM")
        dc1, dc2 = st.columns(2)
        dc1.metric("EXPECTED DEFAULT LOSS",    f"${default_el.value:,.0f}")
        dc2.metric("DEFAULT LOSS VAR (99%)",   f"${default_var99.value:,.0f}")

# ── Tab 3: Rates Risk ───────────────────────────────────────────────────────

with tab3:
    if yield_curve is None:
        st.info("FRED API KEY REQUIRED — ENTER KEY IN SIDEBAR TO ENABLE RATES RISK FEATURES")
    else:
        st.subheader("U.S. TREASURY YIELD CURVE")
        st.plotly_chart(
            yield_curve_chart(yield_curve, recession),
            use_container_width=True, key="tab3_yield_curve",
        )
        methodology_note(
            "Current data obtained from FRED (Federal Reserve Economic Data) API to construct a par yield curve based on yields of different maturities, contrasting periodic movements as well to monitor the yield curve movements and economic narrative."
            "Useful to monitor fixed income allocation and positioning, as of 06/22, longer term yields reflecting the expectations of the market for slowing economic growth as policy divergence between the FED and other central banks is raising expectations for rate hikes."
            "Duration measures help provide estimates for portfolio impact, identifying the most exposed constituents across your portfolio with KRD metrics for each maturity."
            "DV01 will serve as an instrumental measure to help calculate optimal hedges. Ultimately a list of desk products would be compiled, referencing the DV01 and other sensitivity measures to perform an optimization model that would calculate the hedges that reduce the highest level of exposure at the lowest marginal cost. As an Call To Action to Sales to offer hedges and products based on the portfolio analytics or for trade desks to hedge open positions and warehoused risk."
        )

        if not (portfolio["asset_class"] == "BOND").any():
            st.info("NO BOND POSITIONS IN PORTFOLIO — RATES RISK NOT APPLICABLE")
        else:
            st.subheader("RATES SENSITIVITIES")
            dur_cols = st.columns(4)
            port_dur  = next((o for o in rates_outputs["duration"]  if "Portfolio" in o.metric_name), None)
            port_conv = next((o for o in rates_outputs["convexity"] if "Portfolio" in o.metric_name), None)
            port_dv01 = next((o for o in rates_outputs["dv01"]      if "Portfolio" in o.metric_name), None)
            if port_dur:
                dur_cols[0].metric("MODIFIED DURATION", f"{port_dur.value:.2f}%")
                dur_cols[1].metric("MACAULAY DURATION",
                                   f"{port_dur.metadata.get('macaulay_duration', 0):.2f}Y")
            if port_dv01:
                dur_cols[2].metric("PORTFOLIO DV01", f"${port_dv01.value:,.2f}/BP")
            if port_conv:
                dur_cols[3].metric("CONVEXITY", f"{port_conv.value:.2f}Y²")

            st.caption(
                "APPROXIMATION: RATES METRICS COMPUTED VIA ETF PROXY DURATION. "
                "FULL CASH FLOW DISCOUNTING NOT IMPLEMENTED."
            )

            col_dv, col_krd = st.columns(2)
            with col_dv:
                st.subheader("DV01 BY POSITION")
                st.plotly_chart(
                    dv01_bar_chart(rates_outputs["dv01"]),
                    use_container_width=True, key="tab3_dv01",
                )
            with col_krd:
                st.subheader("KEY RATE DURATION HEATMAP")
                st.plotly_chart(
                    krd_heatmap(rates_outputs["krd"], portfolio),
                    use_container_width=True, key="tab3_krd",
                )

# ── Tab 4: Stress Testing ───────────────────────────────────────────────────

with tab4:
    st.subheader("HISTORICAL SCENARIO P&L — FACTOR DECOMPOSITION")
    st.plotly_chart(
        scenario_pnl_chart(stress_results),
        use_container_width=True, key="tab4_scenario_bar",
    )
    methodology_note(
        "Historical scenario analysis obtains historical market data for major stress scenarios and models the P&L effects to the current portfolio. The methodology assesses how current exposures would perform under market conditions analogous to historical and severe market stress scenarios. Scenario P&L is also decomposed by asset class (Rates, Credit, Equity, and Commodities) to identify the primary exposures and drivers of portfolio performance under each stress event. The below feature allow to model portfolio positioning and P&L for customized scenarios, which may align to external or in house research views."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("SCENARIO DRILL-DOWN")
        selected_scenario = st.selectbox(
            "SELECT SCENARIO",
            options=[r.scenario_name for r in stress_results],
        )
        selected_result = next(r for r in stress_results if r.scenario_name == selected_scenario)
        st.plotly_chart(
            scenario_waterfall(selected_result),
            use_container_width=True, key="tab4_waterfall",
        )
    with col_s2:
        st.subheader("MONTE CARLO LOSS DISTRIBUTION")
        st.plotly_chart(
            mc_loss_histogram(mc_sim, portfolio_value, [0.95, 0.99]),
            use_container_width=True, key="tab4_mc_hist",
        )

    st.markdown('<div class="bbg-section"></div>', unsafe_allow_html=True)
    st.subheader("HYPOTHETICAL SCENARIO BUILDER")
    st.caption("DEFINE A CUSTOM SHOCK ACROSS ALL FOUR RISK DIMENSIONS")
    hyp_cols = st.columns(4)
    rate_shock      = hyp_cols[0].slider("RATE SHIFT (BPS, ALL MATURITIES)", -300, 400, 0, 10)
    credit_shock    = hyp_cols[1].slider("CREDIT SPREAD SHIFT (BPS)",        -200, 800, 0, 10)
    equity_shock    = hyp_cols[2].slider("EQUITY SHOCK (%)",                  -60,  30, 0,  1) / 100
    commodity_shock = hyp_cols[3].slider("COMMODITY SHOCK (%)",               -60,  50, 0,  1) / 100

    if st.button("RUN HYPOTHETICAL SCENARIO"):
        hyp_shock = ScenarioShock(
            name="USER-DEFINED HYPOTHETICAL",
            rate_shift_bps={m: rate_shock
                            for m in ["1M","3M","6M","1Y","2Y","5Y","7Y","10Y","20Y","30Y"]},
            credit_spread_shift_bps=float(credit_shock),
            equity_shock_pct=float(equity_shock),
            commodity_shock_pct=float(commodity_shock),
        )
        hyp_result = apply_scenario(portfolio, _yc, hyp_shock)
        st.metric("ESTIMATED P&L IMPACT", f"${hyp_result.total_pnl_usd:+,.0f}")
        st.plotly_chart(
            scenario_waterfall(hyp_result),
            use_container_width=True, key="tab4_hyp_waterfall",
        )

    st.markdown('<div class="bbg-section"></div>', unsafe_allow_html=True)
    st.subheader("PROBABILISTIC EXPECTED LOSS")
    methodology_note(
        "Probability-weighted average P&L across the five historical stress scenarios, interpreted as the conditional expected hit to the portfolio if one of these black swan or tail risk event scenario were to happen. Probabilities for each event are estimated and weights are normalized as: Normalized Weight=P(Scenario X∣One of these 5 scenarios occurs). The total Conditional Expected Loss can be expressed as:  Conditional_Expected_Loss = Σ_5Scenarios(Scenario_i_PnL * (Scenario_i_Probability / Total_Stress_Probability)"
    )
    try:
        pel = probabilistic_expected_loss(stress_results)
        st.metric("PROBABILITY-WEIGHTED EXPECTED LOSS", f"${pel.value:,.0f}")
        st.caption(
            "WARNING: SCENARIO PROBABILITIES ARE INDICATIVE BASE RATES — "
            "NOT CALIBRATED MODEL OUTPUTS. TREAT AS ESTIMATES AND DIRECTIONAL."
        )
    except Exception as e:
        st.warning(str(e))

# ── Tab 5: CRM Summary ──────────────────────────────────────────────────────

with tab5:
    rag_label = {"RED": "LIMIT BREACHED", "AMBER": "WARNING", "GREEN": "WITHIN LIMITS", "GREY": "NO LIMIT SET"}.get(rag, "N/A")
    st.markdown(f"""
    <div style="border: 1px solid #333; border-top: 3px solid
      {'#FF3333' if rag=='RED' else '#FFA500' if rag=='AMBER' else '#00CC44'};
      padding: 0.6rem 1rem; margin-bottom: 1rem; background: #0A0A0A;">
      <span class="bbg-label">LIMIT STATUS</span><br>
      <span class="{'badge-red' if rag=='RED' else 'badge-amber' if rag=='AMBER' else 'badge-green'}"
            style="font-size: 1rem;">{rag_label}</span>
      &nbsp;&nbsp;
      <span style="color: #888; font-size: 0.72rem;">{util_pct} OF ${risk_limit:,.0f} LIMIT</span>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("COMPREHENSIVE RISK MEASURE BREAKDOWN")
    st.caption(
        "CRM = VAR + STRESSED VAR + DEFAULT RISK + LIQUIDITY ADD-ON  |  "
        "BUILT BASED ON BASEL III / FRTB. DEFINED AS AN INTERNAL CALCULATED MODEL APPROACH, AND THE COMPONENTS WERE CHOSEN BASED ON INSITUTIONAL PRACTICES AND USED FACTORS. CRM ESTABLISHES THE INTERNAL RISK LIMIT"
    )

    crm_cols = st.columns(5)
    crm_cols[0].metric("VAR COMPONENT",    f"${crm.var_component:,.0f}")
    crm_cols[1].metric("STRESSED VAR",     f"${crm.stressed_var_component:,.0f}")
    crm_cols[2].metric("DEFAULT RISK",     f"${crm.default_risk_component:,.0f}")
    crm_cols[3].metric("LIQUIDITY ADD-ON", f"${crm.liquidity_addon:,.0f}")
    crm_cols[4].metric("TOTAL CRM",        f"${crm.total_crm:,.0f}")

    if crm.limit_utilisation is not None:
        st.progress(
            min(crm.limit_utilisation, 1.0),
            text=f"{crm.limit_utilisation:.1%} UTILISATION",
        )

    if crm.breach_flags:
        for flag in crm.breach_flags:
            st.error(flag)
    else:
        st.success("NO LIMIT BREACHES DETECTED")
'''
    st.markdown('<div class="bbg-section"></div>', unsafe_allow_html=True)
    st.subheader("AUDIT TRAIL — COMPONENT SOURCES")
    for component, source in crm.component_sources.items():
        st.markdown(
            f"<span style='color:#FFA500;font-size:0.65rem;text-transform:uppercase;"
            f"letter-spacing:0.08em;'>{component.replace('_', ' ')}</span>"
            f"<span style='color:#CCCCCC;font-size:0.72rem;'>&nbsp;&nbsp;{source}</span>",
            unsafe_allow_html=True,
        )

    #st.markdown('<div class="bbg-section"></div>', unsafe_allow_html=True)
    #st.subheader("FULL CRM OUTPUT — JSON AUDIT RECORD")
    #with st.expander("EXPAND AUDIT RECORD"):
    #    st.json(json.dumps(crm.to_risk_output().to_dict(), indent=2))

    st.markdown('<div class="bbg-section"></div>', unsafe_allow_html=True)
    st.subheader("EXPORT")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        all_outputs = []
        for key, outputs in var_results.items():
            if key.startswith("_"):
                continue
            for o in outputs:
                all_outputs.append(o.to_dict())
        for o in comp_outputs:
            all_outputs.append(o.to_dict())
        all_outputs.append(crm.to_risk_output().to_dict())

        st.download_button(
            "DOWNLOAD RISK OUTPUTS (JSON)",
            data=json.dumps(all_outputs, indent=2, default=str),
            file_name="pre_risk_outputs.json",
            mime="application/json",
        )
    with col_exp2:
        csv_rows = [
            {"METRIC": o["metric_name"], "VALUE": o["value"],
             "UNIT": o["unit"], "METHODOLOGY": o["methodology"]}
            for o in all_outputs
        ]
        st.download_button(
            "DOWNLOAD RISK OUTPUTS (CSV)",
            data=pd.DataFrame(csv_rows).to_csv(index=False),
            file_name="pre_risk_outputs.csv",
            mime="text/csv",
        )
'''
