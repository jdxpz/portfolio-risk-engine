# Portfolio Risk Engine (PRE)

A Bloomberg-terminal-styled risk analytics dashboard for multi-asset portfolios,
built with Streamlit. It computes market, rates, and credit risk measures and
rolls them into a Basel III / FRTB-style Comprehensive Risk Measure (CRM).

> **Live demo:** deployed on Render — `https://portfolio-risk-engine.onrender.com`
> _(exact URL confirmed after the first deploy)_

---

## Features

| Tab | What it does |
|-----|--------------|
| **Portfolio Overview** | Positions, weights, cumulative return, P&L attribution, Sharpe / Sortino / Calmar / max drawdown |
| **Market Risk** | VaR (historical, parametric, Monte Carlo), Euler component VaR, incremental VaR, MC loss distribution |
| **Rates Risk** | U.S. Treasury yield curve, modified/Macaulay duration, DV01, key-rate duration, convexity (ETF-proxy method) |
| **Stress Testing** | Historical scenario replays, hypothetical multi-factor shock builder, probabilistic expected loss |
| **CRM Summary** | VaR + stressed VaR + default risk + liquidity add-on, limit utilisation (RAG), full JSON/CSV audit export |

## Architecture

A strictly layered design — each boundary exchanges validated **data contracts**:

```
data/       ingestion + contract validation (prices, yields, portfolio)
analytics/  pure risk computation (VaR, stress, rates, CRM, performance)
charts/     Plotly visualisation
app.py      Streamlit presentation layer (no business logic)
```

## Tech stack

Python 3.12 · Streamlit · pandas / NumPy / SciPy · Plotly · yfinance (prices) · FRED API (yields)

## Data sources & resilience

- **Prices:** Yahoo Finance via `yfinance` (adjusted close, parquet-cached 24h).
- **Yields:** U.S. Treasury par yields + recession flags from FRED (free
  [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) required).
- **Resilience:** live fetches retry with backoff; if a source is unreachable the
  app serves a committed snapshot in `data/fallback/` and shows a clear
  "data as of …" banner, so it always renders.

## Run locally

```bash
git clone https://github.com/jdxpz/portfolio-risk-engine.git
cd portfolio-risk-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FRED_API_KEY="your_fred_key"      # optional; enables the Rates Risk tab
streamlit run app.py
```

Upload your own portfolio CSV (`ticker, asset_class, quantity, cost_basis_usd, currency`;
optional `name`/`description` column for the product label) or use the bundled sample.

## Deployment (Render)

Configured via [`render.yaml`](render.yaml) as a Blueprint:

1. Push this repo to GitHub.
2. On [render.com](https://render.com): **New → Blueprint**, select the repo.
3. Set the `FRED_API_KEY` environment variable (secret) in the dashboard.
4. Deploy. A 1 GB persistent disk keeps the price/yield cache warm across restarts.

## Methodology notes

This is a portfolio / demonstration project. Several measures use transparent
approximations (ETF-proxy duration rather than full cash-flow discounting,
user-supplied PD/LGD assumptions, indicative scenario base rates). These are
labelled in-app and should be read as directional, not as calibrated production
risk numbers.
