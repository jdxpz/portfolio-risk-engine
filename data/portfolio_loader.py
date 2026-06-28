"""
Portfolio ingestion — CSV → Portfolio contract.
Validates schema, classifies assets, normalises weights.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.contracts import (
    PORTFOLIO_COLUMNS,
    AssetClass,
    ContractError,
    validate_portfolio,
)

logger = logging.getLogger(__name__)

# Known ETF proxies and their forced asset classification
ETF_OVERRIDES: dict[str, AssetClass] = {
    "TLT":  AssetClass.BOND,       # iShares 20+ Year Treasury
    "IEF":  AssetClass.BOND,       # iShares 7-10 Year Treasury
    "LQD":  AssetClass.BOND,       # iShares IG Corporate
    "HYG":  AssetClass.BOND,       # iShares HY Corporate
    "UUP":  AssetClass.FX,         # Invesco USD Bull
    "FXE":  AssetClass.FX,         # Invesco Euro
    "GLD":  AssetClass.COMMODITY,  # SPDR Gold
    "SLV":  AssetClass.COMMODITY,  # iShares Silver Trust
    "USO":  AssetClass.COMMODITY,  # United States Oil
    "SHV":  AssetClass.CASH,       # iShares Short Treasury (cash proxy)
}

# ETF proxy → dominant maturity bucket for KRD mapping (years)
ETF_DURATION: dict[str, float] = {
    "TLT": 17.0,
    "IEF": 7.5,
    "LQD": 8.5,
    "HYG": 4.0,
}


def load_portfolio(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and validate a portfolio CSV.

    Required columns: ticker, asset_class, quantity, cost_basis_usd, currency
    Optional column:  weight (recomputed from cost_basis_usd if absent or zero)

    Returns a validated Portfolio DataFrame.
    """
    path = Path(csv_path)
    if not path.exists():
        raise ContractError(f"Portfolio CSV not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ContractError(f"Failed to read portfolio CSV: {exc}") from exc

    df.columns = df.columns.str.strip().str.lower()

    # Rename common variants
    rename_map = {
        "cost_basis": "cost_basis_usd",
        "asset class": "asset_class",
        "assetclass": "asset_class",
        "description": "name",
        "product": "name",
        "product_name": "name",
    }
    df = df.rename(columns=rename_map)

    required_input = ["ticker", "asset_class", "quantity", "cost_basis_usd", "currency"]
    missing = [c for c in required_input if c not in df.columns]
    if missing:
        raise ContractError(f"Portfolio CSV missing columns: {missing}")

    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["currency"] = df["currency"].str.upper().str.strip()
    df["asset_class"] = df["asset_class"].str.upper().str.strip()

    # Optional human-readable product name; default to the ticker if absent.
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    df["name"] = df["name"].astype(str).str.strip()

    # Apply ETF overrides
    for ticker, cls in ETF_OVERRIDES.items():
        mask = df["ticker"] == ticker
        if mask.any():
            original = df.loc[mask, "asset_class"].iloc[0]
            if original != cls.value:
                logger.info("Overriding %s asset_class: %s → %s", ticker, original, cls.value)
            df.loc[mask, "asset_class"] = cls.value

    # Validate asset classes
    valid = {a.value for a in AssetClass}
    invalid = df[~df["asset_class"].isin(valid)]
    if not invalid.empty:
        raise ContractError(
            f"Unknown asset_class values: {invalid['asset_class'].unique().tolist()}. "
            f"Valid values: {sorted(valid)}"
        )

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["cost_basis_usd"] = pd.to_numeric(df["cost_basis_usd"], errors="coerce")

    if df[["quantity", "cost_basis_usd"]].isnull().any().any():
        raise ContractError("Portfolio CSV: non-numeric values in quantity or cost_basis_usd")

    # Compute market value proxy = quantity × cost_basis_usd (cost basis per share)
    df["market_value_usd"] = df["quantity"] * df["cost_basis_usd"]

    total_value = df["market_value_usd"].sum()
    if total_value <= 0:
        raise ContractError("Portfolio: total market value is zero or negative")

    df["weight"] = df["market_value_usd"] / total_value

    # Attach duration proxy for bond ETFs
    df["duration_years"] = df["ticker"].map(ETF_DURATION).fillna(0.0)

    # Reorder to contract columns + extras (name carried through for display)
    output_cols = PORTFOLIO_COLUMNS + ["name", "market_value_usd", "duration_years"]
    for col in output_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[output_cols].reset_index(drop=True)

    return validate_portfolio(df)


def mark_to_market(portfolio_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Revalue a loaded portfolio to market using the most recent price per ticker.

    Overwrites the cost-based ``market_value_usd``/``weight`` with true
    mark-to-market (latest price x quantity) and adds:
      mtm_price           most recent available close per ticker
      cost_value_usd      quantity x cost_basis_usd (retained for P&L)
      unrealized_pnl_usd  market_value_usd - cost_value_usd

    Falls back to cost basis for any ticker with no price so valuation never
    crashes; weights are renormalised on the marked values.
    """
    df = portfolio_df.copy()

    def _latest_price(ticker: str, cost_basis: float) -> float:
        if ticker in prices.columns:
            series = prices[ticker].dropna()
            if not series.empty:
                return float(series.iloc[-1])
        logger.warning("mark_to_market: no price for %s; using cost basis", ticker)
        return float(cost_basis)

    df["mtm_price"] = [
        _latest_price(t, cb) for t, cb in zip(df["ticker"], df["cost_basis_usd"])
    ]
    df["cost_value_usd"] = df["quantity"] * df["cost_basis_usd"]
    df["market_value_usd"] = df["quantity"] * df["mtm_price"]

    total_mv = df["market_value_usd"].sum()
    if total_mv <= 0:
        raise ContractError("mark_to_market: total market value is zero or negative")
    df["weight"] = df["market_value_usd"] / total_mv
    df["unrealized_pnl_usd"] = df["market_value_usd"] - df["cost_value_usd"]

    return validate_portfolio(df)
