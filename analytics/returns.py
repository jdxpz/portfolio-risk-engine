"""
Return engine — RawPrices + Portfolio → Returns, P&L series.
Log returns only. Arithmetic returns derived on demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.contracts import (
    ContractError,
    hash_dataframe,
    validate_raw_prices,
    validate_returns,
)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from RawPrices.
    Drops first row (always NaN after diff). Validates Returns contract.
    """
    validate_raw_prices(prices)
    log_ret = np.log(prices / prices.shift(1)).iloc[1:]

    # Forward-fill isolated NaNs (e.g. halted tickers for one day), then drop any remainder
    log_ret = log_ret.ffill()
    if log_ret.isnull().any().any():
        bad_cols = log_ret.columns[log_ret.isnull().any()].tolist()
        raise ContractError(
            f"Returns: NaNs remain after forward-fill in columns {bad_cols}. "
            "Check for tickers with insufficient price history."
        )

    return validate_returns(log_ret)


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Weighted portfolio return series from asset returns and weight vector.
    weights: Series indexed by ticker, sums to 1.0.
    Returns a daily portfolio log-return Series.
    """
    validate_returns(returns)
    missing = [t for t in weights.index if t not in returns.columns]
    if missing:
        raise ContractError(f"compute_portfolio_returns: tickers not in returns: {missing}")

    aligned = returns[weights.index]
    port_ret = aligned @ weights
    port_ret.name = "portfolio"
    return port_ret


def compute_pnl(
    prices: pd.DataFrame,
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Daily mark-to-market P&L per position and aggregate.

    Returns DataFrame with columns:
      ticker columns: daily $ P&L per position
      'total': aggregate daily $ P&L
      'cumulative': cumulative $ P&L from start
    """
    validate_raw_prices(prices)

    result_frames = []
    for _, row in portfolio_df.iterrows():
        ticker = row["ticker"]
        if ticker not in prices.columns:
            continue
        px = prices[ticker].dropna()
        daily_pnl = px.diff() * row["quantity"]
        daily_pnl.name = ticker
        result_frames.append(daily_pnl)

    if not result_frames:
        raise ContractError("compute_pnl: no valid tickers found in prices")

    pnl = pd.concat(result_frames, axis=1).fillna(0.0)
    pnl["total"] = pnl.sum(axis=1)
    pnl["cumulative"] = pnl["total"].cumsum()
    return pnl


def arithmetic_from_log(log_returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert log returns to arithmetic returns."""
    return np.expm1(log_returns)


def annualise_return(period_return: float, periods: int = 252) -> float:
    """Annualise a cumulative return over `periods` trading days."""
    return (1 + period_return) ** (252 / periods) - 1


def weights_from_portfolio(portfolio_df: pd.DataFrame) -> pd.Series:
    """Extract weight Series from Portfolio DataFrame, indexed by ticker."""
    return pd.Series(
        portfolio_df["weight"].values,
        index=portfolio_df["ticker"].values,
        name="weight",
    )
