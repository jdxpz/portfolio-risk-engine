"""
Fixed income and rates risk — Duration, DV01, KRD, Convexity.
Bonds modelled via ETF proxies + FRED yield curve.
LIMITATION: Duration/DV01 approximation only — not full cash flow discounting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.contracts import (
    YIELD_CURVE_MATURITIES,
    ContractError,
    RiskOutput,
    hash_dataframe,
    validate_yield_curve,
)

# KRD maturity buckets (subset of full curve — the four standard desk buckets)
KRD_BUCKETS = ["2Y", "5Y", "10Y", "30Y"]

# Approximate bucket boundaries for yield curve interpolation (years)
MATURITY_YEARS = {
    "1M": 1/12, "3M": 3/12, "6M": 6/12, "1Y": 1.0,
    "2Y": 2.0,  "5Y": 5.0,  "7Y": 7.0,  "10Y": 10.0,
    "20Y": 20.0, "30Y": 30.0,
}

LIMITATION_NOTE = (
    "APPROXIMATION: Bond risk computed via duration/DV01 using ETF proxy duration. "
    "Full cash flow discounting not implemented (no QuantLib). "
    "Results are estimates suitable for risk monitoring, not exact pricing."
)


def _get_bond_positions(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    bonds = portfolio_df[portfolio_df["asset_class"] == "BOND"].copy()
    if bonds.empty:
        raise ContractError("rates_risk: no BOND positions in portfolio")
    return bonds


def _current_yield(yield_curve: pd.DataFrame, maturity_label: str) -> float:
    """Most recent yield for a given maturity."""
    validate_yield_curve(yield_curve)
    if maturity_label not in yield_curve.columns:
        raise ContractError(f"rates_risk: maturity '{maturity_label}' not in YieldCurve")
    return float(yield_curve[maturity_label].iloc[-1])


def compute_duration(
    portfolio_df: pd.DataFrame,
    yield_curve: pd.DataFrame,
) -> list[RiskOutput]:
    """
    Macaulay and Modified Duration per bond position and portfolio aggregate.
    Uses ETF proxy duration from portfolio_loader (duration_years column).
    """
    h = hash_dataframe(yield_curve)
    bonds = _get_bond_positions(portfolio_df)

    outputs = []
    total_mv = bonds["market_value_usd"].sum()
    weighted_mac_dur = 0.0
    weighted_mod_dur = 0.0

    for _, row in bonds.iterrows():
        ticker = row["ticker"]
        mac_dur = row["duration_years"]          # from ETF_DURATION in portfolio_loader

        # Approximate yield for this position using dominant maturity bucket
        # Map ETF duration to closest KRD bucket
        dur = mac_dur
        if dur <= 2.5:
            ytm = _current_yield(yield_curve, "2Y")
        elif dur <= 6.0:
            ytm = _current_yield(yield_curve, "5Y")
        elif dur <= 12.0:
            ytm = _current_yield(yield_curve, "10Y")
        else:
            ytm = _current_yield(yield_curve, "30Y")

        # Semiannual compounding convention for bond ETFs
        mod_dur = mac_dur / (1 + ytm / 2)
        
        mv = row["market_value_usd"]
        weight = mv / total_mv

        weighted_mac_dur += weight * mac_dur
        weighted_mod_dur += weight * mod_dur

        outputs.append(RiskOutput(
            metric_name=f"Duration — {ticker}",
            value=mac_dur,
            unit="years",
            methodology="ETF proxy duration (Macaulay)",
            inputs_hash=h,
            metadata={
                "ticker": ticker,
                "macaulay_duration": mac_dur,
                "modified_duration": mod_dur,
                "ytm_used": ytm,
                "market_value_usd": mv,
                "LIMITATION": LIMITATION_NOTE,
            },
        ))

    outputs.append(RiskOutput(
        metric_name="Portfolio Modified Duration",
        value=weighted_mod_dur,
        unit="years",
        methodology="Value-weighted modified duration (ETF proxies)",
        inputs_hash=h,
        metadata={
            "macaulay_duration": weighted_mac_dur,
            "modified_duration": weighted_mod_dur,
            "bond_positions": bonds["ticker"].tolist(),
            "LIMITATION": LIMITATION_NOTE,
        },
    ))
    return outputs


def compute_dv01(
    portfolio_df: pd.DataFrame,
    yield_curve: pd.DataFrame,
) -> list[RiskOutput]:
    """
    DV01 (Dollar Value of 1bp) per bond position and aggregate.
    DV01 = Modified Duration × Market Value × 0.0001
    """
    h = hash_dataframe(yield_curve)
    bonds = _get_bond_positions(portfolio_df)
    outputs = []
    total_dv01 = 0.0

    for _, row in bonds.iterrows():
        ticker = row["ticker"]
        mac_dur = row["duration_years"]
        mv = row["market_value_usd"]

        dur = mac_dur
        if dur <= 2.5:
            ytm = _current_yield(yield_curve, "2Y")
        elif dur <= 6.0:
            ytm = _current_yield(yield_curve, "5Y")
        elif dur <= 12.0:
            ytm = _current_yield(yield_curve, "10Y")
        else:
            ytm = _current_yield(yield_curve, "30Y")

        mod_dur = mac_dur / (1 + ytm / 2)
        dv01 = mod_dur * mv * 0.0001

        total_dv01 += dv01
        outputs.append(RiskOutput(
            metric_name=f"DV01 — {ticker}",
            value=dv01,
            unit="USD/bp",
            methodology="Modified duration × Market value × 0.0001",
            inputs_hash=h,
            metadata={
                "ticker": ticker,
                "modified_duration": mod_dur,
                "market_value_usd": mv,
                "ytm_used": ytm,
                "LIMITATION": LIMITATION_NOTE,
            },
        ))

    outputs.append(RiskOutput(
        metric_name="Portfolio DV01",
        value=total_dv01,
        unit="USD/bp",
        methodology="Sum of position DV01s (parallel shift approximation)",
        inputs_hash=h,
        metadata={
            "bond_positions": bonds["ticker"].tolist(),
            "LIMITATION": LIMITATION_NOTE,
        },
    ))
    return outputs


def compute_key_rate_duration(
    portfolio_df: pd.DataFrame,
    yield_curve: pd.DataFrame,
    shock_bps: float = 1.0,
) -> list[RiskOutput]:
    """
    Key Rate Duration (KRD) across 2Y, 5Y, 10Y, 30Y buckets.
    Each bucket shocked independently; other maturities held constant.
    Returns KRD in years per bucket, per position and aggregate.

    Interpretation: KRD_10Y = 5 means a 1bp shock to the 10Y rate
    causes approximately $5 × shock_bps dollar P&L change per $10,000 face.
    """
    h = hash_dataframe(yield_curve)
    bonds = _get_bond_positions(portfolio_df)

    # Piecewise linear interpolation weights for each ETF onto KRD buckets
    # Based on ETF duration → maturity bucket attribution
    def krd_weights(dur: float) -> dict[str, float]:
        """Distribute a position's rate sensitivity across KRD buckets."""
        if dur <= 2.0:
            return {"2Y": 1.0, "5Y": 0.0, "10Y": 0.0, "30Y": 0.0}
        elif dur <= 5.0:
            alpha = (dur - 2.0) / 3.0
            return {"2Y": 1 - alpha, "5Y": alpha, "10Y": 0.0, "30Y": 0.0}
        elif dur <= 10.0:
            alpha = (dur - 5.0) / 5.0
            return {"2Y": 0.0, "5Y": 1 - alpha, "10Y": alpha, "30Y": 0.0}
        else:
            alpha = min((dur - 10.0) / 20.0, 1.0)
            return {"2Y": 0.0, "5Y": 0.0, "10Y": 1 - alpha, "30Y": alpha}

    outputs = []
    portfolio_krd: dict[str, float] = {b: 0.0 for b in KRD_BUCKETS}

    for _, row in bonds.iterrows():
        ticker = row["ticker"]
        mac_dur = row["duration_years"]
        mv = row["market_value_usd"]

        dur = mac_dur
        if dur <= 2.5:
            ytm = _current_yield(yield_curve, "2Y")
        elif dur <= 6.0:
            ytm = _current_yield(yield_curve, "5Y")
        elif dur <= 12.0:
            ytm = _current_yield(yield_curve, "10Y")
        else:
            ytm = _current_yield(yield_curve, "30Y")

        mod_dur = mac_dur / (1 + ytm / 2)
        wts = krd_weights(mac_dur)

        position_krd = {}
        for bucket, bkt_wt in wts.items():
            # KRD in years for this bucket = mod_dur × attribution weight
            krd_years = mod_dur * bkt_wt
            # DV01-equivalent: $ loss per bp shock at this bucket
            dollar_krd = krd_years * mv * 0.0001 * shock_bps
            position_krd[bucket] = krd_years
            portfolio_krd[bucket] += dollar_krd

        outputs.append(RiskOutput(
            metric_name=f"KRD — {ticker}",
            value=mod_dur,
            unit="years",
            methodology=f"Key Rate Duration ({', '.join(KRD_BUCKETS)} buckets, {shock_bps}bp shock)",
            inputs_hash=h,
            metadata={
                "ticker": ticker,
                "krd_by_bucket": position_krd,
                "market_value_usd": mv,
                "LIMITATION": LIMITATION_NOTE,
            },
        ))

    for bucket, dollar_impact in portfolio_krd.items():
        outputs.append(RiskOutput(
            metric_name=f"Portfolio KRD — {bucket}",
            value=dollar_impact,
            unit="USD/bp",
            methodology=f"Aggregated key rate DV01 at {bucket} bucket",
            inputs_hash=h,
            metadata={
                "bucket": bucket,
                "shock_bps": shock_bps,
                "LIMITATION": LIMITATION_NOTE,
            },
        ))
    return outputs


def compute_convexity(
    portfolio_df: pd.DataFrame,
    yield_curve: pd.DataFrame,
) -> list[RiskOutput]:
    """
    Convexity — second-order price sensitivity to yield changes.
    Approximated as Duration² / (1 + y)² for ETF proxies.
    Convexity correction: ΔP/P ≈ -D×Δy + ½×C×(Δy)²
    """
    h = hash_dataframe(yield_curve)
    bonds = _get_bond_positions(portfolio_df)
    outputs = []
    total_mv = bonds["market_value_usd"].sum()
    weighted_convexity = 0.0

    for _, row in bonds.iterrows():
        ticker = row["ticker"]
        mac_dur = row["duration_years"]
        mv = row["market_value_usd"]

        dur = mac_dur
        if dur <= 2.5:
            ytm = _current_yield(yield_curve, "2Y")
        elif dur <= 6.0:
            ytm = _current_yield(yield_curve, "5Y")
        elif dur <= 12.0:
            ytm = _current_yield(yield_curve, "10Y")
        else:
            ytm = _current_yield(yield_curve, "30Y")

        mod_dur = mac_dur / (1 + ytm / 2)
        # Approximation: convexity ≈ (mod_dur)² + mod_dur (bullet bond approximation)
        convexity = mod_dur ** 2 + mod_dur
        weight = mv / total_mv
        weighted_convexity += weight * convexity

        outputs.append(RiskOutput(
            metric_name=f"Convexity — {ticker}",
            value=convexity,
            unit="years²",
            methodology="Duration-squared approximation (ETF proxy)",
            inputs_hash=h,
            metadata={
                "ticker": ticker,
                "modified_duration": mod_dur,
                "ytm_used": ytm,
                "market_value_usd": mv,
                "LIMITATION": LIMITATION_NOTE,
            },
        ))

    outputs.append(RiskOutput(
        metric_name="Portfolio Convexity",
        value=weighted_convexity,
        unit="years²",
        methodology="Value-weighted convexity (ETF proxies)",
        inputs_hash=h,
        metadata={
            "bond_positions": bonds["ticker"].tolist(),
            "LIMITATION": LIMITATION_NOTE,
        },
    ))
    return outputs
