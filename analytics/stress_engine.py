"""
Stress engine — historical replay, hypothetical scenario P&L, rate sensitivity shocks.
All modes output a common ScenarioResult for CRM aggregation.
Monte Carlo simulation calls into var_engine to avoid code duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from data.contracts import (
    ContractError,
    RiskOutput,
    ScenarioShock,
    hash_dataframe,
)

LIMITATION_NOTE = (
    "Stress P&L is sensitivity-based (first-order approximation). "
    "Actual P&L in a stress event includes non-linear effects not captured here."
)


@dataclass
class ScenarioResult:
    """Universal output of the stress engine — consumed by CRM aggregator."""
    scenario_name: str
    total_pnl_usd: float
    factor_pnl: dict[str, float]          # {"rates": x, "credit": y, "equity": z, "commodity": w}
    methodology: str
    computed_at: datetime = field(default_factory=datetime.utcnow)
    inputs_hash: str = ""
    probability: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_risk_output(self) -> RiskOutput:
        return RiskOutput(
            metric_name=f"Stress P&L — {self.scenario_name}",
            value=self.total_pnl_usd,
            unit="USD",
            methodology=self.methodology,
            computed_at=self.computed_at,
            inputs_hash=self.inputs_hash,
            metadata={
                "scenario_name": self.scenario_name,
                "factor_pnl": self.factor_pnl,
                "probability": self.probability,
                "LIMITATION": LIMITATION_NOTE,
                **self.metadata,
            },
        )


# ---------------------------------------------------------------------------
# Sensitivity-based P&L engine
# ---------------------------------------------------------------------------

def _rates_pnl(
    portfolio_df: pd.DataFrame,
    yield_curve: pd.DataFrame,
    rate_shift_bps: dict[str, float],
) -> float:
    """
    Dollar P&L from rate shifts using DV01 and KRD distribution.
    Delegates to rates_risk module; isolated here to keep stress_engine pure.
    Returns 0 if no yield curve is available or no bond positions exist.
    """
    from analytics.rates_risk import compute_dv01, compute_key_rate_duration

    bond_positions = portfolio_df[portfolio_df["asset_class"] == "BOND"]
    if bond_positions.empty or yield_curve.empty:
        return 0.0

    # If all maturities shift equally → use aggregate DV01
    shifts = list(rate_shift_bps.values())
    if len(set(shifts)) == 1:
        dv01_outputs = compute_dv01(portfolio_df, yield_curve)
        portfolio_dv01 = next(
            (o.value for o in dv01_outputs if o.metric_name == "Portfolio DV01"), 0.0
        )
        return -portfolio_dv01 * shifts[0]  # negative: rates up → bond prices down

    # Non-parallel: use KRD by bucket
    krd_outputs = compute_key_rate_duration(portfolio_df, yield_curve)
    pnl = 0.0
    for output in krd_outputs:
        if not output.metric_name.startswith("Portfolio KRD —"):
            continue
        bucket = output.metadata.get("bucket", "")
        if bucket in rate_shift_bps:
            shift = rate_shift_bps[bucket]
            pnl += -output.value * shift   # output.value is USD/bp; shift in bps
    return pnl


def _credit_pnl(
    portfolio_df: pd.DataFrame,
    yield_curve: pd.DataFrame,
    spread_shift_bps: float,
) -> float:
    """Dollar P&L from credit spread widening using DV01 equivalent."""
    from analytics.rates_risk import compute_dv01
    if spread_shift_bps == 0:
        return 0.0
    bond_positions = portfolio_df[portfolio_df["asset_class"] == "BOND"]
    if bond_positions.empty or yield_curve.empty:
        return 0.0
    dv01_outputs = compute_dv01(portfolio_df, yield_curve)
    portfolio_dv01 = next(
        (o.value for o in dv01_outputs if o.metric_name == "Portfolio DV01"), 0.0
    )
    return -portfolio_dv01 * spread_shift_bps


def _equity_pnl(portfolio_df: pd.DataFrame, equity_shock_pct: float) -> float:
    """Dollar P&L from equity shock applied to all EQUITY positions."""
    if equity_shock_pct == 0:
        return 0.0
    eq = portfolio_df[portfolio_df["asset_class"] == "EQUITY"]
    return float((eq["market_value_usd"] * equity_shock_pct).sum())


def _commodity_pnl(portfolio_df: pd.DataFrame, commodity_shock_pct: float) -> float:
    """Dollar P&L from commodity shock applied to all COMMODITY positions."""
    if commodity_shock_pct == 0:
        return 0.0
    comm = portfolio_df[portfolio_df["asset_class"] == "COMMODITY"]
    return float((comm["market_value_usd"] * commodity_shock_pct).sum())


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def apply_scenario(
    portfolio_df: pd.DataFrame,
    yield_curve: pd.DataFrame,
    shock: ScenarioShock,
) -> ScenarioResult:
    """
    Apply a ScenarioShock to the portfolio and compute factor-decomposed P&L.
    Works for both historical replay and hypothetical scenarios — same engine.
    """
    h = hash_dataframe(portfolio_df)

    rates_impact = _rates_pnl(portfolio_df, yield_curve, shock.rate_shift_bps)
    credit_impact = _credit_pnl(portfolio_df, yield_curve, shock.credit_spread_shift_bps)
    equity_impact = _equity_pnl(portfolio_df, shock.equity_shock_pct)
    commodity_impact = _commodity_pnl(portfolio_df, shock.commodity_shock_pct)

    total = rates_impact + credit_impact + equity_impact + commodity_impact

    return ScenarioResult(
        scenario_name=shock.name,
        total_pnl_usd=total,
        factor_pnl={
            "rates": rates_impact,
            "credit": credit_impact,
            "equity": equity_impact,
            "commodity": commodity_impact,
        },
        methodology="Sensitivity-based first-order approximation",
        inputs_hash=h,
        probability=shock.probability,
        metadata={"description": shock.description},
    )


def run_all_scenarios(
    portfolio_df: pd.DataFrame,
    yield_curve: pd.DataFrame,
    scenarios: list[ScenarioShock],
) -> list[ScenarioResult]:
    """Run all scenarios and return list of ScenarioResults."""
    return [apply_scenario(portfolio_df, yield_curve, s) for s in scenarios]


def probabilistic_expected_loss(results: list[ScenarioResult]) -> RiskOutput:
    """
    Probability-weighted expected loss across scenarios.
    Only uses scenarios with non-None probability.
    """
    weighted = [r for r in results if r.probability is not None]
    if not weighted:
        raise ContractError("probabilistic_expected_loss: no scenarios have probability assigned")

    total_prob = sum(r.probability for r in weighted)  # type: ignore[arg-type]
    expected_loss = sum(
        r.total_pnl_usd * (r.probability / total_prob)   # type: ignore[operator]
        for r in weighted
    )

    return RiskOutput(
        metric_name="Probabilistic Expected Loss",
        value=expected_loss,
        unit="USD",
        methodology="Probability-weighted scenario P&L (base rate probabilities)",
        metadata={
            "n_scenarios": len(weighted),
            "total_probability_weight": total_prob,
            "scenarios": [r.scenario_name for r in weighted],
            "WARNING": (
                "Scenario probabilities are indicative base rates, not calibrated model outputs. "
                "Treat as directional, not precise."
            ),
        },
    )


def monte_carlo_loss_distribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    n_simulations: int = 10_000,
    student_t_df: int = 5,
    lookback_days: int = 252,
    confidence_levels: list[float] = [0.95, 0.99],
    seed: int | None = 42,
) -> tuple[list[RiskOutput], np.ndarray]:
    """
    MC loss distribution — delegates to var_engine.monte_carlo_var.
    Returns (list of VaR/CVaR at each confidence level, simulated returns array).
    Shared code path: stress_engine calls var_engine, not re-implemented.
    """
    from analytics.var_engine import monte_carlo_var

    outputs = []
    sim_rets = None
    for conf in confidence_levels:
        var_out, cvar_out, sim_rets = monte_carlo_var(
            returns, weights, portfolio_value, conf,
            n_simulations, student_t_df, lookback_days, seed,
        )
        outputs.extend([var_out, cvar_out])

    return outputs, sim_rets  # type: ignore[return-value]
