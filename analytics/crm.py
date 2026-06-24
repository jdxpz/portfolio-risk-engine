"""
CRM aggregator — Comprehensive Risk Measure.
Formula: CRM = VaR + Stressed VaR + Incremental Default Risk + Liquidity Add-on
Mirrors Basel III / FRTB internal models approach.
All components traceable to source RiskOutput objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from data.contracts import ContractError, RiskOutput


# Liquidity add-on: bid-ask spread × position size
# Spread assumptions by asset class (in pct of market value)
DEFAULT_LIQUIDITY_SPREADS: dict[str, float] = {
    "EQUITY":    0.0005,   # 5bps
    "BOND":      0.0015,   # 15bps (wider for OTC bond market)
    "FX":        0.0002,   # 2bps
    "COMMODITY": 0.0010,   # 10bps
    "CASH":      0.0001,   # 1bp
}


@dataclass
class CRMBreakdown:
    var_component: float
    stressed_var_component: float
    default_risk_component: float
    liquidity_addon: float
    total_crm: float
    computed_at: datetime = field(default_factory=datetime.utcnow)
    risk_limit_usd: float | None = None
    limit_utilisation: float | None = None
    breach_flags: list[str] = field(default_factory=list)
    component_sources: dict[str, str] = field(default_factory=dict)

    def to_risk_output(self) -> RiskOutput:
        return RiskOutput(
            metric_name="Comprehensive Risk Measure (CRM)",
            value=self.total_crm,
            unit="USD",
            methodology=(
                "CRM = VaR + Stressed VaR + Incremental Default Risk + Liquidity Add-on. "
                "Mirrors Basel III/FRTB internal models approach. "
                "See metadata for component breakdown."
            ),
            computed_at=self.computed_at,
            metadata={
                "var_component": self.var_component,
                "stressed_var_component": self.stressed_var_component,
                "default_risk_component": self.default_risk_component,
                "liquidity_addon": self.liquidity_addon,
                "total_crm": self.total_crm,
                "risk_limit_usd": self.risk_limit_usd,
                "limit_utilisation_pct": (
                    self.limit_utilisation * 100 if self.limit_utilisation else None
                ),
                "breach_flags": self.breach_flags,
                "component_sources": self.component_sources,
                "formula": "CRM = VaR + sVaR + IDR + LiqAddon",
            },
        )

    def utilisation_rag(self) -> str:
        """Red / Amber / Green based on limit utilisation."""
        if self.limit_utilisation is None:
            return "GREY"
        if self.limit_utilisation >= 1.0:
            return "RED"
        if self.limit_utilisation >= 0.75:
            return "AMBER"
        return "GREEN"


def compute_liquidity_addon(
    portfolio_df: pd.DataFrame,
    spread_overrides: dict[str, float] | None = None,
) -> float:
    """
    Liquidity add-on = sum over positions of (bid-ask spread × market value × ½).
    The ½ reflects half-spread cost (one-way liquidation cost).
    """
    spreads = {**DEFAULT_LIQUIDITY_SPREADS, **(spread_overrides or {})}
    addon = 0.0
    for _, row in portfolio_df.iterrows():
        cls = row["asset_class"]
        mv = row["market_value_usd"]
        spread = spreads.get(cls, 0.001)
        addon += spread * mv * 0.5
    return addon


def aggregate_crm(
    var_outputs: dict[str, list[RiskOutput]],
    stressed_var_pnl: float,
    default_risk_output: RiskOutput | None,
    portfolio_df: pd.DataFrame,
    risk_limit_usd: float | None = None,
    var_confidence: float = 0.99,
    spread_overrides: dict[str, float] | None = None,
) -> CRMBreakdown:
    """
    Aggregate all risk components into a single CRM number.

    var_outputs:       output of var_engine.compute_all_var()
    stressed_var_pnl:  worst-case scenario P&L from stress_engine (absolute value)
    default_risk_output: RiskOutput from default_risk.simulate_default_loss (99th percentile)
    portfolio_df:      Portfolio contract DataFrame
    risk_limit_usd:    optional VaR limit for utilisation tracking
    """
    # Extract base VaR at chosen confidence (Monte Carlo preferred)
    conf_key = f"monte_carlo_{int(var_confidence * 100)}"
    if conf_key not in var_outputs:
        conf_key = f"historical_{int(var_confidence * 100)}"
    if conf_key not in var_outputs:
        raise ContractError(
            f"CRM: VaR output for confidence {var_confidence} not found in var_outputs. "
            f"Available keys: {list(var_outputs.keys())}"
        )

    var_component = var_outputs[conf_key][0].value
    var_source = var_outputs[conf_key][0].methodology

    # Stressed VaR (worst historical scenario loss, absolute)
    stressed_var_component = abs(stressed_var_pnl)

    # Incremental default risk
    if default_risk_output is not None:
        default_risk_component = default_risk_output.value
        default_source = default_risk_output.methodology
    else:
        default_risk_component = 0.0
        default_source = "Not computed (no bond positions with PD assumptions)"

    # Liquidity add-on
    liquidity_addon = compute_liquidity_addon(portfolio_df, spread_overrides)

    total_crm = (
        var_component
        + stressed_var_component
        + default_risk_component
        + liquidity_addon
    )

    # Limit monitoring
    utilisation = total_crm / risk_limit_usd if risk_limit_usd and risk_limit_usd > 0 else None

    # Breach detection
    breach_flags = []
    if utilisation is not None:
        if utilisation >= 1.0:
            breach_flags.append(f"CRM LIMIT BREACHED: {utilisation:.1%} utilisation")
        elif utilisation >= 0.90:
            breach_flags.append(f"CRM WARNING: {utilisation:.1%} utilisation (>90% threshold)")

    # Individual component breach checks (heuristic: any single component > 50% of total)
    for name, val in [
        ("VaR", var_component),
        ("Stressed VaR", stressed_var_component),
        ("Default Risk", default_risk_component),
    ]:
        if total_crm > 0 and val / total_crm > 0.70:
            breach_flags.append(f"CONCENTRATION: {name} is {val/total_crm:.0%} of total CRM")

    return CRMBreakdown(
        var_component=var_component,
        stressed_var_component=stressed_var_component,
        default_risk_component=default_risk_component,
        liquidity_addon=liquidity_addon,
        total_crm=total_crm,
        risk_limit_usd=risk_limit_usd,
        limit_utilisation=utilisation,
        breach_flags=breach_flags,
        component_sources={
            "var": var_source,
            "stressed_var": "Worst-case historical scenario P&L (stress_engine)",
            "default_risk": default_source,
            "liquidity": "Bid-ask spread × position size (half-spread convention)",
        },
    )
