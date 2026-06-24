"""
Component VaR — Euler decomposition of portfolio VaR into per-asset contributions.
Marginal VaR × weight = Component VaR. Sum-check enforced against total VaR.
Incremental VaR: VaR delta from adding a hypothetical new position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from data.contracts import ContractError, RiskOutput, hash_dataframe


SUM_TOLERANCE = 0.01   # 1% relative tolerance for component sum check


def component_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    lookback_days: int = 252,
) -> tuple[list[RiskOutput], RiskOutput]:
    """
    Euler decomposition: Component_VaR_i = w_i × (∂VaR/∂w_i).
    Under normality, marginal VaR_i = (Σw)_i / σ_p × z × σ_p = (Σw)_i / σ_p × VaR_p.

    Returns:
      component_outputs: list of RiskOutput, one per asset
      total_check: RiskOutput for sum verification
    """
    h = hash_dataframe(returns)
    rets = returns[weights.index].iloc[-lookback_days:].dropna()
    w = weights.values

    cov = rets.cov().values
    sigma_p = float(np.sqrt(w @ cov @ w))
    if sigma_p == 0:
        raise ContractError("component_var: portfolio has zero variance")

    z = stats.norm.ppf(confidence)
    total_var_usd = z * sigma_p * portfolio_value

    # Marginal contribution vector: ∂σ_p/∂w_i = (Σw)_i / σ_p
    # Component VaR_i = w_i × z × (Σw)_i / σ_p × portfolio_value
    marginal_sigma = (cov @ w) / sigma_p
    component_var_pct = w * z * marginal_sigma
    component_var_usd = component_var_pct * portfolio_value

    outputs = []
    for i, ticker in enumerate(weights.index):
        pct_contribution = float(component_var_usd[i] / total_var_usd) if total_var_usd != 0 else 0.0
        outputs.append(RiskOutput(
            metric_name=f"Component VaR — {ticker}",
            value=float(component_var_usd[i]),
            unit="USD",
            methodology="Euler decomposition (parametric, covariance-based)",
            confidence_level=confidence,
            inputs_hash=h,
            metadata={
                "ticker": ticker,
                "weight": float(w[i]),
                "marginal_var_pct": float(z * marginal_sigma[i]),
                "component_var_pct": float(component_var_pct[i]),
                "pct_of_total_var": pct_contribution,
            },
        ))

    # Sum check
    component_sum = float(component_var_usd.sum())
    relative_error = abs(component_sum - total_var_usd) / (total_var_usd + 1e-10)
    if relative_error > SUM_TOLERANCE:
        raise ContractError(
            f"Component VaR sum check failed: sum={component_sum:.2f}, "
            f"total={total_var_usd:.2f}, rel_error={relative_error:.4%}"
        )

    total_check = RiskOutput(
        metric_name="Component VaR Sum Check",
        value=component_sum,
        unit="USD",
        methodology="Euler decomposition — sum verification",
        confidence_level=confidence,
        inputs_hash=h,
        metadata={
            "total_parametric_var": total_var_usd,
            "relative_error": relative_error,
            "sum_check_passed": True,
        },
    )
    return outputs, total_check


def incremental_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    new_ticker: str,
    new_position_value: float,
    confidence: float = 0.95,
    lookback_days: int = 252,
) -> RiskOutput:
    """
    Incremental VaR: change in total portfolio VaR from adding a new position.
    new_position_value: dollar value of the proposed new position.
    new_ticker must be present in returns.
    """
    h = hash_dataframe(returns)

    if new_ticker not in returns.columns:
        raise ContractError(
            f"incremental_var: '{new_ticker}' not in returns. "
            "Fetch prices for this ticker first."
        )

    tickers_before = list(weights.index)
    tickers_after = tickers_before + ([new_ticker] if new_ticker not in tickers_before else [])

    new_total_value = portfolio_value + new_position_value
    new_weight_val = new_position_value / new_total_value

    # Rescale existing weights
    scale = portfolio_value / new_total_value
    existing_weights = weights.values * scale

    if new_ticker in weights.index:
        idx = list(weights.index).index(new_ticker)
        existing_weights[idx] += new_weight_val
        w_after = pd.Series(existing_weights, index=weights.index)
    else:
        w_after = pd.Series(
            np.append(existing_weights, new_weight_val),
            index=tickers_after,
        )

    rets_before = returns[tickers_before].iloc[-lookback_days:].dropna()
    rets_after = returns[tickers_after].iloc[-lookback_days:].dropna()

    def portfolio_var(r: pd.DataFrame, w: pd.Series, pv: float) -> float:
        cov = r[w.index].cov().values
        sigma = float(np.sqrt(w.values @ cov @ w.values))
        return stats.norm.ppf(confidence) * sigma * pv

    var_before = portfolio_var(rets_before, weights, portfolio_value)
    var_after = portfolio_var(rets_after, w_after, new_total_value)
    incremental = var_after - var_before

    return RiskOutput(
        metric_name=f"Incremental VaR — {new_ticker}",
        value=incremental,
        unit="USD",
        methodology="Parametric VaR delta (before vs after position)",
        confidence_level=confidence,
        inputs_hash=h,
        metadata={
            "new_ticker": new_ticker,
            "new_position_value": new_position_value,
            "var_before": var_before,
            "var_after": var_after,
            "diversification_benefit": var_before + new_position_value * stats.norm.ppf(confidence)
                                       * float(rets_after[new_ticker].std()) - var_after,
        },
    )
