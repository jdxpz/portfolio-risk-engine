"""
VaR engine — Historical, Parametric, and Monte Carlo VaR + CVaR.
All three methods return RiskOutput with the same interface.
Monte Carlo uses Cholesky decomposition + Student-t innovations (df=5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from data.contracts import RiskOutput, hash_dataframe


N_SIMULATIONS = 10_000
STUDENT_T_DF = 5
TRADING_DAYS = 252


def _make_output(
    metric: str,
    value: float,
    confidence: float,
    methodology: str,
    inputs_hash: str,
    unit: str = "USD",
    metadata: dict | None = None,
) -> RiskOutput:
    return RiskOutput(
        metric_name=metric,
        value=value,
        unit=unit,
        methodology=methodology,
        confidence_level=confidence,
        inputs_hash=inputs_hash,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Historical VaR
# ---------------------------------------------------------------------------

def historical_var(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    lookback_days: int = 252,
) -> tuple[RiskOutput, RiskOutput]:
    """
    Historical simulation VaR and CVaR.
    No distributional assumption — reads off empirical quantile.
    Returns (VaR, CVaR) as RiskOutput pair.
    """
    h = hash_dataframe(portfolio_returns.to_frame())
    rets = portfolio_returns.iloc[-lookback_days:].dropna().values

    var_pct = float(np.percentile(rets, (1 - confidence) * 100))
    var_usd = abs(var_pct) * portfolio_value

    tail = rets[rets <= var_pct]
    cvar_pct = float(tail.mean()) if len(tail) > 0 else var_pct
    cvar_usd = abs(cvar_pct) * portfolio_value

    var_out = _make_output(
        "Historical VaR", var_usd, confidence,
        f"Historical simulation ({lookback_days}d lookback)", h,
        metadata={"var_pct": var_pct, "lookback_days": lookback_days},
    )
    cvar_out = _make_output(
        "Historical CVaR (ES)", cvar_usd, confidence,
        f"Historical simulation ({lookback_days}d lookback) — tail average", h,
        metadata={"cvar_pct": cvar_pct, "n_tail_obs": int(len(tail))},
    )
    return var_out, cvar_out


# ---------------------------------------------------------------------------
# Parametric VaR
# ---------------------------------------------------------------------------

def parametric_var(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    lookback_days: int = 252,
) -> tuple[RiskOutput, RiskOutput]:
    """
    Parametric (variance-covariance) VaR assuming normality.
    Fast analytic solution — underestimates tail risk.
    Returns (VaR, CVaR).
    """
    h = hash_dataframe(portfolio_returns.to_frame())
    rets = portfolio_returns.iloc[-lookback_days:].dropna()

    mu = float(rets.mean())
    sigma = float(rets.std())
    z = stats.norm.ppf(1 - confidence)

    var_pct = mu + z * sigma          # negative number
    var_usd = abs(var_pct) * portfolio_value

    # CVaR = E[X | X < VaR] under normality
    phi = stats.norm.pdf(z)
    cvar_pct = mu - sigma * phi / (1 - confidence)
    cvar_usd = abs(cvar_pct) * portfolio_value

    var_out = _make_output(
        "Parametric VaR", var_usd, confidence,
        "Variance-covariance (normal distribution)", h,
        metadata={"mu_daily": mu, "sigma_daily": sigma, "z_score": z},
    )
    cvar_out = _make_output(
        "Parametric CVaR (ES)", cvar_usd, confidence,
        "Variance-covariance (normal distribution) — analytic tail", h,
        metadata={"mu_daily": mu, "sigma_daily": sigma},
    )
    return var_out, cvar_out


# ---------------------------------------------------------------------------
# Monte Carlo VaR
# ---------------------------------------------------------------------------

def monte_carlo_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    n_simulations: int = N_SIMULATIONS,
    student_t_df: int = STUDENT_T_DF,
    lookback_days: int = 252,
    seed: int | None = 42,
) -> tuple[RiskOutput, RiskOutput, np.ndarray]:
    """
    Monte Carlo VaR using Cholesky decomposition + Student-t innovations.
    Returns (VaR, CVaR, simulated_portfolio_returns array) for downstream use.

    The simulated returns array is shared with stress_engine to avoid duplicate simulation.
    """
    h = hash_dataframe(returns)
    rets = returns.iloc[-lookback_days:].dropna()

    aligned = rets[weights.index]
    cov = aligned.cov().values
    mu = aligned.mean().values
    w = weights.values

    # Cholesky decomposition — requires positive definite matrix
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Regularise if not PD (common with correlated assets)
        cov_reg = cov + np.eye(len(cov)) * 1e-8
        L = np.linalg.cholesky(cov_reg)

    rng = np.random.default_rng(seed)

    # Student-t innovations: scale to unit variance
    t_scale = np.sqrt((student_t_df - 2) / student_t_df)
    innovations = rng.standard_t(df=student_t_df, size=(n_simulations, len(weights))) / t_scale

    # Correlated shocks: Z = innovations @ L.T + mu
    sim_asset_returns = innovations @ L.T + mu

    # Portfolio returns for each simulation
    sim_port_returns = sim_asset_returns @ w

    var_pct = float(np.percentile(sim_port_returns, (1 - confidence) * 100))
    var_usd = abs(var_pct) * portfolio_value

    tail = sim_port_returns[sim_port_returns <= var_pct]
    cvar_pct = float(tail.mean()) if len(tail) > 0 else var_pct
    cvar_usd = abs(cvar_pct) * portfolio_value

    var_out = _make_output(
        "Monte Carlo VaR", var_usd, confidence,
        f"Monte Carlo ({n_simulations:,} paths, Student-t df={student_t_df}, Cholesky)", h,
        metadata={
            "n_simulations": n_simulations,
            "student_t_df": student_t_df,
            "var_pct": var_pct,
            "seed": seed,
        },
    )
    cvar_out = _make_output(
        "Monte Carlo CVaR (ES)", cvar_usd, confidence,
        f"Monte Carlo ({n_simulations:,} paths, Student-t df={student_t_df}) — tail average", h,
        metadata={"cvar_pct": cvar_pct, "n_tail_obs": int(len(tail))},
    )
    return var_out, cvar_out, sim_port_returns


# ---------------------------------------------------------------------------
# Run all three methods at once
# ---------------------------------------------------------------------------

def compute_all_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_levels: list[float] = [0.95, 0.99],
    lookback_days: int = 252,
) -> dict[str, list[RiskOutput]]:
    """
    Compute VaR and CVaR for all three methodologies at each confidence level.
    Returns dict keyed by methodology name, value is [VaR, CVaR].
    Also returns Monte Carlo simulated returns for downstream modules.
    """
    port_rets = (returns[weights.index] @ weights)
    port_rets.name = "portfolio"

    results: dict[str, list[RiskOutput]] = {}
    mc_sim_cache: dict[float, np.ndarray] = {}

    for conf in confidence_levels:
        suffix = f"_{int(conf*100)}"

        h_var, h_cvar = historical_var(port_rets, portfolio_value, conf, lookback_days)
        results[f"historical{suffix}"] = [h_var, h_cvar]

        p_var, p_cvar = parametric_var(port_rets, portfolio_value, conf, lookback_days)
        results[f"parametric{suffix}"] = [p_var, p_cvar]

        mc_var, mc_cvar, sim_rets = monte_carlo_var(
            returns, weights, portfolio_value, conf, lookback_days=lookback_days
        )
        results[f"monte_carlo{suffix}"] = [mc_var, mc_cvar]
        mc_sim_cache[conf] = sim_rets

    results["_mc_simulations"] = mc_sim_cache   # type: ignore[assignment]
    return results
