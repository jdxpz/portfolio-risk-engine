"""
Incremental default loss — credit event simulation for bond holdings.
PD and LGD are user-supplied assumptions, clearly labelled.
Bernoulli trial per position; distribution of default losses output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.contracts import ContractError, RiskOutput, hash_dataframe


def simulate_default_loss(
    portfolio_df: pd.DataFrame,
    pd_assumptions: dict[str, float],
    lgd_assumptions: dict[str, float],
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> tuple[RiskOutput, RiskOutput, np.ndarray]:
    """
    Simulate credit event losses across bond positions.

    pd_assumptions:  {ticker: probability_of_default}  e.g. {"HYG": 0.03}
    lgd_assumptions: {ticker: loss_given_default}       e.g. {"HYG": 0.40}

    These are USER-SUPPLIED ASSUMPTIONS — not model outputs. Labelled explicitly
    in all RiskOutput metadata.

    Returns:
      expected_loss: RiskOutput with mean default loss
      var_99_loss:   RiskOutput with 99th percentile default loss
      loss_distribution: np.ndarray of simulated total losses
    """
    bond_positions = portfolio_df[portfolio_df["asset_class"] == "BOND"].copy()
    if bond_positions.empty:
        raise ContractError("simulate_default_loss: no BOND positions in portfolio")

    h = hash_dataframe(bond_positions)
    rng = np.random.default_rng(seed)

    tickers = bond_positions["ticker"].tolist()
    market_values = bond_positions["market_value_usd"].values

    # Validate assumptions provided for at least some bond tickers
    covered = [t for t in tickers if t in pd_assumptions]
    if not covered:
        raise ContractError(
            f"No PD assumptions provided for bond tickers: {tickers}. "
            "Provide pd_assumptions dict with at least one ticker."
        )

    # Build per-position PD and LGD arrays (default 0 for tickers not provided)
    pd_arr = np.array([pd_assumptions.get(t, 0.0) for t in tickers])
    lgd_arr = np.array([lgd_assumptions.get(t, 0.40) for t in tickers])  # 40% LGD default

    # Bernoulli trials: default_matrix[sim, position] = 1 if default
    u = rng.random((n_simulations, len(tickers)))
    default_matrix = (u < pd_arr).astype(float)

    # Loss per position per simulation
    loss_matrix = default_matrix * lgd_arr * market_values  # shape: (n_sim, n_positions)
    total_losses = loss_matrix.sum(axis=1)                   # shape: (n_sim,)

    expected_loss = float(total_losses.mean())
    var_99 = float(np.percentile(total_losses, 99))

    el_output = RiskOutput(
        metric_name="Expected Default Loss",
        value=expected_loss,
        unit="USD",
        methodology="Bernoulli default simulation",
        inputs_hash=h,
        metadata={
            "n_simulations": n_simulations,
            "bond_positions": tickers,
            "pd_assumptions_USER_SUPPLIED": pd_assumptions,
            "lgd_assumptions_USER_SUPPLIED": lgd_assumptions,
            "WARNING": "PD and LGD are user inputs — not calibrated model estimates",
        },
    )
    var_output = RiskOutput(
        metric_name="Default Loss VaR (99%)",
        value=var_99,
        unit="USD",
        methodology="Bernoulli default simulation — 99th percentile",
        confidence_level=0.99,
        inputs_hash=h,
        metadata={
            "n_simulations": n_simulations,
            "pd_assumptions_USER_SUPPLIED": pd_assumptions,
            "lgd_assumptions_USER_SUPPLIED": lgd_assumptions,
        },
    )
    return el_output, var_output, total_losses
