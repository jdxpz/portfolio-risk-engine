"""
Performance metrics — Sharpe, Sortino, Calmar, max drawdown, P&L attribution.
Rolling + static. Annualised at 252 trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.contracts import ContractError, RiskOutput, hash_dataframe

TRADING_DAYS = 252


def _validate_returns(returns: pd.Series, name: str = "returns") -> None:
    if returns.empty:
        raise ContractError(f"performance.{name}: empty return series")
    if returns.isnull().any():
        raise ContractError(f"performance.{name}: NaNs in return series")


def sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualise: bool = True,
) -> RiskOutput:
    """Annualised Sharpe ratio. risk_free_rate in annual decimal (e.g. 0.05)."""
    _validate_returns(portfolio_returns)
    h = hash_dataframe(portfolio_returns.to_frame())
    rf_daily = risk_free_rate / TRADING_DAYS
    excess = portfolio_returns - rf_daily
    mean_excess = float(excess.mean())
    sigma = float(excess.std())

    if sigma == 0:
        ratio = 0.0
    elif annualise:
        ratio = (mean_excess / sigma) * np.sqrt(TRADING_DAYS)
    else:
        ratio = mean_excess / sigma

    return RiskOutput(
        metric_name="Sharpe Ratio",
        value=ratio,
        unit="ratio",
        methodology=f"Annualised excess return / annualised volatility (rf={risk_free_rate:.2%})",
        inputs_hash=h,
        metadata={
            "risk_free_rate_annual": risk_free_rate,
            "mean_excess_daily": mean_excess,
            "sigma_daily": sigma,
            "n_periods": len(portfolio_returns),
        },
    )


def sortino_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualise: bool = True,
) -> RiskOutput:
    """Sortino ratio — penalises only downside volatility."""
    _validate_returns(portfolio_returns)
    h = hash_dataframe(portfolio_returns.to_frame())
    rf_daily = risk_free_rate / TRADING_DAYS
    excess = portfolio_returns - rf_daily
    downside = excess[excess < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else float(excess.std())

    mean_excess = float(excess.mean())
    if downside_std == 0:
        ratio = 0.0
    elif annualise:
        ratio = (mean_excess / downside_std) * np.sqrt(TRADING_DAYS)
    else:
        ratio = mean_excess / downside_std

    return RiskOutput(
        metric_name="Sortino Ratio",
        value=ratio,
        unit="ratio",
        methodology=f"Annualised excess return / downside deviation (rf={risk_free_rate:.2%})",
        inputs_hash=h,
        metadata={
            "risk_free_rate_annual": risk_free_rate,
            "downside_std_daily": downside_std,
            "n_negative_periods": int(len(downside)),
        },
    )


def max_drawdown(portfolio_returns: pd.Series) -> tuple[RiskOutput, pd.Series]:
    """
    Maximum drawdown: peak-to-trough decline in cumulative return.
    Returns (RiskOutput, drawdown_series) for charting.
    """
    _validate_returns(portfolio_returns)
    h = hash_dataframe(portfolio_returns.to_frame())
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_peak = cumulative.cummax()
    drawdown_series = (cumulative - rolling_peak) / rolling_peak
    mdd = float(drawdown_series.min())

    # Find trough and peak dates
    trough_date = drawdown_series.idxmin()
    peak_date = cumulative[:trough_date].idxmax() if trough_date is not None else None

    # Recovery: first date after trough where cumulative exceeds prior peak
    if trough_date is not None:
        post_trough = cumulative[trough_date:]
        peak_value = rolling_peak[trough_date]
        recovery_mask = post_trough >= peak_value
        recovery_date = post_trough[recovery_mask].index[0] if recovery_mask.any() else None
    else:
        recovery_date = None

    recovery_days = (
        (recovery_date - trough_date).days
        if recovery_date and trough_date
        else None
    )

    return RiskOutput(
        metric_name="Maximum Drawdown",
        value=mdd,
        unit="pct",
        methodology="Peak-to-trough cumulative return decline",
        inputs_hash=h,
        metadata={
            "peak_date": str(peak_date) if peak_date else None,
            "trough_date": str(trough_date) if trough_date else None,
            "recovery_date": str(recovery_date) if recovery_date else None,
            "recovery_days": recovery_days,
        },
    ), drawdown_series


def calmar_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> RiskOutput:
    """Calmar ratio = annualised return / abs(max drawdown)."""
    _validate_returns(portfolio_returns)
    h = hash_dataframe(portfolio_returns.to_frame())

    ann_return = float(portfolio_returns.mean()) * TRADING_DAYS
    mdd_output, _ = max_drawdown(portfolio_returns)
    mdd = abs(mdd_output.value)
    ratio = ann_return / mdd if mdd > 0 else 0.0

    return RiskOutput(
        metric_name="Calmar Ratio",
        value=ratio,
        unit="ratio",
        methodology="Annualised return / |Max drawdown|",
        inputs_hash=h,
        metadata={
            "annualised_return": ann_return,
            "max_drawdown": -mdd,
        },
    )


def rolling_sharpe(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    window: int = 90,
) -> pd.Series:
    """Rolling Sharpe ratio on a window of `window` trading days."""
    _validate_returns(portfolio_returns)
    rf_daily = risk_free_rate / TRADING_DAYS
    excess = portfolio_returns - rf_daily
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    result = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS)
    result.name = f"rolling_sharpe_{window}d"
    return result


def pnl_attribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    asset_classes: pd.Series,
) -> dict[str, RiskOutput]:
    """
    P&L attribution decomposed by asset class and position.
    Returns dict: asset_class → RiskOutput with contribution metrics.

    Methodology: Brinson-style. Each position's contribution = weight × return.
    Attribution by asset class = sum of position contributions within class.
    """
    h = hash_dataframe(returns)
    tickers = weights.index.tolist()
    available = [t for t in tickers if t in returns.columns]

    contributions = {}
    for ticker in available:
        contrib = returns[ticker] * weights[ticker]
        cls = asset_classes.get(ticker, "UNKNOWN")
        if cls not in contributions:
            contributions[cls] = []
        contributions[cls].append(contrib)

    outputs = {}
    for cls, series_list in contributions.items():
        class_contrib = pd.concat(series_list, axis=1).sum(axis=1)
        total_contrib = float(class_contrib.sum())
        ann_contrib = float(class_contrib.mean()) * TRADING_DAYS

        outputs[cls] = RiskOutput(
            metric_name=f"P&L Attribution — {cls}",
            value=ann_contrib,
            unit="pct_annualised",
            methodology="Brinson attribution: weight × return, summed by asset class",
            inputs_hash=h,
            metadata={
                "asset_class": cls,
                "cumulative_contribution": total_contrib,
                "annualised_contribution": ann_contrib,
                "n_positions": len(series_list),
            },
        )
    return outputs


def compute_all_performance(
    portfolio_returns: pd.Series,
    returns: pd.DataFrame,
    weights: pd.Series,
    asset_classes: pd.Series,
    risk_free_rate: float = 0.05,
    rolling_window: int = 90,
) -> dict[str, RiskOutput | pd.Series]:
    """Convenience wrapper: all performance metrics in one call."""
    mdd_out, drawdown_series = max_drawdown(portfolio_returns)
    return {
        "sharpe": sharpe_ratio(portfolio_returns, risk_free_rate),
        "sortino": sortino_ratio(portfolio_returns, risk_free_rate),
        "calmar": calmar_ratio(portfolio_returns, risk_free_rate),
        "max_drawdown": mdd_out,
        "drawdown_series": drawdown_series,
        "rolling_sharpe": rolling_sharpe(portfolio_returns, risk_free_rate, rolling_window),
        "attribution": pnl_attribution(returns, weights, asset_classes),
    }
