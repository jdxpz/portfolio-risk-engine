"""
Data contracts for PRE — Portfolio Risk Engine.
All module boundaries exchange these exact schemas. Violations raise ContractError at the boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    BOND = "BOND"
    FX = "FX"
    COMMODITY = "COMMODITY"
    CASH = "CASH"


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

class ContractError(ValueError):
    """Raised when a data contract is violated at a module boundary."""


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(f"{name}: missing columns {missing}")


def _require_no_nans(df: pd.DataFrame, name: str) -> None:
    if df.isnull().any().any():
        bad = df.columns[df.isnull().any()].tolist()
        raise ContractError(f"{name}: NaNs found in columns {bad}")


def _require_datetime_index(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ContractError(f"{name}: index must be DatetimeIndex, got {type(df.index)}")


# ---------------------------------------------------------------------------
# RawPrices
# ---------------------------------------------------------------------------

def validate_raw_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Daily adjusted close prices. NaNs permitted here."""
    _require_datetime_index(df, "RawPrices")
    if df.empty:
        raise ContractError("RawPrices: DataFrame is empty")
    if not all(df.dtypes == "float64"):
        df = df.astype("float64")
    return df


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def validate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns. No NaNs permitted."""
    _require_datetime_index(df, "Returns")
    _require_no_nans(df, "Returns")
    if df.empty:
        raise ContractError("Returns: DataFrame is empty")
    return df


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

PORTFOLIO_COLUMNS = ["ticker", "asset_class", "quantity", "cost_basis_usd", "weight", "currency"]


def validate_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, PORTFOLIO_COLUMNS, "Portfolio")
    weight_sum = df["weight"].sum()
    if abs(weight_sum - 1.0) > 0.001:
        raise ContractError(f"Portfolio: weights sum to {weight_sum:.6f}, expected 1.0 ± 0.001")
    valid_classes = {a.value for a in AssetClass}
    invalid = df[~df["asset_class"].isin(valid_classes)]["asset_class"].unique()
    if len(invalid):
        raise ContractError(f"Portfolio: unknown asset_class values {invalid.tolist()}")
    return df


# ---------------------------------------------------------------------------
# YieldCurve
# ---------------------------------------------------------------------------

YIELD_CURVE_MATURITIES = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]


def validate_yield_curve(df: pd.DataFrame) -> pd.DataFrame:
    _require_datetime_index(df, "YieldCurve")
    _require_columns(df, YIELD_CURVE_MATURITIES, "YieldCurve")
    _require_no_nans(df, "YieldCurve")
    if df.max().max() > 1.0:
        raise ContractError("YieldCurve: values must be in decimal (e.g. 0.045), not percentage")
    return df


# ---------------------------------------------------------------------------
# ScenarioShock
# ---------------------------------------------------------------------------

@dataclass
class ScenarioShock:
    name: str
    rate_shift_bps: dict[str, float]          # keys must be subset of YIELD_CURVE_MATURITIES
    credit_spread_shift_bps: float            # uniform shift applied to bond holdings
    equity_shock_pct: float                   # e.g. -0.20 for -20%
    commodity_shock_pct: float                # e.g. -0.30 for -30%
    probability: Optional[float] = None       # optional base-rate probability for weighting
    description: str = ""

    def __post_init__(self) -> None:
        invalid_keys = [k for k in self.rate_shift_bps if k not in YIELD_CURVE_MATURITIES]
        if invalid_keys:
            raise ContractError(
                f"ScenarioShock '{self.name}': invalid maturity keys {invalid_keys}. "
                f"Must be subset of {YIELD_CURVE_MATURITIES}"
            )
        if self.probability is not None and not (0.0 <= self.probability <= 1.0):
            raise ContractError(f"ScenarioShock '{self.name}': probability must be in [0, 1]")


# ---------------------------------------------------------------------------
# RiskOutput
# ---------------------------------------------------------------------------

@dataclass
class RiskOutput:
    metric_name: str
    value: float
    unit: str                                 # e.g. "USD", "pct", "years", "bps"
    methodology: str
    computed_at: datetime = field(default_factory=datetime.utcnow)
    confidence_level: Optional[float] = None  # e.g. 0.95, 0.99
    inputs_hash: Optional[str] = None         # SHA-256 of input DataFrame
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "methodology": self.methodology,
            "computed_at": self.computed_at.isoformat(),
            "confidence_level": self.confidence_level,
            "inputs_hash": self.inputs_hash,
            "metadata": self.metadata,
        }


def hash_dataframe(df: pd.DataFrame) -> str:
    """SHA-256 hash of a DataFrame for audit trail."""
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()
