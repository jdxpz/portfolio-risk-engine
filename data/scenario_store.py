"""
Scenario store — hardcoded historical shock vectors + user-defined scenario registry.

Historical shock sources:
  GFC:          S&P500 peak-to-trough Oct 2007–Mar 2009; 10Y UST rally ~200bps;
                IG spreads +400bps, HY spreads +1500bps; oil -70%
  COVID:        Feb 19 – Mar 23 2020; S&P500 -34%; 10Y UST -100bps;
                HY spreads +700bps; oil -55%
  Rate shock:   Jan–Oct 2022; 10Y UST +250bps; S&P500 -25%; IG spreads +100bps
  Taper tantrum: May–Sep 2013; 10Y UST +130bps; S&P500 -5%; EM equities -15%
  Dot-com:      Mar 2000–Oct 2002; Nasdaq -78%, S&P -49%; 10Y UST -150bps
"""

from __future__ import annotations

from data.contracts import ScenarioShock

# ---------------------------------------------------------------------------
# Historical scenarios
# ---------------------------------------------------------------------------

HISTORICAL_SCENARIOS: list[ScenarioShock] = [
    ScenarioShock(
        name="GFC (2007–2009)",
        description="Global Financial Crisis peak-to-trough. Equities -55%, credit collapse, rates rally.",
        rate_shift_bps={
            "2Y": -200, "5Y": -180, "10Y": -150, "20Y": -100, "30Y": -80,
        },
        credit_spread_shift_bps=800.0,    # blended IG/HY widening
        equity_shock_pct=-0.55,
        commodity_shock_pct=-0.70,        # oil peak-to-trough
        probability=0.02,
    ),
    ScenarioShock(
        name="COVID Crash (Feb–Mar 2020)",
        description="Pandemic shock. S&P -34% in 33 days, rate flight-to-quality, HY spreads +700bps.",
        rate_shift_bps={
            "1M": -150, "3M": -150, "6M": -130, "1Y": -120,
            "2Y": -100, "5Y": -90, "10Y": -80, "20Y": -60, "30Y": -50,
        },
        credit_spread_shift_bps=500.0,
        equity_shock_pct=-0.34,
        commodity_shock_pct=-0.55,
        probability=0.03,
    ),
    ScenarioShock(
        name="2022 Rate Shock",
        description="Fastest Fed hiking cycle in 40 years. Rates +250bps, bonds -20%, equities -25%.",
        rate_shift_bps={
            "1M": 400, "3M": 420, "6M": 380, "1Y": 340,
            "2Y": 300, "5Y": 260, "10Y": 240, "20Y": 210, "30Y": 190,
        },
        credit_spread_shift_bps=120.0,
        equity_shock_pct=-0.25,
        commodity_shock_pct=0.40,         # energy spike
        probability=0.05,
    ),
    ScenarioShock(
        name="Taper Tantrum (2013)",
        description="Fed signals QE tapering. 10Y +130bps in 4 months. EM equities -15%.",
        rate_shift_bps={
            "2Y": 70, "5Y": 110, "10Y": 130, "20Y": 120, "30Y": 100,
        },
        credit_spread_shift_bps=60.0,
        equity_shock_pct=-0.06,
        commodity_shock_pct=-0.12,
        probability=0.08,
    ),
    ScenarioShock(
        name="Dot-com Unwind (2000–2002)",
        description="Tech bust. Nasdaq -78%, S&P -49%. Rates rally as Fed cuts 500bps.",
        rate_shift_bps={
            "2Y": -300, "5Y": -250, "10Y": -200, "20Y": -150, "30Y": -120,
        },
        credit_spread_shift_bps=400.0,
        equity_shock_pct=-0.49,
        commodity_shock_pct=-0.25,
        probability=0.03,
    ),
]

# ---------------------------------------------------------------------------
# Hypothetical sensitivity scenarios (rate curve shapes)
# ---------------------------------------------------------------------------

RATE_SENSITIVITY_SCENARIOS: list[ScenarioShock] = [
    ScenarioShock(
        name="Parallel Shift +100bps",
        description="All maturities shift up 100bps simultaneously.",
        rate_shift_bps={m: 100 for m in ["1M","3M","6M","1Y","2Y","5Y","7Y","10Y","20Y","30Y"]},
        credit_spread_shift_bps=0.0,
        equity_shock_pct=0.0,
        commodity_shock_pct=0.0,
    ),
    ScenarioShock(
        name="Parallel Shift -100bps",
        description="All maturities shift down 100bps simultaneously.",
        rate_shift_bps={m: -100 for m in ["1M","3M","6M","1Y","2Y","5Y","7Y","10Y","20Y","30Y"]},
        credit_spread_shift_bps=0.0,
        equity_shock_pct=0.0,
        commodity_shock_pct=0.0,
    ),
    ScenarioShock(
        name="Bear Steepener",
        description="Short end anchored, long end sells off. 10Y +150bps, 2Y +20bps.",
        rate_shift_bps={"1M": 0, "3M": 5, "6M": 10, "1Y": 20, "2Y": 30,
                        "5Y": 80, "7Y": 110, "10Y": 150, "20Y": 160, "30Y": 170},
        credit_spread_shift_bps=30.0,
        equity_shock_pct=-0.05,
        commodity_shock_pct=0.0,
    ),
    ScenarioShock(
        name="Bull Steepener",
        description="Short end rallies, long end stable. 2Y -150bps, 10Y -20bps.",
        rate_shift_bps={"1M": -200, "3M": -190, "6M": -170, "1Y": -150,
                        "2Y": -120, "5Y": -60, "7Y": -30, "10Y": -20, "20Y": -10, "30Y": 0},
        credit_spread_shift_bps=-20.0,
        equity_shock_pct=0.05,
        commodity_shock_pct=0.0,
    ),
    ScenarioShock(
        name="Bear Flattener",
        description="Short end rises sharply, long end relatively stable. Curve inverts.",
        rate_shift_bps={"1M": 200, "3M": 180, "6M": 160, "1Y": 140,
                        "2Y": 120, "5Y": 60, "7Y": 30, "10Y": 20, "20Y": 10, "30Y": 5},
        credit_spread_shift_bps=50.0,
        equity_shock_pct=-0.08,
        commodity_shock_pct=0.0,
    ),
    ScenarioShock(
        name="Twist (Barbell Squeeze)",
        description="Middle of curve sells off; wings rally. 5Y +100bps, 2Y -50bps, 30Y -30bps.",
        rate_shift_bps={"1M": -80, "3M": -70, "6M": -50, "1Y": -30,
                        "2Y": -20, "5Y": 100, "7Y": 80, "10Y": 60, "20Y": 10, "30Y": -30},
        credit_spread_shift_bps=0.0,
        equity_shock_pct=0.0,
        commodity_shock_pct=0.0,
    ),
]

# ---------------------------------------------------------------------------
# All scenarios accessible as a single registry
# ---------------------------------------------------------------------------

ALL_SCENARIOS: list[ScenarioShock] = HISTORICAL_SCENARIOS + RATE_SENSITIVITY_SCENARIOS

_user_scenarios: list[ScenarioShock] = []


def register_scenario(shock: ScenarioShock) -> None:
    """Register a user-defined scenario at runtime."""
    _user_scenarios.append(shock)


def get_all_scenarios() -> list[ScenarioShock]:
    return ALL_SCENARIOS + _user_scenarios


def get_scenario_by_name(name: str) -> ScenarioShock:
    for s in get_all_scenarios():
        if s.name == name:
            return s
    raise KeyError(f"Scenario '{name}' not found. Available: {[s.name for s in get_all_scenarios()]}")
