"""
FRED yield curve fetcher — U.S. Treasury par yields + USREC recession flags.

Reliability: live FRED calls are retried with backoff; on failure
fetch_yield_curve falls back to a committed snapshot in data/fallback/yields.parquet.
Provenance is recorded in df.attrs["source"] in {"live", "cache", "fallback"}.

Returns YieldCurve contract: DatetimeIndex, decimal values, no NaNs.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data.contracts import (
    YIELD_CURVE_MATURITIES,
    ContractError,
    validate_yield_curve,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "yields"
FALLBACK_PATH = Path(__file__).parent / "fallback" / "yields.parquet"
CACHE_TTL_HOURS = 24
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0

# FRED series IDs -> our maturity labels
FRED_SERIES = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1":   "1Y",
    "DGS2":   "2Y",
    "DGS5":   "5Y",
    "DGS7":   "7Y",
    "DGS10":  "10Y",
    "DGS20":  "20Y",
    "DGS30":  "30Y",
}


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _tag(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df.attrs["source"] = source
    df.attrs["data_date"] = df.index[-1].strftime("%Y-%m-%d") if len(df.index) else None
    return df


def _load_fallback() -> pd.DataFrame | None:
    if not FALLBACK_PATH.exists():
        return None
    df = pd.read_parquet(FALLBACK_PATH)
    df.index = pd.to_datetime(df.index)
    return df


def _fetch_fred(api_key: str, start: str, end: str) -> pd.DataFrame:
    """One live pull of the full curve from FRED. Raises on any series failure."""
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ContractError("fredapi not installed. Run: pip install fredapi") from exc

    fred = Fred(api_key=api_key)
    frames = []
    for series_id, label in FRED_SERIES.items():
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s.name = label
        frames.append(s)

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df[YIELD_CURVE_MATURITIES]      # enforce column order
    df = df.ffill().dropna()             # fill weekends/holidays, drop leading gaps
    df = df / 100.0                      # FRED returns percent; contract wants decimal
    return df


def fetch_yield_curve(
    start: str,
    end: str | None = None,
    fred_api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch the U.S. Treasury yield curve from FRED (decimal values, 0.045 = 4.5%).

    Resolution order: fresh cache -> live FRED (retried) -> committed fallback.
    FRED_API_KEY env var is used if fred_api_key is not supplied.
    """
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    cache_path = CACHE_DIR / f"yields_{start}_{end}.parquet"

    if _cache_is_fresh(cache_path):
        logger.info("Cache hit: %s", cache_path.name)
        return _tag(validate_yield_curve(pd.read_parquet(cache_path)), "cache")

    api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    last_exc: Exception | None = None

    if api_key:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("Fetching yield curve from FRED (attempt %d/%d)",
                            attempt, MAX_RETRIES)
                df = _fetch_fred(api_key, start, end)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
                logger.info("Yield curve cached: %s rows", len(df))
                return _tag(validate_yield_curve(df), "live")
            except Exception as exc:
                last_exc = exc
                logger.warning("FRED attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SEC * attempt)
        logger.error("FRED unavailable after %d attempts: %s", MAX_RETRIES, last_exc)

    # No key, or live FRED exhausted — try the committed fallback snapshot.
    fallback = _load_fallback()
    if fallback is not None:
        logger.warning("Serving committed fallback yield snapshot")
        return _tag(validate_yield_curve(fallback), "fallback")

    if not api_key:
        raise ContractError(
            "FRED API key not found. Set FRED_API_KEY environment variable "
            "or pass fred_api_key, and no fallback snapshot is available."
        )
    raise ContractError(
        f"FRED fetch failed after {MAX_RETRIES} attempts and no fallback "
        f"snapshot is available: {last_exc}"
    )


def fetch_recession_flags(
    start: str,
    end: str | None = None,
    fred_api_key: str | None = None,
) -> pd.Series:
    """
    Fetch USREC recession indicator from FRED (0/1 monthly), resampled to
    business-day frequency for chart overlays. Returns an empty Series on any
    failure (the overlay is non-essential).
    """
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("No FRED API key — recession flags unavailable")
        return pd.Series(dtype=float, name="recession")

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        s = fred.get_series("USREC", observation_start=start, observation_end=end)
        s.index = pd.to_datetime(s.index)
        s = s.resample("B").ffill()
        s.name = "recession"
        return s
    except Exception as exc:
        logger.warning("Could not fetch USREC: %s", exc)
        return pd.Series(dtype=float, name="recession")
