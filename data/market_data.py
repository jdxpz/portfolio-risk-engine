"""
Market data ingestion — yfinance prices with parquet cache (24h TTL).

Reliability: live fetches are retried with backoff. If yfinance is unreachable
(e.g. Yahoo throttling a datacenter IP, as commonly happens from cloud hosts),
fetch_prices falls back to a committed snapshot in data/fallback/prices.parquet
so the app always renders. The returned DataFrame records provenance in
df.attrs["source"] in {"live", "cache", "fallback"} and df.attrs["data_date"].

Raises ContractError at the boundary when no data (live or fallback) is available.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from data.contracts import ContractError, validate_raw_prices

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "prices"
FALLBACK_PATH = Path(__file__).parent / "fallback" / "prices.parquet"
CACHE_TTL_HOURS = 24
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


def _cache_path(tickers: list[str], start: str, end: str) -> Path:
    key = "_".join(sorted(tickers)) + f"_{start}_{end}"
    safe = key.replace("-", "").replace(" ", "")[:120]
    return CACHE_DIR / f"{safe}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.utcnow() - datetime.utcfromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the 'Close' price level out of a yfinance download result."""
    # yfinance 1.x always returns MultiIndex columns (Price, Ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"].copy()
        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers[0])
    else:
        # Fallback for any version returning flat columns.
        df = raw[["Close"]].copy()
        df.columns = tickers

    df.index = pd.to_datetime(df.index)
    df = df.astype("float64")

    missing = set(tickers) - set(df.columns)
    if missing:
        logger.warning("Tickers not returned by yfinance: %s", missing)

    return df


def _tag(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Attach provenance after validation (validate may return a fresh frame)."""
    df.attrs["source"] = source
    df.attrs["data_date"] = df.index[-1].strftime("%Y-%m-%d") if len(df.index) else None
    return df


def _load_fallback() -> pd.DataFrame | None:
    if not FALLBACK_PATH.exists():
        return None
    df = pd.read_parquet(FALLBACK_PATH)
    df.index = pd.to_datetime(df.index)
    return df.astype("float64")


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for tickers over [start, end].
    Returns RawPrices contract: DatetimeIndex, float64 columns, NaNs permitted.

    Resolution order: fresh parquet cache -> live yfinance (retried) ->
    committed fallback snapshot. df.attrs records which source was used.
    """
    if not tickers:
        raise ContractError("fetch_prices: tickers list is empty")

    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    cache_path = _cache_path(tickers, start, end)

    if _cache_is_fresh(cache_path):
        logger.info("Cache hit: %s", cache_path.name)
        return _tag(validate_raw_prices(pd.read_parquet(cache_path)), "cache")

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Fetching prices from yfinance (attempt %d/%d): %s",
                        attempt, MAX_RETRIES, tickers)
            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )
            if raw.empty:
                raise ContractError(
                    f"yfinance returned empty data for {tickers} over {start} -> {end}"
                )
            df = _extract_close(raw, tickers)
            # A partial fetch (some tickers rate-limited -> missing or all-NaN)
            # must not be cached or served. Treat it as a failure so we retry and
            # ultimately fall back to the complete committed snapshot.
            incomplete = [t for t in tickers if t not in df.columns or df[t].isna().all()]
            if incomplete:
                raise ContractError(
                    f"yfinance returned incomplete data; missing/empty tickers: {incomplete}"
                )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            logger.info("Cached to %s", cache_path.name)
            return _tag(validate_raw_prices(df), "live")
        except Exception as exc:
            last_exc = exc
            logger.warning("yfinance attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)

    # Live fetch exhausted — fall back to the committed snapshot if present.
    logger.error("yfinance unavailable after %d attempts: %s", MAX_RETRIES, last_exc)
    fallback = _load_fallback()
    if fallback is not None:
        logger.warning("Serving committed fallback price snapshot")
        return _tag(validate_raw_prices(fallback), "fallback")

    raise ContractError(
        f"yfinance download failed after {MAX_RETRIES} attempts and no fallback "
        f"snapshot is available: {last_exc}"
    )
