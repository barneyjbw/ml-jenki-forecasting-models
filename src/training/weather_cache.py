"""
GCS-backed weather cache for Open-Meteo archive API.

Weather for historical dates is deterministic, so we cache per-location parquet
files and only fetch the delta (last few days) on each retrain. This eliminates
the thousand-day requests that trigger 429 rate limits.

On API failure, falls back to stale cache if it covers the requested range.
"""
from __future__ import annotations

import io
import logging
from typing import Iterable

import pandas as pd
import requests
from google.api_core.exceptions import NotFound
from google.cloud.exceptions import NotFound as StorageNotFound
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.gcs import download_bytes, upload_bytes

logger = logging.getLogger(__name__)

WEATHER_CACHE_URI = "gs://jenki-forecast/weather-cache/revenue/{location}.parquet"


def _fetch_range(start: str, end: str, lat: float, lng: float, variables: Iterable[str]) -> pd.DataFrame:
    """Call Open-Meteo archive for a date range with aggressive retry."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(variables),
        "timezone": "Europe/London",
    }

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=15, max=180),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get():
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    data = _get()["daily"]
    df = pd.DataFrame(data)
    missing = [c for c in variables if c not in df.columns]
    if missing:
        raise ValueError(f"Open-Meteo response missing columns: {missing}")
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time": "ds"})


def _read_cache(location: str, variables: list[str]) -> pd.DataFrame | None:
    uri = WEATHER_CACHE_URI.format(location=location)
    try:
        data = download_bytes(uri)
    except (NotFound, StorageNotFound, FileNotFoundError):
        return None
    except Exception as e:
        # Surface a 404 buried in a generic exception.
        if "404" in str(e) or "No such object" in str(e):
            return None
        raise

    df = pd.read_parquet(io.BytesIO(data))
    df["ds"] = pd.to_datetime(df["ds"])
    # Schema drift: if cache missing any requested variable, force full refetch.
    if any(v not in df.columns for v in variables):
        logger.info(f"{location}: weather cache schema stale — will refetch")
        return None
    return df.sort_values("ds").reset_index(drop=True)


def _write_cache(location: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    upload_bytes(buf.getvalue(), WEATHER_CACHE_URI.format(location=location), content_type="application/octet-stream")


def fetch_weather(
    start_date: str,
    end_date: str,
    lat: float,
    lng: float,
    variables: list[str],
    location: str,
) -> pd.DataFrame:
    """Fetch weather via GCS cache, calling Open-Meteo only for missing dates.

    Semantics:
      1. Load cache (if any).
      2. Determine missing dates in [start_date, end_date].
      3. If missing, fetch them and merge + persist.
      4. On API failure, return whatever the cache covers for the window.
         Raise only if both cache + API are unusable.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    requested = pd.date_range(start, end, freq="D")

    cached = _read_cache(location, variables)

    if cached is not None and not cached.empty:
        have = set(cached["ds"].dt.normalize())
        missing_dates = sorted(d for d in requested if d.normalize() not in have)
    else:
        missing_dates = list(requested)

    if missing_dates:
        api_start = missing_dates[0].strftime("%Y-%m-%d")
        api_end = missing_dates[-1].strftime("%Y-%m-%d")
        days = len(missing_dates)
        logger.info(f"{location}: fetching {days} missing weather day(s) {api_start}..{api_end}")
        try:
            fresh = _fetch_range(api_start, api_end, lat, lng, variables)
            combined = (
                pd.concat([cached, fresh], ignore_index=True)
                if cached is not None
                else fresh
            )
            combined = combined.drop_duplicates(subset="ds", keep="last").sort_values("ds").reset_index(drop=True)
            try:
                _write_cache(location, combined)
            except Exception as e:
                logger.warning(f"{location}: failed to persist weather cache: {e}")
            cached = combined
        except Exception as e:
            if cached is None or cached.empty:
                logger.error(f"{location}: weather API failed and no cache — propagating")
                raise
            # Stale-cache fallback: check coverage of requested window.
            have = set(cached["ds"].dt.normalize())
            still_missing = [d for d in requested if d.normalize() not in have]
            if still_missing:
                logger.error(
                    f"{location}: weather API failed ({e}); cache missing "
                    f"{len(still_missing)} day(s) in requested window — propagating"
                )
                raise
            logger.warning(
                f"{location}: weather API failed ({e}) — serving stale cache "
                f"(covers full requested window)"
            )

    mask = (cached["ds"] >= start) & (cached["ds"] <= end)
    out = cached.loc[mask, ["ds"] + list(variables)].copy().reset_index(drop=True)
    return out
