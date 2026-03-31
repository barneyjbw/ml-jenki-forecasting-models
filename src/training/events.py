"""
PredictHQ events feature — daily event count per location.

Fetches events within a radius of each Jenki location for a given date range.
Filters to high-impact categories only (concerts, sports, conferences, expos).

At training time : fetch historical events for the full training window.
At forecast time : fetch events for the 14-day forecast window.

Set the PREDICTHQ_TOKEN environment variable or pass token= explicitly.

Usage:
    from src.training.events import get_events_df
    df = get_events_df("borough", "2024-11-01", "2026-03-27")
    # Returns DataFrame with columns: ds, event_count, has_major_event
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

PREDICTHQ_BASE = "https://api.predicthq.com/v1"

# Categories that drive meaningful footfall changes near coffee shops.
# Excluded: community (small local events), daylight-saving, academic (non-footfall)
HIGH_IMPACT_CATEGORIES = [
    "concerts",
    "sports",
    "conferences",
    "expos",
    "festivals",
    "performing-arts",
]

# Radius (km) around each location to search for events.
# Larger for locations near major venues (Canary Wharf → O2 Arena 1.5km away).
SEARCH_RADIUS_KM: dict[str, float] = {
    "battersea":     0.25,
    "borough":       0.25,
    "canary_wharf":  0.25,
    "covent_garden": 0.25,
    "spitalfields":  0.25,
}

# Minimum PHQ rank to count as an event (0-100 scale, higher = bigger event).
# Rank 30+ filters out very small gigs while keeping meaningful events.
MIN_RANK = 30

JENKI_COORDS: dict[str, tuple[float, float]] = {
    "battersea":     (51.4818, -0.1446),
    "borough":       (51.5053, -0.0912),
    "canary_wharf":  (51.5041, -0.0198),
    "covent_garden": (51.5129, -0.1226),
    "spitalfields":  (51.5197, -0.0755),
}


def _get_token() -> str:
    token = os.environ.get("PREDICTHQ_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "PREDICTHQ_TOKEN not set. "
            "Sign up at predicthq.com and set the env var with your access token."
        )
    return token


def _fetch_page(token: str, params: dict) -> dict:
    resp = requests.get(
        f"{PREDICTHQ_BASE}/events/",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_events(
    location: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
) -> pd.DataFrame:
    """
    Fetch all events within the search radius for a location and date range.
    Returns raw DataFrame with one row per event.
    """
    token = token or _get_token()
    lat, lng = JENKI_COORDS[location]
    radius = SEARCH_RADIUS_KM[location]

    all_events = []
    offset = 0
    page_size = 200

    while True:
        params = {
            "category": ",".join(HIGH_IMPACT_CATEGORIES),
            "within": f"{radius}km@{lat},{lng}",
            "active.gte": start_date,
            "active.lte": end_date,
            "rank.gte": MIN_RANK,
            "limit": page_size,
            "offset": offset,
            "sort": "start",
        }
        data = _fetch_page(token, params)
        results = data.get("results", [])
        all_events.extend(results)
        logger.info(
            f"{location}: fetched {len(all_events)} / {data.get('count', '?')} events"
        )

        if len(all_events) >= data.get("count", 0) or not results:
            break
        offset += page_size
        time.sleep(0.2)  # stay within rate limits

    if not all_events:
        return pd.DataFrame(columns=["ds", "event_count", "has_major_event"])

    rows = []
    for e in all_events:
        start = e.get("start", "")[:10]
        end   = e.get("end",   start)[:10]
        rank  = e.get("rank", 0)
        # Expand multi-day events across all their dates
        d = pd.Timestamp(start)
        while str(d.date()) <= end:
            rows.append({"ds": d, "rank": rank, "category": e.get("category", "")})
            d += timedelta(days=1)

    return pd.DataFrame(rows)


def get_events_df(
    location: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
) -> pd.DataFrame:
    """
    Return a daily events summary for a location and date range.

    Columns:
        ds               : date
        event_count      : number of qualifying events that day
        has_major_event  : 1 if any event has rank >= 70 (large venue / high attendance)
    """
    raw = fetch_events(location, start_date, end_date, token)

    # Build a full date spine and join
    dates = pd.date_range(start_date, end_date, freq="D")
    spine = pd.DataFrame({"ds": dates})

    if raw.empty:
        spine["event_count"]    = 0.0
        spine["has_major_event"] = 0.0
        return spine

    daily = (
        raw.groupby("ds")
        .agg(
            event_count=("rank", "count"),
            has_major_event=("rank", lambda x: float((x >= 70).any())),
        )
        .reset_index()
    )
    result = spine.merge(daily, on="ds", how="left").fillna(0)
    return result
