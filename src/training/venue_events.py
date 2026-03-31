"""
Phase 2 venue-level event calendar.

Fetches events from PredictHQ within an expanded radius around each Jenki location,
groups them by venue name, and returns Prophet-compatible holiday entries.

Key insight: grouping by venue (e.g., "Shakespeare's Globe Event") lets Prophet learn a
stable coefficient from many occurrences, unlike individual named-event holidays which
have only 1-2 occurrences per training window. Shakespeare's Globe runs ~130 shows/season.

Note: radius is kept to 0.5km for all locations — venue attendees only create spillover
footfall if they pass the store on foot. Parades and marathons (different mechanics)
are already handled in london_calendar.py Phase 1 with a larger effective area.

Canonical examples:
  - Borough: Shakespeare's Globe (~130 shows Apr–Oct, ~400m away)
  - Covent Garden: Royal Opera House, London Coliseum, West End theatres (~300–500m)

Results are cached to data/venue_events/{location}.json. Run scripts/fetch_venue_events.py
to populate. If no cache exists, gracefully returns an empty DataFrame.

Required env var (or passed explicitly): PREDICTHQ_TOKEN
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from src.training.events import HIGH_IMPACT_CATEGORIES, PREDICTHQ_BASE
from src.training.data_prep import JENKI_COORDS
from src.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/venue_events")

# 0.5km for all locations — venue attendees only drive nearby footfall if they're
# walking past the store. 2km would capture venues whose crowds travel directly
# via tube without passing the Jenki (e.g. O2 Arena → North Greenwich, not CW).
# Parades/marathons (large radius OK) are handled in london_calendar.py Phase 1.
VENUE_SEARCH_RADIUS_KM: dict[str, float] = {
    "battersea":     0.5,
    "borough":       0.5,
    "canary_wharf":  0.5,
    "covent_garden": 0.5,
    "spitalfields":  0.5,
}

# Higher rank threshold than event_scoring — only large/notable venues.
# Rank 60+ = arena concerts, major performing arts, large sporting events.
VENUE_MIN_RANK = 60

# Venue name keyword → Prophet holiday label.
# Checked in order; first match wins. Grouping multiple shows under one label
# is intentional so Prophet sees many occurrences and learns a stable coefficient.
VENUE_LABEL_MAP: list[tuple[str, str]] = [
    # Canary Wharf area
    ("The O2",                  "O2 Arena Event"),
    ("O2 Arena",                "O2 Arena Event"),
    # Borough area
    ("Shakespeare's Globe",     "Shakespeare's Globe Event"),
    ("Globe Theatre",           "Shakespeare's Globe Event"),
    ("Flat Iron Square",        "Flat Iron Square Event"),
    ("Tobacco Dock",            "Tobacco Dock Event"),
    ("Ministry of Sound",       "Ministry of Sound Event"),
    # Covent Garden area
    ("Royal Opera House",       "Royal Opera House Event"),
    ("London Coliseum",         "London Coliseum Event"),
    ("Lyceum Theatre",          "West End Theatre Event"),
    ("Theatre Royal Drury Lane","West End Theatre Event"),
    ("Aldwych Theatre",         "West End Theatre Event"),
    ("Savoy Theatre",           "West End Theatre Event"),
    # Battersea area
    ("Battersea Power Station", "Battersea Power Station Event"),
    ("OVO Arena",               "OVO Arena Event"),
    # Spitalfields area
    ("Rich Mix",                "Rich Mix Event"),
    ("Troxy",                   "Troxy Event"),
    # General London arenas (if within radius)
    ("Alexandra Palace",        "Alexandra Palace Event"),
    ("Roundhouse",              "Roundhouse Event"),
    ("Barbican",                "Barbican Event"),
    ("Southbank Centre",        "Southbank Centre Event"),
    ("Royal Festival Hall",     "Southbank Centre Event"),
    ("Queen Elizabeth Hall",    "Southbank Centre Event"),
]


def _venue_label(venue_name: str) -> str | None:
    """Map a venue name to a holiday label, or None if unmapped."""
    for keyword, label in VENUE_LABEL_MAP:
        if keyword.lower() in venue_name.lower():
            return label
    return None


def fetch_venue_events(
    location: str,
    start_date: str,
    end_date: str,
    token: str,
) -> list[dict]:
    """
    Fetch high-rank events within the venue search radius, grouped by venue name.

    Returns list of {"date": "YYYY-MM-DD", "holiday": "<venue label>"} dicts.
    Only events at mapped venues (VENUE_LABEL_MAP) are included.
    Multi-day events are expanded across all their dates.
    """
    lat, lng = JENKI_COORDS[location]
    radius = VENUE_SEARCH_RADIUS_KM[location]
    all_events: list[dict] = []
    offset = 0

    while True:
        params = {
            "category": ",".join(HIGH_IMPACT_CATEGORIES),
            "within": f"{radius}km@{lat},{lng}",
            "active.gte": start_date,
            "active.lte": end_date,
            "rank.gte": VENUE_MIN_RANK,
            "limit": 200,
            "offset": offset,
            "sort": "start",
        }
        resp = requests.get(
            f"{PREDICTHQ_BASE}/events/",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        all_events.extend(results)
        logger.info(
            f"{location}: fetched {len(all_events)}/{data.get('count', '?')} venue events "
            f"(radius={radius}km, rank≥{VENUE_MIN_RANK})"
        )
        if len(all_events) >= data.get("count", 0) or not results:
            break
        offset += 200
        time.sleep(0.2)

    rows: list[dict] = []
    for e in all_events:
        start_str = e.get("start", "")[:10]
        end_str   = e.get("end", start_str)[:10]

        # Extract venue name from entities
        venue_name = ""
        for entity in e.get("entities", []):
            if entity.get("type") == "venue":
                venue_name = entity.get("name", "")
                break

        label = _venue_label(venue_name)
        if label is None:
            continue  # Skip events at unmapped venues

        # Expand multi-day events (cap at 14 days to avoid infinite loops on bad data)
        d = pd.Timestamp(start_str)
        end_ts = pd.Timestamp(end_str)
        days = 0
        while str(d.date()) <= end_str and days < 14:
            rows.append({"date": str(d.date()), "holiday": label})
            d += pd.Timedelta(days=1)
            days += 1

    # Deduplicate: same venue label on same day (matinee + evening at same venue)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for r in rows:
        key = (r["date"], r["holiday"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Summary by venue label
    if unique:
        label_counts: dict[str, int] = {}
        for r in unique:
            label_counts[r["holiday"]] = label_counts.get(r["holiday"], 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(label_counts.items()))
        logger.info(f"{location}: {len(unique)} venue event-days → {summary}")
    else:
        logger.info(f"{location}: 0 venue event-days found at mapped venues")

    return unique


def cache_venue_events(
    location: str,
    start_date: str,
    end_date: str,
    token: str | None = None,
    force_refresh: bool = False,
) -> None:
    """
    Fetch and cache venue events for a location.
    Merges with existing cache (only new dates are re-fetched if force_refresh=False).
    """
    token = token or os.environ.get("PREDICTHQ_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "PREDICTHQ_TOKEN not set. "
            "Set the env var or pass token= explicitly."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{location}.json"

    # Load existing cache
    existing: list[dict] = []
    if cache_path.exists() and not force_refresh:
        with open(cache_path) as f:
            existing = json.load(f)

    # Fetch new events
    new_events = fetch_venue_events(location, start_date, end_date, token)

    # Merge: existing + new, deduplicated
    all_events = {(e["date"], e["holiday"]): e for e in existing}
    for e in new_events:
        all_events[(e["date"], e["holiday"])] = e

    merged = sorted(all_events.values(), key=lambda x: x["date"])
    with open(cache_path, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info(f"{location}: saved {len(merged)} total venue event-days to {cache_path}")


def get_venue_events_df(location: str, years) -> pd.DataFrame:
    """
    Load cached venue events for a location, filtered to the requested years.

    Returns Prophet-compatible DataFrame with columns: ds (datetime), holiday (str).
    Returns empty DataFrame if no cache exists — graceful degradation without crashing.
    """
    cache_path = CACHE_DIR / f"{location}.json"
    if not cache_path.exists():
        return pd.DataFrame(columns=["ds", "holiday"])

    with open(cache_path) as f:
        events = json.load(f)

    if not events:
        return pd.DataFrame(columns=["ds", "holiday"])

    df = pd.DataFrame(events)
    df["ds"] = pd.to_datetime(df["date"])
    year_set = set(years)
    df = df[df["ds"].dt.year.isin(year_set)][["ds", "holiday"]].reset_index(drop=True)
    return df
