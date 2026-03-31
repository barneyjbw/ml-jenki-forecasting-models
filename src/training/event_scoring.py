"""
Gemini-scored event impact feature.

For each training/forecast date, fetches nearby events from PredictHQ and asks
Gemini Flash to score the expected footfall impact on a scale from -1.0 to 1.0.

  score > 0 : nearby events bring passing crowds past the Jenki store
  score < 0 : nearby events suppress foot traffic or block access
  score = 0 : no qualifying events within the search radius

Results are cached to data/event_scores/{location}.json so API calls are only made
for new dates. Subsequent runs load from cache instantly.

Required env vars (or passed explicitly):
    PREDICTHQ_TOKEN : PredictHQ API access token
    GEMINI_API_KEY  : Google AI Studio API key (get from aistudio.google.com)
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from google import genai
from google.genai import types

LONDON_TZ = ZoneInfo("Europe/London")

# Jenki closing hours (local time) by weekday (0=Mon … 6=Sun).
# Events starting at or after closing are irrelevant — no daytime footfall.
# Events must start within [open, close) local time to be relevant.
# Multi-day events (conferences, expos) are exempt — they run daytime regardless.
JENKI_OPEN_HOUR  = {0: 9, 1: 9, 2: 9, 3: 9, 4: 9, 5: 9, 6: 10}   # Mon-Sat 9:30, Sun 10
JENKI_CLOSE_HOUR = {0: 20, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 19}  # Mon-Sat 8pm, Sun 7pm

from src.training.events import (
    HIGH_IMPACT_CATEGORIES,
    JENKI_COORDS,
    MIN_RANK,
    PREDICTHQ_BASE,
    SEARCH_RADIUS_KM,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/event_scores")

GEMINI_MODEL = "gemini-2.5-flash"

# Jenki customer profile: young professionals, health-conscious, mid-to-high income.
# Stores are small grab-and-go / dine-in matcha bars in high-footfall urban locations.
LOCATION_CONTEXT: dict[str, str] = {
    "battersea": (
        "Jenki Battersea Power Station: inside the Battersea Power Station retail development "
        "in south London. Customers are primarily office workers, shoppers, and residents of "
        "the surrounding Nine Elms development. Nearby tube: Battersea Power Station (Northern line)."
    ),
    "borough": (
        "Jenki Borough Market: adjacent to Borough Market near London Bridge, one of London's "
        "busiest food destinations. High tourist and local foot traffic, especially weekends. "
        "Nearby stations: London Bridge (Jubilee/Northern/National Rail)."
    ),
    "canary_wharf": (
        "Jenki Canary Wharf: in the Canary Wharf financial district, east London. "
        "Core customers are financial sector office workers on weekdays. "
        "The O2 Arena is ~1.5km away — major events there significantly affect area foot traffic. "
        "Nearby stations: Canary Wharf (Jubilee/Elizabeth line/DLR)."
    ),
    "covent_garden": (
        "Jenki Covent Garden: in the heart of Covent Garden, one of London's busiest tourist "
        "and entertainment districts. Mix of tourists, theatre-goers, and office workers. "
        "High baseline foot traffic. Nearby station: Covent Garden (Piccadilly line)."
    ),
    "spitalfields": (
        "Jenki Spitalfields: near Spitalfields Market in east London. "
        "Mix of market visitors, nearby office workers (Liverpool Street area), and local residents. "
        "Nearby station: Liverpool Street (Central/Circle/Hammersmith/Metropolitan/Elizabeth line)."
    ),
}


def _fetch_events_detailed(
    location: str,
    start_date: str,
    end_date: str,
    token: str,
) -> pd.DataFrame:
    """
    Fetch events with title, rank, and category for scoring.
    Returns one row per (event, date) pair — multi-day events are expanded.
    """
    lat, lng = JENKI_COORDS[location]
    radius = SEARCH_RADIUS_KM[location]
    all_events = []
    offset = 0

    while True:
        params = {
            "category": ",".join(HIGH_IMPACT_CATEGORIES),
            "within": f"{radius}km@{lat},{lng}",
            "active.gte": start_date,
            "active.lte": end_date,
            "rank.gte": MIN_RANK,
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
        logger.info(f"{location}: fetched {len(all_events)}/{data.get('count', '?')} events")
        if len(all_events) >= data.get("count", 0) or not results:
            break
        offset += 200
        time.sleep(0.2)

    if not all_events:
        return pd.DataFrame(columns=["ds", "title", "rank", "category"])

    rows = []
    for e in all_events:
        start_str = e.get("start", "")[:10]
        end_str   = e.get("end", start_str)[:10]
        # Genuine multi-day: conference/expo spanning 2+ calendar days.
        # Overnight events (midnight→6am) also have end > start but are NOT daytime events.
        is_multi_day = (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days >= 2

        # Parse start datetime and convert to London local time for hour filter.
        # Events starting at or after Jenki closing time are irrelevant (nightlife).
        # Multi-day events (conferences, expos) are always kept regardless of start time.
        try:
            start_utc = pd.Timestamp(e.get("start", start_str), tz="UTC")
            start_local = start_utc.astimezone(LONDON_TZ)
            start_hour = start_local.hour
            weekday = start_local.weekday()
            open_hour  = JENKI_OPEN_HOUR[weekday]
            close_hour = JENKI_CLOSE_HOUR[weekday]
            within_hours = open_hour <= start_hour < close_hour
        except Exception:
            within_hours = True  # keep if we can't parse

        if not (is_multi_day or within_hours):
            continue  # skip purely nocturnal single-day events

        d = pd.Timestamp(start_str)
        while str(d.date()) <= end_str:
            rows.append({
                "ds":       d,
                "title":    e.get("title", "Unknown event"),
                "rank":     e.get("rank", 0),
                "category": e.get("category", ""),
            })
            d += timedelta(days=1)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Deduplicate: same show on same day can appear twice (matinee + evening)
        df = df.drop_duplicates(subset=["ds", "title"]).reset_index(drop=True)
    return df


def _build_prompt(location: str, date: str, events: list[dict]) -> str:
    loc_ctx = LOCATION_CONTEXT.get(location, location)
    event_lines = "\n".join(
        f"- {e['title']} ({e['category']}, PHQ rank {e['rank']}/100)"
        for e in sorted(events, key=lambda x: -x["rank"])
    )
    return f"""You are a retail foot traffic analyst. Your job is to estimate the impact of nearby events on customer visits to a Jenki matcha bar on a specific day.

STORE CONTEXT:
{loc_ctx}

Jenki sells premium matcha drinks (£5-8). Customers are health-conscious young professionals and tourists. The store is small — capacity ~20 people — so it benefits from passing foot traffic, not destination visits.

DATE: {date}

NEARBY EVENTS (within search radius, ranked by size):
{event_lines}

SCORING INSTRUCTIONS:
Rate the net impact of these events on Jenki foot traffic from -1.0 to +1.0.

Positive impact (events that bring crowds past the store or into the area):
+0.8 to +1.0 = Massive concerts, major festivals, huge sporting finals — tens of thousands of people flooding the area
+0.4 to +0.7 = Large sports matches, popular markets, outdoor festivals — thousands of extra people nearby
+0.1 to +0.3 = Medium conferences, minor events — some extra footfall but limited spillover

Zero or near-zero:
0.0 = No meaningful impact on passing foot traffic

Negative impact (events that suppress access or displace customers):
-0.1 to -0.3 = Minor road closures, small protests — slight disruption
-0.4 to -0.6 = Large marches blocking routes, significant station disruption
-0.7 to -1.0 = Area lockdowns, major infrastructure failure — severe access suppression

KEY CONSIDERATIONS:
- Multi-day corporate conferences bring office workers but few impulse buyers; score low positive (0.1-0.2)
- Concerts and festivals at nearby venues create strong passing-crowd boosts pre/post event
- Events that close roads near the store hurt even if they bring people to the wider area
- The O2 Arena events are very significant for Canary Wharf (1.5km away)
- Borough Market is already high-traffic; events need to be large to move the needle

Output a single number between -1.0 and 1.0, one decimal place. No explanation."""


def _extract_text(resp) -> str | None:
    """Extract text from Gemini response, handling thinking-model multi-part responses."""
    # Standard response
    if resp.text is not None:
        return resp.text
    # Thinking models (e.g. 3.1-pro-preview) return parts: [thinking, text]
    try:
        for part in resp.candidates[0].content.parts:
            if hasattr(part, "text") and part.text and not getattr(part, "thought", False):
                return part.text
    except (IndexError, AttributeError):
        pass
    return None


def _score_day(client: genai.Client, location: str, date: str, events: list[dict]) -> float:
    """Ask Gemini to score footfall impact. Returns float in [-1.0, 1.0]."""
    prompt = _build_prompt(location, date, events)
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=64, temperature=0.1),
            )
            text = _extract_text(resp)
            if text is None:
                logger.warning(f"No text in Gemini response for {location} {date}, using 0.0")
                return 0.0
            # Extract first number found in response (handles any extra explanation)
            import re
            match = re.search(r"-?\d+\.?\d*", text)
            if match:
                return max(-1.0, min(1.0, float(match.group())))
            logger.warning(f"No number in Gemini response '{text}' for {location} {date}, using 0.0")
            return 0.0
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 10 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"Gemini error for {location} {date}: {e}, using 0.0")
                return 0.0
    return 0.0


def get_event_scores_df(
    location: str,
    start_date: str,
    end_date: str,
    predicthq_token: str | None = None,
    gemini_key: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a daily DataFrame with columns [ds, event_impact_score] for the date range.

    Days with no qualifying nearby events get score 0.0 without calling the LLM.
    Results are cached to data/event_scores/{location}.json — only new dates trigger API calls.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{location}.json"

    cache: dict[str, float] = {}
    if cache_path.exists() and not force_refresh:
        with open(cache_path) as f:
            cache = json.load(f)

    all_dates = pd.date_range(start_date, end_date, freq="D")
    uncached = [d for d in all_dates if str(d.date()) not in cache]

    if uncached:
        phq_token = predicthq_token or os.environ.get("PREDICTHQ_TOKEN", "")
        if not phq_token:
            raise EnvironmentError("PREDICTHQ_TOKEN not set")

        api_key = gemini_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")

        score_start = str(uncached[0].date())
        score_end   = str(uncached[-1].date())
        logger.info(f"{location}: scoring {len(uncached)} new dates ({score_start} to {score_end})")

        raw = _fetch_events_detailed(location, score_start, score_end, phq_token)

        events_by_date: dict[str, list[dict]] = {}
        if not raw.empty:
            for _, row in raw.iterrows():
                key = str(row["ds"].date())
                events_by_date.setdefault(key, []).append({
                    "title":    row["title"],
                    "category": row["category"],
                    "rank":     int(row["rank"]),
                })

        client = genai.Client(api_key=api_key)
        llm_calls = 0

        for d in uncached:
            key = str(d.date())
            day_events = events_by_date.get(key, [])
            if day_events:
                cache[key] = _score_day(client, location, key, day_events)
                llm_calls += 1
                time.sleep(0.05)
            else:
                cache[key] = 0.0
            # Save after every scored day so progress isn't lost on failure
            if day_events:
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2, sort_keys=True)

        logger.info(
            f"{location}: Gemini scored {llm_calls} event-days, "
            f"{len(uncached) - llm_calls} zero-event days skipped"
        )

        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)

    rows = [
        {"ds": pd.Timestamp(d), "event_impact_score": cache.get(str(d.date()), 0.0)}
        for d in all_dates
    ]
    return pd.DataFrame(rows)
