"""
London social calendar — predictable recurring events that shift consumer footfall.

Phase 1: Algorithmic events — all dates computable years ahead (production-safe, no APIs).
Phase 2: Venue-level events — loaded from data/venue_events/{location}.json (cache).
         Run scripts/fetch_venue_events.py to populate. Gracefully absent if no cache.

Phase 1 events:
  - Father's Day           (suppresses, like Mothering Sunday)
  - London Marathon        (major crowd event, Apr)
  - Pride London           (large crowd event, Jun/Jul)
  - Notting Hill Carnival  (Aug bank holiday weekend)
  - Chinese New Year parade (London, Feb — near CG/Soho)
  - St Patrick's Day parade (London, nearest Sun to Mar 17)
  - Valentine's Day        (Feb 14)
  - New Year's Eve         (Dec 31)
  - Bonfire Night          (Nov 5)
  - Diwali                 (variable Oct/Nov)
  - Chelsea Flower Show    (May, 5-day run)
  - Wimbledon finals wknd  (July)

Phase 2 venue events (from cache, location-specific):
  - borough:       Shakespeare's Globe, Flat Iron Square
  - covent_garden: Royal Opera House, London Coliseum, West End theatres
  - battersea:     Battersea Power Station events
  - spitalfields:  Rich Mix
"""

import calendar
from datetime import date, timedelta

from dateutil.easter import easter

import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of weekday (0=Mon … 6=Sun) in the given month."""
    first = date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + timedelta(days=delta + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of weekday in the given month."""
    last = date(year, month, calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _aug_bank_holiday(year: int) -> date:
    """August bank holiday England = last Monday of August."""
    return _last_weekday(year, 8, 0)


# ---------------------------------------------------------------------------
# Per-event date generators
# ---------------------------------------------------------------------------

def _fathers_day_dates(years):
    """3rd Sunday of June — suppresses revenue like Mothering Sunday."""
    for y in years:
        yield date(y, 6, _nth_weekday(y, 6, 6, 3).day), "Father's Day"


def _london_marathon_dates(years):
    """London Marathon — last Sunday of April (route passes Borough, Tower Bridge, CG embankment)."""
    # 2024 was moved to April 21 (3rd Sunday); from 2025 onward last Sunday of April.
    # Hardcode known exceptions; default to last Sunday of April.
    known = {2024: date(2024, 4, 21), 2025: date(2025, 4, 27)}
    for y in years:
        d = known.get(y, _last_weekday(y, 4, 6))
        yield d, "London Marathon"


def _pride_london_dates(years):
    """Pride in London — last Saturday of June. Major crowd event for CG/Soho."""
    known = {2024: date(2024, 6, 29), 2025: date(2025, 6, 28), 2026: date(2026, 6, 27)}
    for y in years:
        d = known.get(y, _last_weekday(y, 6, 5))
        yield d, "Pride London"


def _notting_hill_carnival_dates(years):
    """Notting Hill Carnival — Sunday and Monday of August bank holiday weekend."""
    for y in years:
        monday = _aug_bank_holiday(y)
        sunday = monday - timedelta(days=1)
        yield sunday, "Notting Hill Carnival"
        yield monday, "Notting Hill Carnival"


def _chinese_new_year_dates(years):
    """
    London Chinese New Year parade (Trafalgar Square / Chinatown).
    Highly relevant for Covent Garden (~300m from Chinatown).
    Dates are lunar-calendar-based — hardcoded for accuracy.
    """
    known = {
        2024: date(2024, 2, 11),
        2025: date(2025, 2,  2),
        2026: date(2026, 2, 22),
        2027: date(2027, 2, 14),
        2028: date(2028, 2,  6),
        2029: date(2029, 1, 28),
    }
    for y in years:
        if y in known:
            yield known[y], "Chinese New Year Parade London"


def _st_patricks_day_dates(years):
    """
    London St Patrick's Day parade — Sunday nearest to March 17.
    Brings pub/café crowds to central London.
    """
    for y in years:
        mar17 = date(y, 3, 17)
        # Find nearest Sunday (could be before or after Mar 17)
        days_to_sun = (6 - mar17.weekday()) % 7
        days_from_sun = mar17.weekday() + 1 if mar17.weekday() != 6 else 0
        if days_to_sun <= days_from_sun:
            parade = mar17 + timedelta(days=days_to_sun)
        else:
            parade = mar17 - timedelta(days=days_from_sun)
        yield parade, "St Patrick's Day Parade London"


def _valentines_day_dates(years):
    """Valentine's Day — Feb 14. Suppresses daytime coffee shop trade (couple-focused evening)."""
    for y in years:
        yield date(y, 2, 14), "Valentine's Day"


def _new_years_eve_dates(years):
    """New Year's Eve — Dec 31. Central London very busy from afternoon."""
    for y in years:
        yield date(y, 12, 31), "New Year's Eve"


def _bonfire_night_dates(years):
    """Bonfire Night — Nov 5. Evening-focused; suppresses daytime trade."""
    for y in years:
        yield date(y, 11, 5), "Bonfire Night"


def _diwali_dates(years):
    """
    Diwali — variable Hindu festival, typically October/November.
    London celebrates with events around Trafalgar Sq (near CG).
    Hardcoded — lunar calendar.
    """
    known = {
        2024: date(2024, 11,  1),
        2025: date(2025, 10, 20),
        2026: date(2026, 11,  8),
        2027: date(2027, 10, 29),
        2028: date(2028, 10, 17),
    }
    for y in years:
        if y in known:
            yield known[y], "Diwali"


def _chelsea_flower_show_dates(years):
    """
    Chelsea Flower Show — runs Tue–Sat of the 3rd full week of May.
    Draws affluent visitors to SW London; mild positive for central locations.
    """
    known = {
        2024: date(2024, 5, 21),  # Tuesday start
        2025: date(2025, 5, 20),
        2026: date(2026, 5, 19),
        2027: date(2027, 5, 25),
    }
    for y in years:
        if y in known:
            start = known[y]
            for i in range(5):  # Tue–Sat (5 days)
                yield start + timedelta(days=i), "Chelsea Flower Show"


def _wimbledon_finals_dates(years):
    """
    Wimbledon finals weekend — Ladies' final Saturday, Gentlemen's final Sunday.
    Peak tourist footfall in London. Fortnight starts last Monday of June.
    """
    known_starts = {
        2024: date(2024, 7,  1),
        2025: date(2025, 6, 30),
        2026: date(2026, 6, 29),
        2027: date(2027, 6, 28),
    }
    for y in years:
        if y in known_starts:
            start = known_starts[y]
            finals_saturday = start + timedelta(days=12)  # 13th day
            finals_sunday   = start + timedelta(days=13)  # 14th day
            yield finals_saturday, "Wimbledon Finals Weekend"
            yield finals_sunday,   "Wimbledon Finals Weekend"


def _easter_sunday_dates(years):
    """
    Easter Sunday — family day, suppresses coffee shop revenue like Mothering Sunday.
    Not a UK bank holiday (Easter Monday is), so not in the holidays library.
    Calculable years ahead.
    """
    for y in years:
        yield easter(y), "Easter Sunday"


def _christmas_eve_dates(years):
    """
    Christmas Eve (Dec 24) — stores close early (typically noon–2pm).
    Not a UK bank holiday, so not in the holidays library.
    Clear suppressor: model consistently over-predicts Dec 24 for all locations.
    """
    for y in years:
        yield date(y, 12, 24), "Christmas Eve"


def _black_friday_dates(years):
    """
    Black Friday — Friday after US Thanksgiving (4th Thursday of November).
    Drives increased footfall in retail/market areas.
    """
    for y in years:
        # 4th Thursday of November
        thanksgiving = _nth_weekday(y, 11, 3, 4)
        yield thanksgiving + timedelta(days=1), "Black Friday"


# ---------------------------------------------------------------------------
# Phase 2 additions
# ---------------------------------------------------------------------------

def _london_fashion_week_dates(years):
    """
    London Fashion Week — February and September editions, Mon–Fri (~5 days each).
    Somerset House (200m from Jenki CG) is the main hub. Fashion industry crowds
    mix with tourists in the Covent Garden / Strand area throughout the week.
    3+ occurrences within CG's training window: Sept 2024, Feb 2025, Sept 2025.
    """
    # Approximate Monday start dates (officially Mon–Fri)
    known: dict[int, list[date]] = {
        2024: [date(2024, 9, 13)],           # Sept 2024 (Feb 2024 = before training window)
        2025: [date(2025, 2, 21), date(2025, 9, 12)],
        2026: [date(2026, 2, 20), date(2026, 9, 11)],
        2027: [date(2027, 2, 19), date(2027, 9, 10)],
    }
    for y in years:
        for start in known.get(y, []):
            for i in range(5):  # Mon–Fri
                yield start + timedelta(days=i), "London Fashion Week"


def _trooping_the_colour_dates(years):
    """
    Trooping the Colour — 2nd Saturday of June (King's Official Birthday Parade).
    Procession passes The Mall → Buckingham Palace; crowds fill Trafalgar Sq / CG area.
    1 occurrence within current training window (Jun 14, 2025).
    Note: single occurrence makes the coefficient noisy — worth monitoring.
    """
    for y in years:
        yield _nth_weekday(y, 6, 5, 2), "Trooping the Colour"


# ---------------------------------------------------------------------------
# Location relevance
# ---------------------------------------------------------------------------

# Events relevant to each location. "universal" events (Father's Day etc.) affect
# all locations. Area-specific events only affect nearby locations.
# Prophet will still learn a near-zero coefficient if an event has no effect —
# but fewer irrelevant events = less noise, especially with short training history.
LOCATION_EVENTS: dict[str, set[str]] = {
    # Universal: behavioural suppressors / national uplift days
    "_universal": {
        "Father's Day",
        "Valentine's Day",
        "New Year's Eve",
        "Bonfire Night",
        "London Marathon",   # route passes Tower Bridge, Embankment — all central locations affected
        "Easter Sunday",     # family day, suppresses footfall like Mothering Sunday
        "Christmas Eve",     # early close (~noon), suppresses afternoon/evening trade
        # Black Friday: removed — hurts CG (+0.19pp) with only 2 noisy training occurrences.
        # Revisit at 3+ occurrences (Nov 2026 training data).
    },
    # Central / tourist London — large crowd events nearby
    "covent_garden": {
        "Pride London",
        "Chinese New Year Parade London",
        "St Patrick's Day Parade London",
        "Diwali",
        "Notting Hill Carnival",    # draws crowds from CG direction
        "Wimbledon Finals Weekend", # tourist surge in central London
        "Chelsea Flower Show",
        # Phase 2 additions
        "London Fashion Week",      # Somerset House 200m away; 3+ occurrences in training
        "Trooping the Colour",      # Trafalgar Sq crowds; 1 occurrence (Jun 2025) — monitor coefficient
    },
    "borough":      None,  # Opt out — 1 occurrence per annual event in 17-month window adds noise (EXP-30 confirmed)
    "spitalfields": None,  # Too short (opens Jan 2026) — revisit at 24+ months
    "canary_wharf": None,  # No calendar events in range; venue events via venue_events.py cache
    "battersea":    None,  # Too short (opens Jan 2026) — revisit at 24+ months
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_london_events_df(years, location: str | None = None) -> pd.DataFrame:
    """
    Return a Prophet-compatible holidays DataFrame for the given years.

    Phase 1 (social calendar): Algorithmic, computable years ahead.
      - If location=None: all events.
      - If LOCATION_EVENTS[location] is None: skip social calendar for this location.
      - Otherwise: universal events + location-specific events.

    Phase 2 (venue events): Loaded from data/venue_events/{location}.json cache.
      - Always merged if cache exists, regardless of Phase 1 opt-in status.
      - Run scripts/fetch_venue_events.py to populate the cache.
      - If no cache: silently skipped (graceful degradation).

    Columns: ds (datetime), holiday (str)
    """
    generators = [
        _fathers_day_dates,
        _london_marathon_dates,
        _pride_london_dates,
        _notting_hill_carnival_dates,
        _chinese_new_year_dates,
        _st_patricks_day_dates,
        _valentines_day_dates,
        _new_years_eve_dates,
        _bonfire_night_dates,
        _diwali_dates,
        _chelsea_flower_show_dates,
        _wimbledon_finals_dates,
        _easter_sunday_dates,
        _christmas_eve_dates,
        _black_friday_dates,
        # Phase 2
        _london_fashion_week_dates,
        _trooping_the_colour_dates,
    ]

    # ---- Phase 1: algorithmic social calendar ----
    # None value in LOCATION_EVENTS = opt out of 1-occurrence social calendar events.
    # Strong universal suppressors/uplifts (Easter Sunday, Black Friday) always apply.
    _ALWAYS_ON = {"Easter Sunday", "Christmas Eve"}

    if location is not None:
        loc_events = LOCATION_EVENTS.get(location, set())
        if loc_events is None:
            allowed: set[str] | None = _ALWAYS_ON  # opted-out: only apply strong universals
        else:
            allowed = LOCATION_EVENTS["_universal"] | loc_events
    else:
        allowed = None  # include all events

    rows = []
    for gen in generators:
        for d, name in gen(years):
            if allowed is None or name in allowed:
                rows.append({"ds": str(d), "holiday": name})

    # ---- Phase 2: venue-level events from PredictHQ cache ----
    if location is not None:
        try:
            from src.training.venue_events import get_venue_events_df
            venue_df = get_venue_events_df(location, years)
            for _, row in venue_df.iterrows():
                rows.append({"ds": str(row["ds"].date()), "holiday": row["holiday"]})
        except ImportError:
            pass  # venue_events module not available

    return pd.DataFrame(rows).drop_duplicates(subset=["ds", "holiday"]).reset_index(drop=True)
