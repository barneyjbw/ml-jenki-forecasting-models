"""
Fetch and cache venue-level events from PredictHQ for Jenki locations.

Populates data/venue_events/{location}.json with dates where a major nearby
venue has an event (concert, performing-arts, sports, etc. at rank ≥ 60).
Re-running updates the cache without replacing existing entries.

Relevant locations:
  - borough:       Shakespeare's Globe (~400m), Flat Iron Square (~300m)
  - covent_garden: Royal Opera House (~300m), London Coliseum / ENO (~400m),
                   West End theatres (Lyceum, Drury Lane, Savoy…)
  - canary_wharf:  No mapped venues within 0.5km — skip unless needed
  - battersea:     Battersea Power Station events (the venue itself)
  - spitalfields:  Rich Mix (~300m)

Usage:
    python -m scripts.fetch_venue_events
    python -m scripts.fetch_venue_events --location borough
    python -m scripts.fetch_venue_events --start 2024-08-01 --end 2027-12-31
    python -m scripts.fetch_venue_events --force  # re-fetch and replace cache

Requires PREDICTHQ_TOKEN env var (or hard-coded token passed via --token).
"""

import argparse
import os
import sys

from src.training.data_prep import LOCATIONS
from src.training.venue_events import VENUE_SEARCH_RADIUS_KM, cache_venue_events

# Training window start; add 1 year of future dates for forward forecasting.
DEFAULT_START = "2024-08-01"
DEFAULT_END   = "2027-12-31"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch venue events from PredictHQ")
    parser.add_argument("--location", choices=list(LOCATIONS.keys()), default=None,
                        help="Single location (default: all)")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--token", default=None, help="PredictHQ token (overrides env var)")
    parser.add_argument("--force", action="store_true", help="Re-fetch and replace cache")
    args = parser.parse_args()

    token = args.token or os.environ.get("PREDICTHQ_TOKEN", "")
    if not token:
        print("Error: PREDICTHQ_TOKEN not set. Set the env var or pass --token.", file=sys.stderr)
        sys.exit(1)

    locs = [args.location] if args.location else list(LOCATIONS.keys())
    for loc in locs:
        radius = VENUE_SEARCH_RADIUS_KM[loc]
        print(f"\n{'='*50}")
        print(f"  {loc}  (radius={radius}km, {args.start} to {args.end})")
        print(f"{'='*50}")
        try:
            cache_venue_events(loc, args.start, args.end, token=token, force_refresh=args.force)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
