"""
Stale forecast checker. Runs as a Cloud Run Job 2 hours after the main retrain job.

For each location, checks whether today's forecast CSV exists in GCS.
If any are missing, fires a Slack alert. Exits non-zero if any missing
so Cloud Run logs the execution as failed.

Usage:
    python -m scripts.check_forecasts
"""

import sys
from datetime import date

from src.training.data_prep import LOCATIONS
from src.utils.gcs import download_bytes
from src.utils.alerts import alert_stale_forecast
from src.utils.logging import get_logger
from scripts.forecast import LOCATION_SLUGS

logger = get_logger(__name__)

# Authoritative output path (matches upload_forecast primary).
FORECAST_ROOT = "gs://bombe-sales-predictions/jenki/location-revenue-by-day"


def _forecast_exists(location: str, today: str) -> bool:
    slug = LOCATION_SLUGS.get(location, location)
    run_date = date.today().strftime("%Y%m%d")
    uri = f"{FORECAST_ROOT}/{slug}/{run_date}.csv"
    try:
        download_bytes(uri)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    today = date.today().strftime("%Y-%m-%d")
    missing = []

    for location in LOCATIONS:
        if _forecast_exists(location, today):
            logger.info(f"{location}: forecast OK ({today})")
        else:
            logger.error(f"{location}: forecast MISSING for {today}")
            missing.append(location)
            alert_stale_forecast(location, today)

    if missing:
        logger.error(f"Missing forecasts: {missing}")
        sys.exit(1)

    logger.info("All forecasts present.")
