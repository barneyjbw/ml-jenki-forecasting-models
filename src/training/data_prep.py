"""
Load and clean Revel daily CSVs, join weather and footfall features.

Final feature set per location:
  ds, y,
  apparent_temperature_max, precipitation_sum, sunshine_duration, wind_speed_10m_max,
  footfall_actual, footfall_yoy
"""

import io
import os
import re
import glob
from pathlib import Path
from functools import lru_cache

import requests
import pandas as pd

from src.utils.logging import get_logger

# Set to True to fetch + score events via PredictHQ + Claude.
# Requires PREDICTHQ_TOKEN and ANTHROPIC_API_KEY env vars.
# Results are cached to data/event_scores/ — only new dates trigger API calls.
USE_EVENT_SCORING = False

logger = get_logger(__name__)

# DATA_SOURCE=gcs → read Revel CSVs + footfall from GCS (production)
# DATA_SOURCE=local (default) → read from local data/source-data/ (development)
DATA_SOURCE = os.getenv("DATA_SOURCE", "local")

DATA_ROOT = Path("data/source-data")
GCS_SOURCE_ROOT = "gs://bombe-ml-data/revenue-predictor/source-data"

LOCATIONS: dict[str, Path] = {
    "battersea":     DATA_ROOT / "Revel Data - Battersea",
    "borough":       DATA_ROOT / "Revel Data - Borough",
    "canary_wharf":  DATA_ROOT / "Revel Data - Canary Wharf",
    "covent_garden": DATA_ROOT / "Revel Data - Covent Garden",
    "spitalfields":  DATA_ROOT / "Revel Data - Spitalfields",
}

GCS_LOCATIONS: dict[str, str] = {
    "battersea":     f"{GCS_SOURCE_ROOT}/Revel Data - Battersea",
    "borough":       f"{GCS_SOURCE_ROOT}/Revel Data - Borough",
    "canary_wharf":  f"{GCS_SOURCE_ROOT}/Revel Data - Canary Wharf",
    "covent_garden": f"{GCS_SOURCE_ROOT}/Revel Data - Covent Garden",
    "spitalfields":  f"{GCS_SOURCE_ROOT}/Revel Data - Spitalfields",
}

# Truncate training data from this date onwards (inclusive).
# Spitalfields closed Oct-Dec 2025 and reopened at a permanently lower revenue level.
TRAINING_START: dict[str, str | None] = {
    "battersea":     None,
    "borough":       None,
    "canary_wharf":  None,
    "covent_garden": None,
    "spitalfields":  "2026-01-01",
}

# Specific dates to exclude from training (anomalous days).
# Specific dates to exclude from training (anomalous days).
# NOTE: Anomalous days in the opening period (CG Aug 28/Sep 5 2024 at £10, Borough Aug 1 2025 at £73)
# were tested but kept — removing them destabilises Prophet's trend/seasonality estimation
# and causes val/test regression. Prophet averages them out rather than memorising them.
EXCLUDE_DATES: dict[str, list[str]] = {
    "battersea": ["2026-01-18"],  # first day of trading, £10.95 revenue
}

JENKI_COORDS: dict[str, tuple[float, float]] = {
    "battersea":     (51.4818, -0.1446),
    "borough":       (51.5053, -0.0912),
    "canary_wharf":  (51.5041, -0.0198),
    "covent_garden": (51.5129, -0.1226),
    "spitalfields":  (51.5197, -0.0755),
}

# Primary TfL station per location (highest exit volume within walking distance)
PRIMARY_STATION: dict[str, str] = {
    "battersea":     "Battersea Power Station",
    "borough":       "London Bridge",
    "canary_wharf":  "Canary Wharf",
    "covent_garden": "Covent Garden",
    "spitalfields":  "Liverpool Street",
}

WEATHER_VARIABLES = [
    "apparent_temperature_max",
    "precipitation_sum",
    "precipitation_hours",
    "sunshine_duration",
    "wind_speed_10m_max",
    "daylight_duration",
]

FOOTFALL_FEATURES = [
    "footfall_actual",
    "footfall_yoy",
]

# Derived features computed from raw weather variables.
# temp_sq = (apparent_temperature_max - 15)^2
# Reference 15°C is a fixed constant (ideal mild day), so no data leakage at forecast time.
DERIVED_FEATURES = [
    "temp_sq",
]

# Cross-location signal. Computed from other locations' revenue history — no leakage.
# At forecast time, use most recent 7d/28d ratio as a scalar held constant for the horizon.
NETWORK_FEATURES = [
    "network_momentum",
]

# Stacked ensemble peer forecast cache directory.
# Populated by scripts/generate_peer_forecasts.py — must be run after all base models trained.
PEER_FORECAST_DIR = Path("data/peer_forecasts")

# Claude-scored event impact. Requires PREDICTHQ_TOKEN + ANTHROPIC_API_KEY.
# Cached to data/event_scores/ so API calls only happen for new dates.
EVENT_FEATURES = [
    "event_impact_score",
]

ALL_REGRESSORS = WEATHER_VARIABLES + FOOTFALL_FEATURES + DERIVED_FEATURES

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_date(filename: str) -> str | None:
    match = _DATE_RE.search(filename)
    return match.group(1) if match else None


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV from a local path or a gs:// URI."""
    path = str(path)
    if path.startswith("gs://"):
        from src.utils.gcs import download_bytes
        return pd.read_csv(io.BytesIO(download_bytes(path)))
    return pd.read_csv(path)


def _list_csvs(location: str) -> list[str]:
    """Return CSV paths/URIs for a location, using GCS or local depending on DATA_SOURCE."""
    if DATA_SOURCE == "gcs":
        from src.utils.gcs import list_blobs
        prefix = GCS_LOCATIONS[location]
        return sorted(uri for uri in list_blobs(prefix + "/") if uri.endswith(".csv"))
    return sorted(glob.glob(str(LOCATIONS[location] / "*.csv")))


@lru_cache(maxsize=1)
def _load_station_footfall() -> pd.DataFrame:
    """
    Load and combine both StationFootfall CSVs.
    Returns DataFrame with columns: date, station, exits
    """
    if DATA_SOURCE == "gcs":
        paths = [
            f"{GCS_SOURCE_ROOT}/StationFootfall_2024_2025 -2.csv",
            f"{GCS_SOURCE_ROOT}/StationFootfall_2025_2026.csv",
        ]
    else:
        paths = [
            str(DATA_ROOT / "StationFootfall_2024_2025 -2.csv"),
            str(DATA_ROOT / "StationFootfall_2025_2026.csv"),
        ]

    dfs = []
    for path in paths:
        df = _read_csv(path)
        df["date"] = pd.to_datetime(df["TravelDate"].astype(str), format="%Y%m%d")
        df = df.rename(columns={"Station": "station", "ExitTapCount": "exits"})
        dfs.append(df[["date", "station", "exits"]])

    combined = pd.concat(dfs)
    combined = combined.drop_duplicates(subset=["date", "station"])
    return combined.sort_values("date").reset_index(drop=True)


def _get_footfall_features(location: str, dates: pd.Series) -> pd.DataFrame:
    """
    Compute footfall features for a given location and set of dates.

    footfall_actual : exits for that date (falls back to same day last year if unavailable)
    footfall_yoy    : exits for same date last year (falls back to dow average)

    Note: TfL StationFootfall data has a ~2 week lag. For forecast dates (and recent
    training dates beyond the data cutoff), footfall_actual equals footfall_yoy.
    """
    station = PRIMARY_STATION[location]
    sf = _load_station_footfall()
    station_df = sf[sf["station"] == station][["date", "exits"]].set_index("date")

    # Day-of-week averages — fallback only when yoy date is missing
    dow_avg = (
        station_df.copy()
        .assign(dow=lambda x: x.index.day_name())
        .groupby("dow")["exits"]
        .mean()
        .to_dict()
    )

    rows = []
    for date in dates:
        date = pd.Timestamp(date)
        yoy_date = date - pd.DateOffset(years=1)

        if yoy_date in station_df.index:
            footfall_yoy = station_df.loc[yoy_date, "exits"]
        else:
            footfall_yoy = dow_avg.get(date.day_name(), float("nan"))

        if date in station_df.index:
            raw = station_df.loc[date, "exits"]
            # Sanity check: if exits are implausibly low (<10% of dow average), treat as data gap
            threshold = dow_avg.get(date.day_name(), 0) * 0.1
            footfall_actual = raw if raw >= threshold else dow_avg.get(date.day_name(), footfall_yoy)
        else:
            footfall_actual = footfall_yoy

        rows.append({"ds": date, "footfall_actual": footfall_actual, "footfall_yoy": footfall_yoy})

    return pd.DataFrame(rows)



def _fetch_weather(start_date: str, end_date: str, lat: float, lng: float) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(WEATHER_VARIABLES),
        "timezone": "Europe/London",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time": "ds"})


def _load_revenue(location: str) -> pd.DataFrame:
    files = _list_csvs(location)
    if not files:
        raise FileNotFoundError(f"No CSV files found for '{location}' in {folder}")

    rows = []
    for f in files:
        date_str = _extract_date(Path(f).name)
        if not date_str:
            continue
        try:
            df = _read_csv(f)
            revenue = df["Total Sales"].sum()
            rows.append({"ds": date_str, "y": revenue})
        except Exception as e:
            logger.info(f"Skipping {f}: {e}")

    daily = pd.DataFrame(rows)
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)
    before = len(daily)
    daily = daily[daily["y"] > 0].reset_index(drop=True)
    if before - len(daily):
        logger.info(f"{location}: dropped {before - len(daily)} zero-revenue day(s)")

    cutoff = TRAINING_START.get(location)
    if cutoff:
        daily = daily[daily["ds"] >= cutoff].reset_index(drop=True)
        logger.info(f"{location}: truncated to {cutoff} onwards ({len(daily)} days)")

    excluded = EXCLUDE_DATES.get(location, [])
    if excluded:
        daily = daily[~daily["ds"].isin(pd.to_datetime(excluded))].reset_index(drop=True)
        logger.info(f"{location}: excluded {len(excluded)} anomalous day(s): {excluded}")

    return daily


def _compute_network_momentum(location: str, dates: pd.Series) -> pd.DataFrame:
    """
    Cross-location revenue trend signal.

    momentum = (7-day rolling mean of other-location revenue)
             / (28-day rolling mean of other-location revenue)

    > 1.0  → network trending above recent baseline
    < 1.0  → network trending below recent baseline
    = 1.0  → neutral fallback (insufficient history)

    Uses historical actuals only, so no data leakage. At forecast time,
    hold the most recent computed value constant across the horizon.
    """
    other_locs = [loc for loc in LOCATIONS if loc != location]

    dfs = []
    for loc in other_locs:
        rev = _load_revenue(loc)[["ds", "y"]].rename(columns={"y": loc})
        rev = rev.drop_duplicates(subset="ds")
        dfs.append(rev.set_index("ds"))

    combined = pd.concat(dfs, axis=1)
    combined["network_revenue"] = combined.sum(axis=1, min_count=2)

    roll7  = combined["network_revenue"].rolling(7,  min_periods=4).mean()
    roll28 = combined["network_revenue"].rolling(28, min_periods=14).mean()
    momentum = (roll7 / roll28).clip(0.5, 2.0).rename("network_momentum")
    momentum_df = momentum.reset_index().rename(columns={"index": "ds"})

    result = pd.DataFrame({"ds": pd.to_datetime(dates)})
    result = result.merge(momentum_df, on="ds", how="left")
    result["network_momentum"] = result["network_momentum"].fillna(1.0)
    return result[["ds", "network_momentum"]]


def load_training_data(location: str) -> pd.DataFrame:
    """
    Return full feature DataFrame for training:
    ds, y + all weather and footfall regressors.
    """
    revenue = _load_revenue(location)
    start = revenue["ds"].min().strftime("%Y-%m-%d")
    end = revenue["ds"].max().strftime("%Y-%m-%d")

    lat, lng = JENKI_COORDS[location]
    logger.info(f"{location}: fetching weather {start} to {end}")
    weather = _fetch_weather(start, end, lat, lng)

    footfall = _get_footfall_features(location, revenue["ds"])
    network  = _compute_network_momentum(location, revenue["ds"])

    df = revenue.merge(weather, on="ds", how="left")
    df = df.merge(footfall, on="ds", how="left")
    df = df.merge(network, on="ds", how="left")

    if USE_EVENT_SCORING:
        from src.training.event_scoring import get_event_scores_df
        start = revenue["ds"].min().strftime("%Y-%m-%d")
        end   = revenue["ds"].max().strftime("%Y-%m-%d")
        events = get_event_scores_df(location, start, end)
        df = df.merge(events, on="ds", how="left")
        df["event_impact_score"] = df["event_impact_score"].fillna(0.0)
    else:
        df["event_impact_score"] = 0.0

    # Stacked ensemble: peer_yhat from pre-generated cache.
    # Values near 1.0 = peers predicting average; >1 = peers predict above-average day.
    # Neutral default (1.0) when cache absent — feature unused unless in MODEL_CONFIG extra_regressors.
    peer_cache = PEER_FORECAST_DIR / f"{location}.parquet"
    if peer_cache.exists():
        peer_df = pd.read_parquet(peer_cache).drop_duplicates(subset="ds")
        df = df.merge(peer_df, on="ds", how="left")
        df["peer_yhat"] = df["peer_yhat"].fillna(1.0)
    else:
        df["peer_yhat"] = 1.0

    # Derived features
    df["temp_sq"] = (df["apparent_temperature_max"] - 15.0) ** 2
    df["rainy_day"] = (df["precipitation_sum"] > 1.0).astype(float)
    df["precip_sq"] = df["precipitation_hours"] ** 2

    # Shakespeare's Globe outdoor season (Borough Market only).
    # Season runs ~April 23 to ~October 26 each year.
    # Thousands of theatre-goers pass through Borough Market before/after shows.
    # Computable years ahead — no leakage at forecast time.
    month = df["ds"].dt.month
    day   = df["ds"].dt.day
    after_open  = (month > 4) | ((month == 4) & (day >= 23))
    before_close = (month < 10) | ((month == 10) & (day <= 26))
    df["globe_season_active"] = (after_open & before_close).astype(float)

    missing = df[ALL_REGRESSORS].isnull().sum()
    if missing.any():
        logger.info(f"{location}: missing values:\n{missing[missing > 0]}")
        df = df.dropna(subset=ALL_REGRESSORS)

    logger.info(
        f"{location}: {len(df)} days, "
        f"{df['ds'].min().date()} to {df['ds'].max().date()}, "
        f"revenue £{df['y'].sum():,.2f}"
    )
    return df.reset_index(drop=True)
