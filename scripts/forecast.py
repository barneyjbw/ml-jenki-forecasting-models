"""
Production inference — generate revenue forecasts for all Jenki locations.

Horizon: 14 days for Borough + Covent Garden, 7 days for Battersea / Canary Wharf / Spitalfields.
Output:  gs://jenki-forecast/{location}/forecast_{YYYY-MM-DD}.csv
Columns: date, predicted_revenue, lower_95, upper_95

Usage:
    python -m scripts.forecast                    # all locations
    python -m scripts.forecast --location borough
"""

import argparse
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.training.data_prep import (
    LOCATIONS, JENKI_COORDS, WEATHER_VARIABLES,
    _get_footfall_features, _load_revenue,
)
from src.training.train import _regressors
from src.utils.gcs import upload_bytes
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_DIR  = Path("models")
GCS_BUCKET = "gs://jenki-forecast"

FORECAST_HORIZON: dict[str, int] = {
    "borough":       14,
    "covent_garden": 14,
    "battersea":      7,
    "canary_wharf":   7,
    "spitalfields":   7,
}


# ---------------------------------------------------------------------------
# Shared inference utilities (imported by plots_with_forecast.py)
# ---------------------------------------------------------------------------

def fetch_weather_forecast(start_date: str, end_date: str, lat: float, lng: float) -> pd.DataFrame:
    """Fetch daily weather from Open-Meteo forecast API (accepts recent + future dates)."""
    params = {
        "latitude":   lat,
        "longitude":  lng,
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      ",".join(WEATHER_VARIABLES),
        "timezone":   "Europe/London",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time": "ds"})


def current_network_momentum(location: str) -> float:
    """Last known 7d/28d cross-location revenue ratio — held constant over the horizon."""
    other_locs = [loc for loc in LOCATIONS if loc != location]
    dfs = []
    for loc in other_locs:
        rev = _load_revenue(loc)[["ds", "y"]].rename(columns={"y": loc}).drop_duplicates("ds")
        dfs.append(rev.set_index("ds"))
    combined = pd.concat(dfs, axis=1)
    net    = combined.sum(axis=1, min_count=2)
    roll7  = net.rolling(7,  min_periods=4).mean()
    roll28 = net.rolling(28, min_periods=14).mean()
    valid  = (roll7 / roll28).clip(0.5, 2.0).dropna()
    return float(valid.iloc[-1]) if len(valid) else 1.0


def build_future_df(location: str, start: str, end: str, regs: list) -> pd.DataFrame:
    """Build regressor DataFrame for a forecast date range."""
    lat, lng = JENKI_COORDS[location]
    dates = pd.date_range(start=start, end=end, freq="D")

    weather  = fetch_weather_forecast(start, end, lat, lng)
    footfall = _get_footfall_features(location, dates)

    df = pd.DataFrame({"ds": pd.to_datetime(dates)})
    df = df.merge(weather,  on="ds", how="left")
    df = df.merge(footfall, on="ds", how="left")

    # Derived features — identical to training pipeline
    df["temp_sq"]   = (df["apparent_temperature_max"] - 15.0) ** 2
    df["rainy_day"] = (df["precipitation_sum"] > 1.0).astype(float)
    df["precip_sq"] = df["precipitation_hours"] ** 2

    month, day = df["ds"].dt.month, df["ds"].dt.day
    df["globe_season_active"] = (
        ((month > 4) | ((month == 4) & (day >= 23))) &
        ((month < 10) | ((month == 10) & (day <= 26)))
    ).astype(float)

    if "network_momentum" in regs:
        df["network_momentum"] = current_network_momentum(location)

    if "peer_yhat" in regs:
        df["peer_yhat"] = 1.0

    return df


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_forecast(location: str) -> pd.DataFrame:
    """Run inference. Returns: date, predicted_revenue, lower_95, upper_95."""
    horizon = FORECAST_HORIZON[location]
    start   = date.today().strftime("%Y-%m-%d")
    end     = (date.today() + timedelta(days=horizon - 1)).strftime("%Y-%m-%d")
    logger.info(f"{location}: forecasting {start} → {end}")

    with open(MODEL_DIR / f"{location}.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    log_y = saved.get("log_y", False)
    regs  = saved["regressors"]

    future = build_future_df(location, start, end, regs)
    fc     = model.predict(future[["ds"] + regs])

    def _out(arr: np.ndarray) -> np.ndarray:
        return np.maximum(np.expm1(arr) if log_y else arr, 0.0)

    result = pd.DataFrame({
        "date":              future["ds"].dt.strftime("%Y-%m-%d"),
        "predicted_revenue": np.round(_out(fc["yhat"].values),       2),
        "lower_95":          np.round(_out(fc["yhat_lower"].values), 2),
        "upper_95":          np.round(_out(fc["yhat_upper"].values), 2),
    })
    logger.info(f"{location}: forecast ready\n{result.to_string(index=False)}")
    return result


# ---------------------------------------------------------------------------
# GCS upload
# ---------------------------------------------------------------------------

def upload_forecast(location: str, df: pd.DataFrame) -> str:
    run_date = date.today().strftime("%Y-%m-%d")
    uri = f"{GCS_BUCKET}/{location}/forecast_{run_date}.csv"
    upload_bytes(df.to_csv(index=False).encode(), uri, content_type="text/csv")
    logger.info(f"Uploaded → {uri}")
    return uri


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", choices=list(LOCATIONS.keys()), default=None)
    args = parser.parse_args()

    locs = [args.location] if args.location else list(LOCATIONS.keys())
    for loc in locs:
        df = run_forecast(loc)
        upload_forecast(loc, df)
