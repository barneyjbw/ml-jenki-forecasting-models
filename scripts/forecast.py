"""
Production inference — generate revenue forecasts for all Jenki locations.

Horizon: 14 days for Borough + Covent Garden, 7 days for Battersea / Canary Wharf / Spitalfields.
Output:  gs://jenki-forecast/revenue-forecast/{location}/{LOCATION_ID}-daily-{DDMMYY}.csv
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.training.data_prep import (
    LOCATIONS, JENKI_COORDS, WEATHER_VARIABLES, STATION_LINES,
    TUBE_STRIKE_DATES, STRIKE_MULTIPLIER,
    _get_footfall_features, _load_revenue, load_training_data,
)
from src.training.train import _regressors, MODEL_CONFIG
from src.utils.gcs import upload_bytes
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_DIR  = Path("models")
GCS_BUCKET = "gs://jenki-forecast"
# Public customer-facing output: Revenue forecasts land under this prefix.
# Writing here is additive to the internal jenki-forecast location.
SALES_PRED_DAILY_ROOT = "gs://bombe-sales-predictions/jenki/location-revenue-by-day"

LOCATION_IDS: dict[str, str] = {
    "battersea":     "L6GF6Z26CV7BM",
    "borough":       "LWVAYYMFT3XKP",
    "canary_wharf":  "LQ4TFTDQYXY3D",
    "covent_garden": "LZX5X6V4QY6MJ",
    "spitalfields":  "LK2EMH64185DE",
}

FORECAST_HORIZON: dict[str, int] = {
    "borough":       28,
    "covent_garden": 28,
    "battersea":     28,
    "canary_wharf":  28,
    "spitalfields":  28,
}


# ---------------------------------------------------------------------------
# Shared inference utilities (imported by plots_with_forecast.py)
# ---------------------------------------------------------------------------

def fetch_weather_forecast(start_date: str, end_date: str, lat: float, lng: float) -> pd.DataFrame:
    """Fetch daily weather from Open-Meteo forecast API.

    The forecast API only serves ~16 days ahead. For longer horizons, fetch what the
    API supports, then extend by holding the last observed row forward. Weather-driven
    accuracy at 17-28d horizon is dominated by seasonality anyway, so this is fine.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    api_end = min(end_ts, start_ts + pd.Timedelta(days=15))  # inclusive: 16 days

    params = {
        "latitude":   lat,
        "longitude":  lng,
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "end_date":   api_end.strftime("%Y-%m-%d"),
        "daily":      ",".join(WEATHER_VARIABLES),
        "timezone":   "Europe/London",
    }

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _get():
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    data = _get()["daily"]
    missing_cols = [c for c in WEATHER_VARIABLES if c not in data]
    if missing_cols:
        raise ValueError(f"Weather forecast API missing columns: {missing_cols}")

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "ds"})

    # Extend past api_end by carrying the 7-day trailing mean forward.
    if end_ts > api_end and len(df):
        tail = df.tail(min(7, len(df)))
        fill = {c: tail[c].mean() if df[c].dtype.kind in "fi" else tail[c].iloc[-1]
                for c in df.columns if c != "ds"}
        extra_dates = pd.date_range(api_end + pd.Timedelta(days=1), end_ts, freq="D")
        extra = pd.DataFrame({"ds": extra_dates, **{k: [v] * len(extra_dates) for k, v in fill.items()}})
        df = pd.concat([df, extra], ignore_index=True)

    return df


def _check_tfl_strikes(location: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Check TfL API for upcoming strikes on relevant lines.
    Returns a DataFrame with ds + major_disruption (0 or 1).

    Only flags strikes and industrial action, not routine disruptions.
    Falls back to 0 if the API is unreachable.
    """
    lines = STATION_LINES.get(location, [])
    strike_dates = set()

    for line_id in lines:
        try:
            url = f"https://api.tfl.gov.uk/Line/{line_id}/Status"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            for line_data in r.json():
                for status in line_data.get("lineStatuses", []):
                    reason = (status.get("reason") or "").lower()
                    is_strike = "strike" in reason or "industrial action" in reason
                    if not is_strike:
                        continue
                    for window in status.get("validityPeriods", []):
                        from_date = pd.Timestamp(window.get("fromDate", "")).normalize()
                        to_date = pd.Timestamp(window.get("toDate", "")).normalize()
                        for d in dates:
                            if from_date <= d <= to_date:
                                strike_dates.add(d)
        except Exception as e:
            logger.warning(f"{location}: TfL API check failed for {line_id} ({e}) - assuming no strike")

    rows = [{"ds": d, "major_disruption": 1.0 if d in strike_dates else 0.0} for d in dates]
    flagged = sum(1 for r in rows if r["major_disruption"] == 1.0)
    if flagged:
        logger.info(f"{location}: TfL strike detected on {flagged} forecast day(s)")
    return pd.DataFrame(rows)


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

    # Forward-fill any missing weather values (partial API response or date gap)
    for col in WEATHER_VARIABLES:
        if df[col].isnull().any():
            logger.warning(f"{location}: NaN in weather column '{col}' — forward-filling")
            df[col] = df[col].ffill().bfill()

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

    # data_source regressor (Covent Garden): future is always Square (0).
    # See Exp C: CG -8.94pp with this regressor; other locations regress.
    if "data_source" in regs:
        df["data_source"] = 0

    return df


# ---------------------------------------------------------------------------
# Empirical confidence intervals
# ---------------------------------------------------------------------------

def _empirical_bounds(
    location: str, saved: dict, yhat: np.ndarray, future_dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ±1.5× day-of-week MAE bounds from in-sample residuals.
    1.5× MAE ≈ 90% coverage under a Laplace error distribution.
    Far more useful than Prophet's ±54% simulated bands.
    """
    log_y = saved.get("log_y", False)
    regs  = saved["regressors"]
    model = saved["model"]

    df = load_training_data(location)
    fc_in = model.predict(df[["ds"] + regs])
    yhat_in = np.maximum(np.expm1(fc_in["yhat"].values) if log_y else fc_in["yhat"].values, 0.0)

    abs_err = np.abs(df["y"].values - yhat_in)
    df_err  = pd.DataFrame({"dow": df["ds"].dt.dayofweek, "err": abs_err})
    dow_mae = df_err.groupby("dow")["err"].mean()

    future_dows = pd.to_datetime(future_dates).dt.dayofweek
    sigma = np.array([dow_mae.get(d, dow_mae.mean()) for d in future_dows])

    return np.maximum(yhat - 1.5 * sigma, 0.0), yhat + 1.5 * sigma


# ---------------------------------------------------------------------------
# Strike adjustment (post-prediction)
# ---------------------------------------------------------------------------

def _apply_strike_adjustment(
    location: str, future: pd.DataFrame, yhat: np.ndarray
) -> np.ndarray:
    """
    Check for upcoming strikes via TfL API + known strike dates.
    Apply per-location multiplier to affected days.
    Non-strike days are completely untouched.
    """
    dates = future["ds"]

    # Check TfL API for live strike announcements
    api_strikes = _check_tfl_strikes(location, dates)
    strike_flags = api_strikes.set_index("ds")["major_disruption"]

    # Also check against our known strike date list
    known_strikes = set(pd.to_datetime(TUBE_STRIKE_DATES))
    for d in dates:
        if d in known_strikes:
            strike_flags.loc[d] = 1.0

    if strike_flags.sum() == 0:
        return yhat

    multiplier = STRIKE_MULTIPLIER.get(location, 1.0)
    adjusted = yhat.copy()
    for i, d in enumerate(dates):
        if strike_flags.get(d, 0.0) == 1.0:
            adjusted[i] = yhat[i] * multiplier
            logger.info(f"{location}: {d.strftime('%Y-%m-%d')} strike adjustment applied ({multiplier:.0%} of normal)")
    return adjusted


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_forecast(location: str, model_dir: Path | None = None) -> pd.DataFrame:
    """Run inference. Returns: date, predicted_revenue, lower_bound, upper_bound."""
    _model_dir = model_dir or MODEL_DIR
    horizon = FORECAST_HORIZON[location]
    start   = date.today().strftime("%Y-%m-%d")
    end     = (date.today() + timedelta(days=horizon - 1)).strftime("%Y-%m-%d")
    logger.info(f"{location}: forecasting {start} → {end}")

    with open(_model_dir / f"{location}.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    log_y = saved.get("log_y", False)
    regs  = saved["regressors"]

    future = build_future_df(location, start, end, regs)
    fc     = model.predict(future[["ds"] + regs])

    def _out(arr: np.ndarray) -> np.ndarray:
        return np.maximum(np.expm1(arr) if log_y else arr, 0.0)

    yhat  = _out(fc["yhat"].values)

    nan_count = int(np.isnan(yhat).sum())
    if nan_count:
        raise ValueError(
            f"{location}: model produced {nan_count} NaN prediction(s) — "
            f"likely NaN in regressors. Check weather/footfall data."
        )

    # Apply strike adjustment (post-prediction, per-location multiplier)
    yhat = _apply_strike_adjustment(location, future, yhat)

    lower, upper = _empirical_bounds(location, saved, yhat, future["ds"])

    result = pd.DataFrame({
        "date":              future["ds"].dt.strftime("%Y-%m-%d"),
        "predicted_revenue": np.round(yhat,  2),
        "lower_bound":       np.round(lower, 2),
        "upper_bound":       np.round(upper, 2),
    })
    logger.info(f"{location}: forecast ready\n{result.to_string(index=False)}")
    return result


# ---------------------------------------------------------------------------
# GCS upload
# ---------------------------------------------------------------------------

LOCATION_SLUGS: dict[str, str] = {
    "battersea":     "battersea",
    "borough":       "borough-market",
    "canary_wharf":  "canary-wharf",
    "covent_garden": "covent-garden",
    "spitalfields":  "spitalfields",
}


def upload_forecast(location: str, df: pd.DataFrame) -> str:
    """Write daily revenue forecast CSV.

    Primary: gs://bombe-sales-predictions/jenki/location-revenue-by-day/{slug}/{yyyymmdd}.csv
    Mirror:  gs://jenki-forecast/revenue-forecast/{slug}/{yyyymmdd}.csv  (internal,
             kept for existing readers/monitors until they migrate).
    """
    slug = LOCATION_SLUGS.get(location, location)
    run_date = date.today().strftime("%Y%m%d")
    payload = df.to_csv(index=False).encode()

    primary = f"{SALES_PRED_DAILY_ROOT}/{slug}/{run_date}.csv"
    upload_bytes(payload, primary, content_type="text/csv")
    logger.info(f"Uploaded → {primary}")

    mirror = f"{GCS_BUCKET}/revenue-forecast/{slug}/{run_date}.csv"
    try:
        upload_bytes(payload, mirror, content_type="text/csv")
    except Exception as e:
        logger.warning(f"mirror upload failed ({mirror}): {e}")

    return primary


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
