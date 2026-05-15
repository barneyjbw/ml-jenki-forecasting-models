"""
Backfill historical revenue forecast CSVs.

For each of the past N days, loads the current promoted model from GCS and
generates a 28-day forecast starting from that date. Uploads to both GCS
buckets under the historical run_date filename.

Optimised: expensive GCS downloads (training data, footfall, weather) happen
once per location, not once per run_date.

Usage:
    python -m scripts.backfill_forecasts [--days 14] [--location borough]
"""
import argparse
import io
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from src.training.data_prep import (
    LOCATIONS, JENKI_COORDS, WEATHER_VARIABLES,
    TUBE_STRIKE_DATES, STRIKE_MULTIPLIER, STATION_LINES,
    _get_footfall_features, load_training_data,
)
from src.training import weather_cache
from src.utils.gcs import download_bytes, upload_bytes
from src.utils.logging import get_logger
from scripts.forecast import (
    LOCATION_SLUGS, GCS_BUCKET, SALES_PRED_DAILY_ROOT,
    FORECAST_HORIZON, fetch_weather_forecast,
    current_network_momentum,
)

logger = get_logger(__name__)

GCS_MODEL_ROOT = f"{GCS_BUCKET}/models"


# ── Per-location precomputation ───────────────────────────────────────────────

def _precompute(location: str, saved: dict, all_dates: pd.DatetimeIndex) -> dict:
    """Compute everything that is constant across run_dates for this location."""
    model = saved["model"]
    log_y = saved.get("log_y", False)
    regs = saved["regressors"]

    # 1. Footfall for the full date range
    logger.info(f"  {location}: precomputing footfall...")
    footfall = _get_footfall_features(location, all_dates)

    # 2. Network momentum (scalar — constant over backfill)
    momentum = None
    if "network_momentum" in regs:
        logger.info(f"  {location}: precomputing network momentum...")
        momentum = current_network_momentum(location)

    # 3. Day-of-week MAE for empirical bounds (requires in-sample prediction)
    logger.info(f"  {location}: precomputing empirical bounds (loading training data)...")
    df_train = load_training_data(location)
    fc_in = model.predict(df_train[["ds"] + regs])
    yhat_in = np.maximum(
        np.expm1(fc_in["yhat"].values) if log_y else fc_in["yhat"].values, 0.0
    )
    abs_err = np.abs(df_train["y"].values - yhat_in)
    df_err = pd.DataFrame({"dow": df_train["ds"].dt.dayofweek, "err": abs_err})
    dow_mae = df_err.groupby("dow")["err"].mean()

    return {"footfall": footfall, "momentum": momentum, "dow_mae": dow_mae}


def _precompute_weather(location: str, earliest: date, latest: date) -> pd.DataFrame:
    """Fetch archive + forecast weather for the full backfill window once."""
    today = date.today()
    lat, lng = JENKI_COORDS[location]
    parts = []

    if earliest < today:
        hist_end = min(latest, today - timedelta(days=1))
        logger.info(f"  {location}: fetching archive weather {earliest}..{hist_end}")
        hist = weather_cache.fetch_weather(
            earliest.strftime("%Y-%m-%d"),
            hist_end.strftime("%Y-%m-%d"),
            lat, lng, WEATHER_VARIABLES, location,
        )
        parts.append(hist)

    if latest >= today:
        fcast_start = max(earliest, today)
        logger.info(f"  {location}: fetching forecast weather {fcast_start}..{latest}")
        fcast = fetch_weather_forecast(
            fcast_start.strftime("%Y-%m-%d"),
            latest.strftime("%Y-%m-%d"),
            lat, lng,
        )
        parts.append(fcast)

    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates("ds").sort_values("ds").reset_index(drop=True)


# ── Per-run_date inference ────────────────────────────────────────────────────

def _build_future_df(
    location: str, run_date: date, regs: list,
    weather_all: pd.DataFrame, cache: dict,
) -> pd.DataFrame:
    horizon = FORECAST_HORIZON[location]
    end = run_date + timedelta(days=horizon - 1)
    dates = pd.date_range(start=run_date, end=end, freq="D")

    # Slice pre-fetched weather
    mask = (weather_all["ds"] >= pd.Timestamp(run_date)) & (weather_all["ds"] <= pd.Timestamp(end))
    weather = weather_all[mask].reset_index(drop=True)

    # Slice pre-fetched footfall
    footfall = cache["footfall"]
    f_mask = (footfall["ds"] >= pd.Timestamp(run_date)) & (footfall["ds"] <= pd.Timestamp(end))
    footfall_slice = footfall[f_mask].reset_index(drop=True)

    df = pd.DataFrame({"ds": pd.to_datetime(dates)})
    df = df.merge(weather, on="ds", how="left")
    df = df.merge(footfall_slice, on="ds", how="left")

    for col in WEATHER_VARIABLES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].ffill().bfill()

    df["temp_sq"] = (df["apparent_temperature_max"] - 15.0) ** 2
    df["rainy_day"] = (df["precipitation_sum"] > 1.0).astype(float)
    df["precip_sq"] = df["precipitation_hours"] ** 2

    month, day = df["ds"].dt.month, df["ds"].dt.day
    df["globe_season_active"] = (
        ((month > 4) | ((month == 4) & (day >= 23))) &
        ((month < 10) | ((month == 10) & (day <= 26)))
    ).astype(float)

    if "network_momentum" in regs:
        df["network_momentum"] = cache["momentum"]
    if "peer_yhat" in regs:
        df["peer_yhat"] = 1.0
    if "data_source" in regs:
        df["data_source"] = 0

    return df


def _apply_bounds(yhat: np.ndarray, future_dates: pd.DatetimeIndex, dow_mae: pd.Series):
    future_dows = pd.to_datetime(future_dates).dt.dayofweek
    sigma = np.array([dow_mae.get(d, dow_mae.mean()) for d in future_dows])
    return np.maximum(yhat - 1.5 * sigma, 0.0), yhat + 1.5 * sigma


def _check_tfl_strikes(location: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    lines = STATION_LINES.get(location, [])
    strike_dates = set()
    for line_id in lines:
        try:
            r = requests.get(f"https://api.tfl.gov.uk/Line/{line_id}/Status", timeout=10)
            r.raise_for_status()
            for line_data in r.json():
                for status in line_data.get("lineStatuses", []):
                    reason = (status.get("reason") or "").lower()
                    if "strike" not in reason and "industrial action" not in reason:
                        continue
                    for window in status.get("validityPeriods", []):
                        from_d = pd.Timestamp(window.get("fromDate", "")).normalize()
                        to_d = pd.Timestamp(window.get("toDate", "")).normalize()
                        for d in dates:
                            if from_d <= d <= to_d:
                                strike_dates.add(d)
        except Exception as e:
            logger.warning(f"{location}: TfL check failed for {line_id}: {e}")
    return pd.DataFrame([
        {"ds": d, "major_disruption": 1.0 if d in strike_dates else 0.0} for d in dates
    ])


def _apply_strike_adjustment(location: str, future: pd.DataFrame, yhat: np.ndarray) -> np.ndarray:
    dates = future["ds"]
    api_strikes = _check_tfl_strikes(location, dates)
    strike_flags = api_strikes.set_index("ds")["major_disruption"]
    known = set(pd.to_datetime(TUBE_STRIKE_DATES))
    for d in dates:
        if d in known:
            strike_flags.loc[d] = 1.0
    if strike_flags.sum() == 0:
        return yhat
    multiplier = STRIKE_MULTIPLIER.get(location, 1.0)
    adjusted = yhat.copy()
    for i, d in enumerate(dates):
        if strike_flags.get(d, 0.0) == 1.0:
            adjusted[i] = yhat[i] * multiplier
    return adjusted


def _run_for_date(
    location: str, run_date: date, saved: dict,
    weather_all: pd.DataFrame, cache: dict,
) -> pd.DataFrame:
    model = saved["model"]
    log_y = saved.get("log_y", False)
    regs = saved["regressors"]

    future = _build_future_df(location, run_date, regs, weather_all, cache)
    fc = model.predict(future[["ds"] + regs])

    yhat = np.maximum(np.expm1(fc["yhat"].values) if log_y else fc["yhat"].values, 0.0)
    yhat = _apply_strike_adjustment(location, future, yhat)
    lower, upper = _apply_bounds(yhat, future["ds"], cache["dow_mae"])

    return pd.DataFrame({
        "date": future["ds"].dt.strftime("%Y-%m-%d"),
        "predicted_revenue": np.round(yhat, 2),
        "lower_bound": np.round(lower, 2),
        "upper_bound": np.round(upper, 2),
    })


# ── Upload ────────────────────────────────────────────────────────────────────

def _upload(location: str, df: pd.DataFrame, run_date: date) -> None:
    slug = LOCATION_SLUGS.get(location, location)
    run_date_str = run_date.strftime("%Y%m%d")
    payload = df.to_csv(index=False).encode()

    primary = f"{SALES_PRED_DAILY_ROOT}/{slug}/{run_date_str}.csv"
    upload_bytes(payload, primary, content_type="text/csv")
    logger.info(f"    → {primary}")

    mirror = f"{GCS_BUCKET}/revenue-forecast/{slug}/{run_date_str}.csv"
    try:
        upload_bytes(payload, mirror, content_type="text/csv")
    except Exception as e:
        logger.warning(f"    mirror failed: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--location", choices=list(LOCATIONS.keys()), default=None)
    args = parser.parse_args()

    targets = [args.location] if args.location else list(LOCATIONS.keys())
    today = date.today()
    run_dates = [today - timedelta(days=d) for d in range(args.days, 0, -1)]
    # Date range covering all forecasts: earliest run_date to latest run_date + 27
    earliest = run_dates[0]
    latest = run_dates[-1] + timedelta(days=FORECAST_HORIZON.get("borough", 28) - 1)
    all_dates = pd.date_range(start=earliest, end=latest, freq="D")

    for loc in targets:
        logger.info(f"{'='*60}\nBackfill revenue: {loc}\n{'='*60}")

        uri = f"{GCS_MODEL_ROOT}/{loc}/current.pkl"
        try:
            saved = pickle.load(io.BytesIO(download_bytes(uri)))
        except Exception as e:
            logger.error(f"{loc}: cannot load model ({uri}): {e}")
            continue

        # One-time precomputation per location
        try:
            cache = _precompute(loc, saved, all_dates)
            weather_all = _precompute_weather(loc, earliest, latest)
        except Exception as e:
            logger.error(f"{loc}: precompute failed: {e}")
            continue

        for run_date in run_dates:
            logger.info(f"  {loc}  run_date={run_date}")
            try:
                df = _run_for_date(loc, run_date, saved, weather_all, cache)
                _upload(loc, df, run_date)
            except Exception as e:
                logger.error(f"  {loc}/{run_date}: {e}")

    logger.info("Revenue backfill complete.")


if __name__ == "__main__":
    main()
