"""
Daily retrain + forecast job. Designed to run as a Cloud Run Job.

For each location independently:
  1. Load latest data from GCS
  2. Data quality checks (anomaly, closure, structural break)
  3. Retrain model
  4. Validate against holdout — only promote if MAPE doesn't regress
  5. Save model to GCS model registry
  6. Run forecast, upload CSV to gs://jenki-forecast/{location}/

One location failing never affects the others.
Slack summary sent at end of run regardless of individual outcomes.

Usage (local):
    DATA_SOURCE=gcs python -m scripts.retrain
    DATA_SOURCE=gcs python -m scripts.retrain --location borough

Usage (Cloud Run Job):
    Set DATA_SOURCE=gcs in the job env vars.
"""

import argparse
import json
import os
import pickle
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.data_prep import (
    load_training_data, LOCATIONS, JENKI_COORDS, WEATHER_VARIABLES,
    TRAINING_START, EXCLUDE_DATES, ALL_REGRESSORS,
    _list_csvs, _extract_date, _read_csv, _fetch_weather,
    _get_footfall_features,
    save_training_parquet, load_training_parquet,
)
from src.training.train import train_location, MODEL_CONFIG, _build_model, _regressors, EVAL_WINDOW
from scripts.forecast import run_forecast, upload_forecast
from src.utils.gcs import upload_bytes, download_bytes
from src.utils.alerts import (
    alert_retrain_failure,
    alert_validation_gate,
    alert_data_quarantine,
    alert_structural_break,
    alert_retrain_success,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

GCS_MODEL_BUCKET  = "gs://jenki-forecast"
MAPE_TOLERANCE    = 1.5   # pp — promote if new MAPE ≤ prev + this
MAPE_MAX_SANE     = 50.0  # pp — above this the model is fundamentally broken, never promote
BREAK_THRESHOLD   = 0.60  # 7d avg / 28d avg below this = structural break
ANOMALY_MULTIPLE  = 4.0   # revenue > N × rolling mean = anomaly
CLOSURE_THRESHOLD = 0.05  # revenue < N × rolling median = likely closure

MODEL_LOCAL_DIR = Path("/tmp/models")  # ephemeral container storage


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------

def _model_meta_uri(location: str) -> str:
    return f"{GCS_MODEL_BUCKET}/models/{location}/metadata.json"


def _model_pkl_uri(location: str, run_date: str | None = None) -> str:
    tag = run_date or date.today().strftime("%Y-%m-%d")
    return f"{GCS_MODEL_BUCKET}/models/{location}/{tag}.pkl"


def _current_pkl_uri(location: str) -> str:
    return f"{GCS_MODEL_BUCKET}/models/{location}/current.pkl"


def _get_prev_mape(location: str) -> float | None:
    """Return the MAPE of the currently deployed model, or None if no model exists yet."""
    try:
        meta = json.loads(download_bytes(_model_meta_uri(location)))
        return meta.get("mape")
    except Exception:
        return None


def _save_model_to_gcs(location: str, model_obj: dict, mape: float, n_days: int) -> None:
    run_date = date.today().strftime("%Y-%m-%d")
    pkl_bytes = pickle.dumps(model_obj)

    # Versioned copy
    upload_bytes(pkl_bytes, _model_pkl_uri(location, run_date), "application/octet-stream")
    # Current pointer
    upload_bytes(pkl_bytes, _current_pkl_uri(location), "application/octet-stream")

    meta = {"mape": round(mape, 4), "trained_at": run_date, "n_days": n_days}
    upload_bytes(json.dumps(meta).encode(), _model_meta_uri(location), "application/json")
    logger.info(f"{location}: model saved to GCS (MAPE={mape:.2f}%)")


# ---------------------------------------------------------------------------
# Incremental training data loader
# ---------------------------------------------------------------------------

def _get_training_data(location: str) -> pd.DataFrame:
    """
    Load training data efficiently using a GCS parquet cache.

    First run: falls back to full rebuild from raw CSVs (slow, once only).
    Subsequent runs: downloads 1 parquet file, appends only new days, re-uploads.

    This replaces the old approach of downloading 500+ CSVs every day.
    """
    existing = load_training_parquet(location)

    if existing is None:
        logger.info(f"{location}: no parquet cache found — running full rebuild (first time only, will be slow)")
        df = load_training_data(location)
        save_training_parquet(location, df)
        return df

    # Check whether there are new dates to append
    last_date = pd.Timestamp(existing["ds"].max())
    yesterday = pd.Timestamp(date.today() - timedelta(days=1))

    if last_date >= yesterday:
        logger.info(f"{location}: parquet up to date ({last_date.date()})")
        return existing

    # Find CSVs with dates after last_date
    all_csvs = _list_csvs(location)
    new_csvs = [
        f for f in all_csvs
        if (d := _extract_date(Path(f).name)) and pd.Timestamp(d) > last_date
    ]

    if not new_csvs:
        logger.info(f"{location}: no new CSVs found since {last_date.date()}")
        return existing

    # Parse revenue from new CSVs only
    rows = []
    for f in new_csvs:
        date_str = _extract_date(Path(f).name)
        try:
            df_csv = _read_csv(f)
            if "Total Sales" not in df_csv.columns:
                continue
            revenue = df_csv["Total Sales"].sum()
            if revenue > 0:
                rows.append({"ds": pd.Timestamp(date_str), "y": revenue})
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not rows:
        logger.info(f"{location}: no new revenue rows after parsing")
        return existing

    new_rev = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)

    # Apply the same filters as load_training_data
    cutoff = TRAINING_START.get(location)
    if cutoff:
        new_rev = new_rev[new_rev["ds"] >= cutoff].reset_index(drop=True)
    excluded = EXCLUDE_DATES.get(location, [])
    if excluded:
        new_rev = new_rev[~new_rev["ds"].isin(pd.to_datetime(excluded))].reset_index(drop=True)

    if new_rev.empty:
        return existing

    # Fetch weather and footfall for new dates only
    lat, lng = JENKI_COORDS[location]
    new_start = new_rev["ds"].min().strftime("%Y-%m-%d")
    new_end   = new_rev["ds"].max().strftime("%Y-%m-%d")
    weather  = _fetch_weather(new_start, new_end, lat, lng)
    footfall = _get_footfall_features(location, new_rev["ds"])

    new_df = new_rev.merge(weather,  on="ds", how="left")
    new_df = new_df.merge(footfall, on="ds", how="left")

    # Network momentum: hold the most recent known value constant for new rows.
    # Network momentum is a smoothed 7d/28d ratio — day-to-day change is tiny.
    last_momentum = float(existing["network_momentum"].iloc[-1]) if "network_momentum" in existing.columns else 1.0
    new_df["network_momentum"] = last_momentum

    # Derived features (identical to load_training_data)
    new_df["temp_sq"]   = (new_df["apparent_temperature_max"] - 15.0) ** 2
    new_df["rainy_day"] = (new_df["precipitation_sum"] > 1.0).astype(float)
    new_df["precip_sq"] = new_df["precipitation_hours"] ** 2

    month = new_df["ds"].dt.month
    day   = new_df["ds"].dt.day
    new_df["globe_season_active"] = (
        ((month > 4) | ((month == 4) & (day >= 23))) &
        ((month < 10) | ((month == 10) & (day <= 26)))
    ).astype(float)

    new_df["event_impact_score"] = 0.0
    new_df["peer_yhat"] = 1.0

    # Merge and forward-fill any gaps
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="ds").sort_values("ds").reset_index(drop=True)

    for col in ALL_REGRESSORS:
        if col in combined.columns and combined[col].isnull().any():
            combined[col] = combined[col].ffill().bfill()

    logger.info(f"{location}: appended {len(new_df)} new day(s) to parquet (total {len(combined)})")
    save_training_parquet(location, combined)
    return combined


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def _quarantine_anomalies(location: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove revenue outliers and log them to GCS quarantine.
    Keeps the DataFrame clean for training.
    """
    rolling_mean = df["y"].rolling(28, min_periods=7).mean().shift(1)
    anomaly_mask = df["y"] > rolling_mean * ANOMALY_MULTIPLE
    anomalies = df.loc[anomaly_mask, "ds"].dt.strftime("%Y-%m-%d").tolist()

    if anomalies:
        q_uri = f"{GCS_MODEL_BUCKET}/quarantine/{location}/anomalies_{date.today()}.json"
        upload_bytes(json.dumps(anomalies).encode(), q_uri, "application/json")
        alert_data_quarantine(location, anomalies, f"Revenue > {ANOMALY_MULTIPLE}× rolling 28-day mean")
        logger.warning(f"{location}: quarantined {len(anomalies)} anomalous days: {anomalies}")
        df = df.loc[~anomaly_mask].reset_index(drop=True)

    return df


def _check_structural_break(location: str, df: pd.DataFrame) -> bool:
    """
    Returns True if recent 7-day avg revenue has dropped > 40% vs prior 28-day baseline.
    Triggers a Slack alert and skips retraining for this location.
    """
    if len(df) < 35:
        return False
    recent   = df["y"].iloc[-7:].mean()
    baseline = df["y"].iloc[-35:-7].mean()
    if baseline > 0 and (recent / baseline) < BREAK_THRESHOLD:
        alert_structural_break(location, recent, baseline)
        logger.warning(f"{location}: structural break — recent £{recent:,.0f} vs baseline £{baseline:,.0f}")
        return True
    return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test(model, location: str, regs: list, log_y: bool) -> None:
    """
    Predict on 3 future days and verify output is sensible.
    Catches: NaN predictions, all-zero output, negative output.
    Runs after fit but before promoting to GCS.
    """
    from datetime import date, timedelta
    from scripts.forecast import build_future_df

    start = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    end   = (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")

    future = build_future_df(location, start, end, regs)
    fc     = model.predict(future[["ds"] + regs])
    yhat   = np.expm1(fc["yhat"].values) if log_y else fc["yhat"].values

    if np.any(np.isnan(yhat)):
        raise ValueError(f"Smoke test failed: NaN in predictions for {start}→{end}")
    if np.all(yhat <= 0):
        raise ValueError(f"Smoke test failed: all-zero/negative predictions")

    logger.info(f"{location}: smoke test passed — avg predicted £{yhat.mean():,.0f}/day")


# ---------------------------------------------------------------------------
# Retrain + validate
# ---------------------------------------------------------------------------

def _retrain_and_validate(location: str, df: pd.DataFrame) -> tuple[dict, float]:
    """
    Retrain on all data and evaluate on the held-out test window.
    Returns (model_artifact, test_mape).
    """
    cfg   = MODEL_CONFIG.get(location, {})
    log_y = cfg.get("log_y", False)
    regs  = _regressors(location)
    n     = len(df)
    ew    = cfg.get("eval_window", EVAL_WINDOW)

    # Holdout eval
    test_df  = df.iloc[-ew:]
    val_df   = df.iloc[-(ew * 2):-ew]
    tv_df    = df.iloc[-(ew * 2):]
    train_df = df.iloc[:-(ew * 2)]

    m_test = _build_model(location)
    fit_tv = pd.concat([train_df, val_df]).reset_index(drop=True).copy()
    if log_y:
        fit_tv["y"] = np.log1p(fit_tv["y"])
    m_test.fit(fit_tv)

    fc = m_test.predict(test_df[["ds"] + regs])
    yhat = np.expm1(fc["yhat"].values) if log_y else fc["yhat"].values
    test_mape = float(np.mean(np.abs((test_df["y"].values - yhat) /
                                     np.where(test_df["y"].values == 0, 1, test_df["y"].values))) * 100)

    # Final model on all data
    m_final = _build_model(location)
    fit_all = df.copy()
    if log_y:
        fit_all["y"] = np.log1p(fit_all["y"])
    m_final.fit(fit_all)

    artifact = {"model": m_final, "log_y": log_y, "regressors": regs}
    logger.info(f"{location}: retrain complete — test MAPE {test_mape:.2f}%")
    return artifact, test_mape


# ---------------------------------------------------------------------------
# Per-location job
# ---------------------------------------------------------------------------

def retrain_location(location: str) -> dict:
    """
    Full retrain pipeline for one location.
    Returns {"mape": float, "promoted": bool}.
    Never raises — all failures are caught and alerted.
    """
    logger.info(f"=== {location}: starting retrain ===")
    try:
        df = _get_training_data(location)

        # Structural break check — pause retraining if revenue collapsed
        if _check_structural_break(location, df):
            return {"mape": _get_prev_mape(location) or 0.0, "promoted": False}

        # Anomaly quarantine
        df = _quarantine_anomalies(location, df)

        # Skip if insufficient data after cleaning
        if len(df) < 30:
            logger.warning(f"{location}: only {len(df)} days after cleaning — skipping")
            return {"mape": _get_prev_mape(location) or 0.0, "promoted": False}

        # Retrain and evaluate
        artifact, new_mape = _retrain_and_validate(location, df)

        # Hard sanity check — MAPE > 50% means the model is broken regardless of history
        if new_mape > MAPE_MAX_SANE:
            msg = f"MAPE {new_mape:.1f}% exceeds sanity limit of {MAPE_MAX_SANE}% — likely bad data or failed fit"
            logger.error(f"{location}: {msg}")
            alert_retrain_failure(location, msg)
            return {"mape": new_mape, "promoted": False}

        # Smoke test — predict 3 days forward and check for NaN / zeros
        _smoke_test(artifact["model"], location, artifact["regressors"], artifact["log_y"])

        # Validation gate
        prev_mape = _get_prev_mape(location)
        if prev_mape is not None and prev_mape > 0.0 and new_mape > prev_mape + MAPE_TOLERANCE:
            alert_validation_gate(location, new_mape, prev_mape)
            logger.warning(f"{location}: validation gate failed ({new_mape:.2f}% > {prev_mape:.2f}% + {MAPE_TOLERANCE}pp)")
            return {"mape": new_mape, "promoted": False}

        # Promote — save model and run forecast
        _save_model_to_gcs(location, artifact, new_mape, len(df))

        # Write to /tmp for forecast.py to load within the same job execution
        MODEL_LOCAL_DIR.mkdir(exist_ok=True)
        with open(MODEL_LOCAL_DIR / f"{location}.pkl", "wb") as f:
            pickle.dump(artifact, f)

        forecast_df = run_forecast(location, model_dir=MODEL_LOCAL_DIR)
        upload_forecast(location, forecast_df)

        logger.info(f"=== {location}: done (MAPE={new_mape:.2f}%, promoted) ===")
        return {"mape": new_mape, "promoted": True}

    except Exception as e:
        logger.error(f"{location}: retrain failed — {e}", exc_info=True)
        alert_retrain_failure(location, str(e))
        return {"mape": _get_prev_mape(location) or 0.0, "promoted": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", choices=list(LOCATIONS.keys()), default=None)
    args = parser.parse_args()

    locs = [args.location] if args.location else list(LOCATIONS.keys())
    results = {loc: retrain_location(loc) for loc in locs}
    alert_retrain_success(results)
    logger.info("Daily retrain job complete.")
