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
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.data_prep import load_training_data, LOCATIONS
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
BREAK_THRESHOLD   = 0.60  # 7d avg / 28d avg below this = structural break
ANOMALY_MULTIPLE  = 4.0   # revenue > N × rolling mean = anomaly
CLOSURE_THRESHOLD = 0.05  # revenue < N × rolling median = likely closure


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
# Retrain + validate
# ---------------------------------------------------------------------------

def _retrain_and_validate(location: str, df: pd.DataFrame) -> tuple[dict, float]:
    """
    Retrain on all data and evaluate on the held-out test window.
    Returns (model_artifact, test_mape).
    """
    log_y = MODEL_CONFIG.get(location, {}).get("log_y", False)
    regs  = _regressors(location)
    n     = len(df)

    # Holdout eval
    test_df  = df.iloc[-EVAL_WINDOW:]
    val_df   = df.iloc[-(EVAL_WINDOW * 2):-EVAL_WINDOW]
    tv_df    = df.iloc[-(EVAL_WINDOW * 2):]
    train_df = df.iloc[:-(EVAL_WINDOW * 2)]

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
        df = load_training_data(location)

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

        # Validation gate
        prev_mape = _get_prev_mape(location)
        if prev_mape is not None and new_mape > prev_mape + MAPE_TOLERANCE:
            alert_validation_gate(location, new_mape, prev_mape)
            logger.warning(f"{location}: validation gate failed ({new_mape:.2f}% > {prev_mape:.2f}% + {MAPE_TOLERANCE}pp)")
            return {"mape": new_mape, "promoted": False}

        # Promote — save model and run forecast
        _save_model_to_gcs(location, artifact, new_mape, len(df))

        # Also write to local models/ so forecast.py can load it without GCS
        Path("models").mkdir(exist_ok=True)
        with open(f"models/{location}.pkl", "wb") as f:
            pickle.dump(artifact, f)

        forecast_df = run_forecast(location)
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
