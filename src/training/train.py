"""
Train a Prophet model per location with weather + footfall regressors.
Evaluates on train / val / test splits before saving final model.

Usage:
    python -m src.training.train                     # all locations
    python -m src.training.train --location borough  # one location
"""

import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
import holidays
from dateutil.easter import easter

from src.training.data_prep import load_training_data, LOCATIONS, ALL_REGRESSORS
from src.training.london_calendar import get_london_events_df
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models")
HORIZON_DAYS = 14
EVAL_WINDOW = 14

# Per-location model configuration.
# log_y: fit on log1p(revenue), predictions are expm1'd back. Saves as {"log_y": True} flag in pickle.
# changepoint_prior_scale: Prophet trend flexibility (default 0.05). Higher = more adaptive trend.
# extra_regressors: location-specific features added on top of ALL_REGRESSORS.
MODEL_CONFIG: dict[str, dict] = {
    "battersea":     {"changepoint_prior_scale": 0.1, "extra_regressors": ["network_momentum"]},
    "borough":       {"changepoint_prior_scale": 0.1, "extra_regressors": ["rainy_day", "precip_sq", "network_momentum"]},
    "canary_wharf":  {"changepoint_prior_scale": 0.1, "extra_regressors": ["network_momentum"]},
    "covent_garden": {"changepoint_prior_scale": 0.1, "log_y": True, "extra_regressors": ["network_momentum"]},
    "spitalfields":  {"changepoint_prior_scale": 0.1, "extra_regressors": ["network_momentum"]},
}



def _uk_holidays_df(location: str | None = None) -> pd.DataFrame:
    years = range(2024, 2030)
    uk = holidays.country_holidays("GB", subdiv="ENG", years=years)
    rows = [{"ds": str(d), "holiday": name} for d, name in uk.items()]
    # Mothering Sunday (4th Sunday of Lent = 3 weeks before Easter).
    for year in years:
        rows.append({"ds": str(easter(year) - timedelta(weeks=3)), "holiday": "Mothering Sunday"})
    bank_holidays = pd.DataFrame(rows)
    london_events = get_london_events_df(years, location=location)
    return pd.concat([bank_holidays, london_events], ignore_index=True).drop_duplicates(subset=["ds", "holiday"])


def _regressors(location: str) -> list[str]:
    extras = MODEL_CONFIG.get(location, {}).get("extra_regressors", [])
    return ALL_REGRESSORS + extras


def _build_model(location: str) -> Prophet:
    cfg = MODEL_CONFIG.get(location, {})
    cps = cfg.get("changepoint_prior_scale", 0.05)
    model = Prophet(
        holidays=_uk_holidays_df(location=location),
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=cps,
        uncertainty_samples=0,  # disable Stan sampling at predict time — CI computed empirically
    )
    for reg in _regressors(location):
        model.add_regressor(reg)
    return model


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return {"MAE": round(mae, 2), "MAPE": round(mape, 2), "RMSE": round(rmse, 2)}


def _evaluate(model: Prophet, df: pd.DataFrame, log_y: bool = False, location: str = "") -> dict:
    """Generate in-sample predictions for df and return metrics."""
    regs = _regressors(location) if location else ALL_REGRESSORS
    future = df[["ds"] + regs].copy()
    forecast = model.predict(future)
    yhat = forecast["yhat"].values
    if log_y:
        yhat = np.expm1(yhat)
    return _metrics(df["y"].values, yhat)


def _apply_log_y(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = np.log1p(df["y"])
    return df


def train_location(location: str) -> None:
    logger.info(f"=== {location} ===")
    df = load_training_data(location)
    log_y = MODEL_CONFIG.get(location, {}).get("log_y", False)

    n = len(df)
    # Require at least 90 days so the train slice (n - 2*EVAL_WINDOW) has 60+ days.
    can_split = n >= 90

    if not can_split:
        logger.info(f"{location}: only {n} days — training on full dataset")
        fit_df = _apply_log_y(df) if log_y else df
        model = _build_model(location)
        model.fit(fit_df)
        logger.info(f"{location} in-sample metrics: {_evaluate(model, df, log_y, location)}")
    else:
        test_df  = df.iloc[-EVAL_WINDOW:]
        val_df   = df.iloc[-(EVAL_WINDOW * 2):-EVAL_WINDOW]
        train_df = df.iloc[:-(EVAL_WINDOW * 2)]

        logger.info(
            f"{location}: train={len(train_df)}d, val={len(val_df)}d, test={len(test_df)}d"
        )

        model = _build_model(location)
        model.fit(_apply_log_y(train_df) if log_y else train_df)
        logger.info(f"{location} val metrics:  {_evaluate(model, val_df, log_y, location)}")

        model = _build_model(location)
        tv_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        model.fit(_apply_log_y(tv_df) if log_y else tv_df)
        logger.info(f"{location} test metrics: {_evaluate(model, test_df, log_y, location)}")

        # Final model on all data
        model = _build_model(location)
        model.fit(_apply_log_y(df) if log_y else df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = MODEL_DIR / f"{location}.pkl"
    with open(artifact, "wb") as f:
        pickle.dump({
            "model": model,
            "log_y": log_y,
            "regressors": _regressors(location),
        }, f)
    logger.info(f"{location}: model saved to {artifact}")


def train_all() -> None:
    for location in LOCATIONS:
        train_location(location)
    logger.info("All models trained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", choices=list(LOCATIONS.keys()), default=None)
    args = parser.parse_args()

    if args.location:
        train_location(args.location)
    else:
        train_all()
