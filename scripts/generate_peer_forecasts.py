"""
Generate cross-location peer forecast features for the stacked ensemble (EXP-32).

Must be run AFTER all 5 base models have been trained.

For each target location, this script:
  1. Loads the saved Prophet models for all OTHER locations.
  2. Generates predictions for every date in the target location's training window.
  3. Normalises each by that other location's 28-day rolling actual mean.
  4. Averages across other locations → `peer_yhat` (a value near 1.0 = network typical,
     >1.0 = peers predicting above-average, <1.0 = peers predicting below-average).
  5. Saves to data/peer_forecasts/{location}.parquet.

At training time, peer_yhat is loaded by load_training_data() if the cache exists.
At inference time, run each base model first then re-run with peer_yhat set from
those base forecasts (two-pass stacking).

Key difference from network_momentum:
  - network_momentum: ratio of rolling ACTUALS (constant across the 14-day horizon)
  - peer_yhat: model PREDICTIONS normalised per-location (varies day-by-day in horizon,
    incorporates holiday/weather/trend from each peer model)

Usage:
    python -m scripts.generate_peer_forecasts
    python -m scripts.generate_peer_forecasts --location borough
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.data_prep import load_training_data, LOCATIONS
from src.utils.logging import get_logger

logger = get_logger(__name__)

PEER_FORECAST_DIR = Path("data/peer_forecasts")
MODEL_DIR = Path("models")


def generate_peer_forecast(target_loc: str) -> None:
    """Generate and cache peer_yhat for target_loc."""
    PEER_FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"{target_loc}: loading training data")
    target_df = load_training_data(target_loc)
    target_dates = target_df["ds"].reset_index(drop=True)

    other_locs = [loc for loc in LOCATIONS if loc != target_loc]
    peer_series: list[np.ndarray] = []

    for other_loc in other_locs:
        model_path = MODEL_DIR / f"{other_loc}.pkl"
        if not model_path.exists():
            logger.info(f"  {other_loc}: model not found, skipping")
            continue

        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        model   = saved["model"]
        log_y   = saved.get("log_y", False)
        regs    = saved["regressors"]

        logger.info(f"  {other_loc}: loading training data for regressor values")
        other_df = load_training_data(other_loc)

        # For each target date, find regressor values from the other location.
        # Dates the other location doesn't have get the other location's column means.
        col_means = other_df[regs].mean().to_dict()
        future = target_df[["ds"]].copy()
        other_reg = other_df[["ds"] + regs].drop_duplicates(subset="ds")  # guard against dup dates
        future = future.merge(other_reg, on="ds", how="left")
        for col in regs:
            future[col] = future[col].fillna(col_means[col])

        fc = model.predict(future[["ds"] + regs])
        yhat = np.expm1(fc["yhat"].values) if log_y else fc["yhat"].values
        yhat = np.maximum(yhat, 0.0)

        # Normalise by other location's rolling 28-day actual mean.
        other_rev = other_df.set_index("ds")["y"].sort_index()
        roll28 = other_rev.rolling(28, min_periods=14).mean()

        roll28_for_target = (
            target_dates.map(roll28.to_dict())
            .fillna(float(other_rev.mean()))
        )
        denom = np.maximum(roll28_for_target.values, 1.0)
        yhat_norm = np.clip(yhat / denom, 0.5, 2.0)

        peer_series.append(yhat_norm)
        logger.info(f"  {other_loc}: done (mean normalised yhat={yhat_norm.mean():.3f})")

    if not peer_series:
        logger.info(f"{target_loc}: no peer models — skipping")
        return

    peer_yhat = np.mean(peer_series, axis=0)

    out = pd.DataFrame({"ds": target_dates, "peer_yhat": peer_yhat})
    out.to_parquet(PEER_FORECAST_DIR / f"{target_loc}.parquet", index=False)
    logger.info(
        f"{target_loc}: saved {len(out)} peer_yhat values "
        f"(mean={peer_yhat.mean():.3f}, std={peer_yhat.std():.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", choices=list(LOCATIONS.keys()), default=None)
    args = parser.parse_args()

    locs = [args.location] if args.location else list(LOCATIONS.keys())
    for loc in locs:
        print(f"\n{'='*50}\n  {loc}\n{'='*50}")
        generate_peer_forecast(loc)


if __name__ == "__main__":
    main()
