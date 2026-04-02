"""
One-time script to build the GCS training parquet cache from raw Revel CSVs.

Run this once before the daily retrain job goes live. After this, retrain.py
does incremental daily updates (download 1 parquet, append 1 new day, re-upload).

Usage:
    DATA_SOURCE=gcs python -m scripts.build_training_data
    DATA_SOURCE=gcs python -m scripts.build_training_data --location borough
"""

import argparse

from src.training.data_prep import load_training_data, save_training_parquet, LOCATIONS
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_location(location: str) -> None:
    logger.info(f"=== {location}: building training parquet ===")
    df = load_training_data(location)
    save_training_parquet(location, df)
    logger.info(f"=== {location}: done ({len(df)} rows) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", choices=list(LOCATIONS.keys()), default=None)
    args = parser.parse_args()

    locs = [args.location] if args.location else list(LOCATIONS.keys())
    for loc in locs:
        build_location(loc)
    logger.info("Training parquet build complete.")
