"""
Forecast vs actual in the eval window — one panel per location.

Borough / Covent Garden: proper holdout (train on all-but-last-28, forecast last 28).
Short-history locations: pseudo-test (train on all-but-last-14, forecast last 14).

Run: python -m scripts.forecast_vs_actual
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import time

from src.training.data_prep import load_training_data, LOCATIONS
from src.training.train import MODEL_CONFIG, _uk_holidays_df, _regressors, _build_model

EVAL_WINDOW = 14
OUT_DIR = Path("plots")


def _load_with_retry(location: str, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            return load_training_data(location)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  {location}: fetch failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise

LOCATION_LABELS = {
    "battersea":     "Battersea",
    "borough":       "Borough",
    "canary_wharf":  "Canary Wharf",
    "covent_garden": "Covent Garden",
    "spitalfields":  "Spitalfields",
}


def _mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100


def _forecast_eval_window(location: str, df: pd.DataFrame):
    """
    Returns (dates, actuals, yhat, yhat_lower, yhat_upper, label, mape_val)
    for the eval window.  Proper holdout for long-history, pseudo-test for short.
    """
    log_y = MODEL_CONFIG.get(location, {}).get("log_y", False)
    regs = _regressors(location)
    can_split = len(df) >= 90

    if can_split:
        eval_df  = df.iloc[-EVAL_WINDOW:]
        train_df = df.iloc[:-EVAL_WINDOW]
        window_label = f"Test window  ({eval_df['ds'].iloc[0].strftime('%d %b')}–{eval_df['ds'].iloc[-1].strftime('%d %b')})"
    else:
        eval_df  = df.iloc[-EVAL_WINDOW:]
        train_df = df.iloc[:-EVAL_WINDOW]
        window_label = f"Last 14 days  ({eval_df['ds'].iloc[0].strftime('%d %b')}–{eval_df['ds'].iloc[-1].strftime('%d %b')})"

    model = _build_model(location)
    fit_df = train_df.copy()
    if log_y:
        fit_df["y"] = np.log1p(fit_df["y"])
    model.fit(fit_df)

    future = eval_df[["ds"] + regs].copy()
    fc = model.predict(future)

    if log_y:
        yhat    = np.expm1(fc["yhat"].values)
        yhat_lo = np.expm1(fc["yhat_lower"].values)
        yhat_hi = np.expm1(fc["yhat_upper"].values)
    else:
        yhat    = fc["yhat"].values
        yhat_lo = fc["yhat_lower"].values
        yhat_hi = fc["yhat_upper"].values

    m = _mape(eval_df["y"].values, yhat)
    return eval_df["ds"].values, eval_df["y"].values, yhat, yhat_lo, yhat_hi, window_label, m, can_split


def make_plot(locations_filter: list[str] | None = None):
    target_locations = locations_filter or list(LOCATIONS.keys())

    print("Loading data...")
    data = {}
    for location in target_locations:
        print(f"  {location}...")
        data[location] = _load_with_retry(location)

    n = len(target_locations)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 8))
    if n == 1:
        axes = [axes]
    fig.suptitle("Forecast vs Actual — Eval Window (last 14 days)", fontsize=15, y=1.01)

    for ax, location in zip(axes, target_locations):
        dates, actuals, yhat, yhat_lo, yhat_hi, window_label, m, is_holdout = _forecast_eval_window(location, data[location])

        x = np.arange(len(dates))
        width = 0.35

        ax.bar(x - width / 2, actuals, width, label="Actual",   color="#4a7ec7", alpha=0.85, zorder=3)
        ax.bar(x + width / 2, yhat,    width, label="Forecast", color="#e07b39", alpha=0.85, zorder=3)

        # Error line + % label between the two bar tops
        ylim_top = max(np.max(actuals), np.max(yhat)) * 1.18
        for i, (a, p) in enumerate(zip(actuals, yhat)):
            lo, hi = min(a, p), max(a, p)
            if hi > lo:
                ax.plot([i, i], [lo, hi], color="#cc3333", linewidth=1.8, zorder=4)
                pct = abs(a - p) / max(a, 1) * 100
                ax.text(i + 0.08, (lo + hi) / 2, f"{pct:.0f}%",
                        ha="left", va="center", fontsize=7, color="#cc3333", fontweight="bold")

        day_labels = [pd.Timestamp(d).strftime("%a\n%d %b") for d in dates]
        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, fontsize=8)

        ax.set_ylim(0, ylim_top)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

        holdout_tag = "holdout" if is_holdout else "pseudo-test*"
        ax.set_title(f"{LOCATION_LABELS[location]}", fontsize=13, fontweight="bold", pad=6)
        ax.set_xlabel(f"MAPE {m:.1f}%  ·  {holdout_tag}  ·  {window_label}", fontsize=9)
        ax.set_ylabel("Revenue (£)", fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4a7ec7", alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color="#e07b39", alpha=0.85),
    ]
    fig.legend(handles, ["Actual", "Forecast"], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    suffix = "_".join(target_locations) if locations_filter else "all"
    out_path = OUT_DIR / f"forecast_vs_actual_{suffix}.png"
    OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--locations", nargs="+", choices=list(LOCATIONS.keys()), default=None)
    args = parser.parse_args()
    make_plot(locations_filter=args.locations)
