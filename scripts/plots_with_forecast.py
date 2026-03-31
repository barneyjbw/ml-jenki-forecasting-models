"""
Per-location revenue plots: full training history + eval holdout + forward forecast.

Borough / Covent Garden: val window (orange) + test window (red) + 14-day forecast (teal).
Short-history locations: in-sample fit + 7-day forecast (teal).

One PNG per location saved to plots/{location}_forecast.png.

Run: python -m scripts.plots_with_forecast
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from src.training.data_prep import load_training_data, LOCATIONS
from src.training.train import MODEL_CONFIG, _build_model, _regressors
from scripts.forecast import build_future_df, FORECAST_HORIZON

MODEL_DIR   = Path("models")
OUT_DIR     = Path("plots")
EVAL_WINDOW = 14

LOCATION_LABELS = {
    "battersea":     "Battersea",
    "borough":       "Borough",
    "canary_wharf":  "Canary Wharf",
    "covent_garden": "Covent Garden",
    "spitalfields":  "Spitalfields",
}

COLOURS = {
    "actual":   "#555555",
    "fitted":   "#4a7ec7",
    "val":      "#e07b39",
    "test":     "#c0392b",
    "forecast": "#1a9e8f",
}


def _mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100


def _unlog(arr, log_y):
    return np.expm1(arr) if log_y else arr


def _clip(arr):
    return np.maximum(arr, 0.0)


def _fmt_gbp(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))


def _get_forward_forecast(location: str, last_date: pd.Timestamp, regs: list) -> pd.DataFrame:
    """Forecast starting the day after last training data."""
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end   = (last_date + pd.Timedelta(days=FORECAST_HORIZON[location])).strftime("%Y-%m-%d")

    with open(MODEL_DIR / f"{location}.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    log_y = saved.get("log_y", False)

    future = build_future_df(location, start, end, regs)
    fc = model.predict(future[["ds"] + regs])

    return pd.DataFrame({
        "ds":   future["ds"],
        "yhat": _clip(_unlog(fc["yhat"].values,       log_y)),
        "lo":   _clip(_unlog(fc["yhat_lower"].values, log_y)),
        "hi":   _clip(_unlog(fc["yhat_upper"].values, log_y)),
    })


def plot_location(location: str, out_dir: Path) -> None:
    df    = load_training_data(location)
    log_y = MODEL_CONFIG.get(location, {}).get("log_y", False)
    regs  = _regressors(location)
    label = LOCATION_LABELS[location]
    n     = len(df)
    can_split = n >= 90

    fwd = _get_forward_forecast(location, df["ds"].max(), regs)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(f"{label} — Revenue: History + Forecast", fontsize=14, fontweight="bold", y=0.99)

    if can_split:
        test_df  = df.iloc[-EVAL_WINDOW:]
        val_df   = df.iloc[-(EVAL_WINDOW * 2):-EVAL_WINDOW]
        train_df = df.iloc[:-(EVAL_WINDOW * 2)]
        tv_df    = pd.concat([train_df, val_df]).reset_index(drop=True)

        # Val predictions (model fit on train only)
        m_val = _build_model(location)
        fit_tr = train_df.copy()
        if log_y:
            fit_tr["y"] = np.log1p(fit_tr["y"])
        m_val.fit(fit_tr)
        yhat_tr = _clip(_unlog(m_val.predict(train_df[["ds"] + regs])["yhat"].values, log_y))

        fc_val = m_val.predict(val_df[["ds"] + regs])
        yhat_val    = _clip(_unlog(fc_val["yhat"].values,       log_y))
        yhat_val_lo = _clip(_unlog(fc_val["yhat_lower"].values, log_y))
        yhat_val_hi = _clip(_unlog(fc_val["yhat_upper"].values, log_y))

        # Test predictions (model fit on train+val)
        m_test = _build_model(location)
        fit_tv = tv_df.copy()
        if log_y:
            fit_tv["y"] = np.log1p(fit_tv["y"])
        m_test.fit(fit_tv)

        fc_test = m_test.predict(test_df[["ds"] + regs])
        yhat_test    = _clip(_unlog(fc_test["yhat"].values,       log_y))
        yhat_test_lo = _clip(_unlog(fc_test["yhat_lower"].values, log_y))
        yhat_test_hi = _clip(_unlog(fc_test["yhat_upper"].values, log_y))

        val_m  = _mape(val_df["y"].values,  yhat_val)
        test_m = _mape(test_df["y"].values, yhat_test)

        # --- Top panel ---
        ax1.plot(train_df["ds"], train_df["y"], color=COLOURS["actual"], lw=0.7, alpha=0.55, label="Actual")
        ax1.plot(train_df["ds"], yhat_tr,       color=COLOURS["fitted"], lw=0.9, alpha=0.45, label="Fitted (train)")

        # Val window
        ax1.axvspan(val_df["ds"].iloc[0], val_df["ds"].iloc[-1], alpha=0.07, color=COLOURS["val"])
        ax1.plot(val_df["ds"], val_df["y"], color=COLOURS["actual"], lw=1.0)
        ax1.plot(val_df["ds"], yhat_val, color=COLOURS["val"], lw=2.2,
                 label=f"Val  MAPE {val_m:.1f}%")
        ax1.fill_between(val_df["ds"], yhat_val_lo, yhat_val_hi, alpha=0.18, color=COLOURS["val"])

        # Test window
        ax1.axvspan(test_df["ds"].iloc[0], test_df["ds"].iloc[-1], alpha=0.07, color=COLOURS["test"])
        ax1.plot(test_df["ds"], test_df["y"], color=COLOURS["actual"], lw=1.0)
        ax1.plot(test_df["ds"], yhat_test, color=COLOURS["test"], lw=2.2,
                 label=f"Test MAPE {test_m:.1f}%")
        ax1.fill_between(test_df["ds"], yhat_test_lo, yhat_test_hi, alpha=0.18, color=COLOURS["test"])

        # --- Error panel ---
        err_val  = np.abs(val_df["y"].values  - yhat_val)
        err_test = np.abs(test_df["y"].values - yhat_test)
        ax2.bar(val_df["ds"],  err_val,  color=COLOURS["val"],  alpha=0.75, width=0.8,
                label=f"Val MAE £{err_val.mean():.0f}")
        ax2.bar(test_df["ds"], err_test, color=COLOURS["test"], alpha=0.75, width=0.8,
                label=f"Test MAE £{err_test.mean():.0f}")

    else:
        # Short-history: in-sample fit only
        with open(MODEL_DIR / f"{location}.pkl", "rb") as f:
            saved = pickle.load(f)
        fc = saved["model"].predict(df[["ds"] + regs])
        yhat    = _clip(_unlog(fc["yhat"].values,       log_y))
        yhat_lo = _clip(_unlog(fc["yhat_lower"].values, log_y))
        yhat_hi = _clip(_unlog(fc["yhat_upper"].values, log_y))
        m = _mape(df["y"].values, yhat)

        ax1.plot(df["ds"], df["y"], color=COLOURS["actual"], lw=0.9, alpha=0.7, label="Actual")
        ax1.plot(df["ds"], yhat,    color=COLOURS["fitted"],  lw=2.0,
                 label=f"Fitted  MAPE {m:.1f}%  (in-sample, short history)")
        ax1.fill_between(df["ds"], yhat_lo, yhat_hi, alpha=0.12, color=COLOURS["fitted"])

        err = np.abs(df["y"].values - yhat)
        ax2.bar(df["ds"], err, color=COLOURS["fitted"], alpha=0.7, width=0.8,
                label=f"MAE £{err.mean():.0f}")

    # --- Forward forecast (both paths) ---
    ax1.axvline(df["ds"].max(), color="#aaa", lw=1.2, ls="--", zorder=5)
    ax1.axvspan(fwd["ds"].iloc[0], fwd["ds"].iloc[-1], alpha=0.06, color=COLOURS["forecast"])
    ax1.fill_between(fwd["ds"], fwd["lo"], fwd["hi"], alpha=0.22, color=COLOURS["forecast"])
    ax1.plot(fwd["ds"], fwd["yhat"], color=COLOURS["forecast"], lw=2.4, ls="--",
             label=f"Forecast (+{FORECAST_HORIZON[location]}d)")

    # Axis styling — top
    _fmt_gbp(ax1)
    ax1.set_ylabel("Daily Revenue (£)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax1.grid(True, alpha=0.18)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # Axis styling — error
    _fmt_gbp(ax2)
    ax2.set_ylabel("Abs Error (£)", fontsize=9)
    ax2.set_xlabel("Date", fontsize=9)
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(True, alpha=0.18)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f"{location}_forecast.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for location in LOCATIONS:
        print(f"Plotting {location}...")
        plot_location(location, OUT_DIR)
    print("Done. Plots in plots/")
