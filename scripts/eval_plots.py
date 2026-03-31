"""
Generate predicted vs actual revenue plots for each location.

For locations with enough data (Borough, Covent Garden):
  - Shows full training + holdout period
  - Val window and test window highlighted

For short-history locations:
  - In-sample fit only

Run: python -m scripts.eval_plots
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.training.data_prep import load_training_data, LOCATIONS
from src.training.train import MODEL_CONFIG, _uk_holidays_df, _regressors
from prophet import Prophet

EVAL_WINDOW = 14
MODEL_DIR = Path("models")
OUT_DIR = Path("plots")


def load_model(location: str) -> dict:
    path = MODEL_DIR / f"{location}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_model(location: str) -> Prophet:
    cfg = MODEL_CONFIG.get(location, {})
    cps = cfg.get("changepoint_prior_scale", 0.05)
    model = Prophet(
        holidays=_uk_holidays_df(),
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=cps,
    )
    for reg in _regressors(location):
        model.add_regressor(reg)
    return model


def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100


def plot_location(location: str, out_dir: Path) -> None:
    df = load_training_data(location)
    log_y = MODEL_CONFIG.get(location, {}).get("log_y", False)
    regs = _regressors(location)
    n = len(df)
    can_split = n >= 90

    if can_split:
        test_df  = df.iloc[-EVAL_WINDOW:]
        val_df   = df.iloc[-(EVAL_WINDOW * 2):-EVAL_WINDOW]
        train_df = df.iloc[:-(EVAL_WINDOW * 2)]
        tv_df    = pd.concat([train_df, val_df]).reset_index(drop=True)

        # Val model: fit on train only
        m_val = _build_model(location)
        fit_train = train_df.copy()
        if log_y:
            fit_train["y"] = np.log1p(fit_train["y"])
        m_val.fit(fit_train)
        fc_val = m_val.predict(val_df[["ds"] + regs])
        yhat_val = np.expm1(fc_val["yhat"].values) if log_y else fc_val["yhat"].values
        yhat_val_lo = np.expm1(fc_val["yhat_lower"].values) if log_y else fc_val["yhat_lower"].values
        yhat_val_hi = np.expm1(fc_val["yhat_upper"].values) if log_y else fc_val["yhat_upper"].values

        # Test model: fit on train+val
        m_test = _build_model(location)
        fit_tv = tv_df.copy()
        if log_y:
            fit_tv["y"] = np.log1p(fit_tv["y"])
        m_test.fit(fit_tv)
        fc_test = m_test.predict(test_df[["ds"] + regs])
        yhat_test = np.expm1(fc_test["yhat"].values) if log_y else fc_test["yhat"].values
        yhat_test_lo = np.expm1(fc_test["yhat_lower"].values) if log_y else fc_test["yhat_lower"].values
        yhat_test_hi = np.expm1(fc_test["yhat_upper"].values) if log_y else fc_test["yhat_upper"].values

        # In-sample on train (train model)
        fc_train = m_val.predict(train_df[["ds"] + regs])
        yhat_train = np.expm1(fc_train["yhat"].values) if log_y else fc_train["yhat"].values

        val_mape  = mape(val_df["y"].values,  yhat_val)
        test_mape = mape(test_df["y"].values, yhat_test)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"{location.replace('_', ' ').title()} — Predicted vs Actual Revenue", fontsize=14)

        # --- Top panel: revenue ---
        ax1.plot(train_df["ds"], train_df["y"], color="#444", linewidth=0.8, alpha=0.6, label="Actual (train)")
        ax1.plot(train_df["ds"], yhat_train, color="steelblue", linewidth=1.0, alpha=0.5, label="Fitted (train)")

        ax1.axvspan(val_df["ds"].iloc[0], val_df["ds"].iloc[-1], alpha=0.08, color="orange", zorder=0)
        ax1.plot(val_df["ds"], val_df["y"], color="#444", linewidth=1.2)
        ax1.plot(val_df["ds"], yhat_val, color="darkorange", linewidth=1.8, label=f"Val pred (MAPE {val_mape:.1f}%)")
        ax1.fill_between(val_df["ds"], yhat_val_lo, yhat_val_hi, alpha=0.15, color="darkorange")

        ax1.axvspan(test_df["ds"].iloc[0], test_df["ds"].iloc[-1], alpha=0.08, color="crimson", zorder=0)
        ax1.plot(test_df["ds"], test_df["y"], color="#444", linewidth=1.2)
        ax1.plot(test_df["ds"], yhat_test, color="crimson", linewidth=1.8, label=f"Test pred (MAPE {test_mape:.1f}%)")
        ax1.fill_between(test_df["ds"], yhat_test_lo, yhat_test_hi, alpha=0.15, color="crimson")

        ax1.set_ylabel("Daily Revenue (£)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.2)

        # --- Bottom panel: absolute error ---
        err_val  = np.abs(val_df["y"].values - yhat_val)
        err_test = np.abs(test_df["y"].values - yhat_test)
        ax2.bar(val_df["ds"],  err_val,  color="darkorange", alpha=0.7, width=0.8, label="Val error")
        ax2.bar(test_df["ds"], err_test, color="crimson",    alpha=0.7, width=0.8, label="Test error")
        ax2.set_ylabel("Abs Error (£)")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

    else:
        # In-sample plot for short-history locations
        artifact = load_model(location)
        model = artifact["model"]
        log_y = artifact.get("log_y", False)
        future = df[["ds"] + regs]
        forecast = model.predict(future)
        yhat = np.expm1(forecast["yhat"].values) if log_y else forecast["yhat"].values
        yhat_lo = np.expm1(forecast["yhat_lower"].values) if log_y else forecast["yhat_lower"].values
        yhat_hi = np.expm1(forecast["yhat_upper"].values) if log_y else forecast["yhat_upper"].values
        insample_mape = mape(df["y"].values, yhat)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(
            f"{location.replace('_', ' ').title()} — In-sample Fit (n={len(df)} days, short history)",
            fontsize=14,
        )

        ax1.plot(df["ds"], df["y"], color="#444", linewidth=0.9, label="Actual")
        ax1.plot(df["ds"], yhat, color="steelblue", linewidth=1.8, label=f"Fitted (MAPE {insample_mape:.1f}%)")
        ax1.fill_between(df["ds"], yhat_lo, yhat_hi, alpha=0.15, color="steelblue")
        ax1.set_ylabel("Daily Revenue (£)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)

        err = np.abs(df["y"].values - yhat)
        ax2.bar(df["ds"], err, color="steelblue", alpha=0.7, width=0.8)
        ax2.set_ylabel("Abs Error (£)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = out_dir / f"{location}_eval.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for location in LOCATIONS:
        print(f"Plotting {location}...")
        plot_location(location, OUT_DIR)
    print("Done. Plots saved to plots/")
