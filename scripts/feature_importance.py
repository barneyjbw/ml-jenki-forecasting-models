"""
Permutation feature importance for Borough and Covent Garden.

For each regressor: shuffle its values in the test set, measure MAPE increase.
Higher delta = feature matters more.

Run: python -m scripts.feature_importance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.data_prep import load_training_data
from src.training.train import MODEL_CONFIG, _regressors, _build_model, _uk_holidays_df

EVAL_WINDOW = 14
N_SHUFFLES = 20
OUT_DIR = Path("plots")


def mape(a, p):
    return np.mean(np.abs((a - p) / np.where(a == 0, 1, a))) * 100


def permutation_importance(location: str) -> pd.DataFrame:
    df = load_training_data(location)
    log_y = MODEL_CONFIG.get(location, {}).get("log_y", False)
    regs = _regressors(location)

    test_df  = df.iloc[-EVAL_WINDOW:]
    tv_df    = df.iloc[:-EVAL_WINDOW]

    # Fit model on train+val
    model = _build_model(location)
    fit = tv_df.copy()
    if log_y:
        fit["y"] = np.log1p(fit["y"])
    model.fit(fit)

    # Baseline MAPE on test
    fc_base = model.predict(test_df[["ds"] + regs])
    yhat_base = np.expm1(fc_base["yhat"].values) if log_y else fc_base["yhat"].values
    base_mape = mape(test_df["y"].values, yhat_base)

    # Permutation importance
    rng = np.random.default_rng(42)
    results = []
    for reg in regs:
        deltas = []
        for _ in range(N_SHUFFLES):
            shuffled = test_df.copy()
            shuffled[reg] = rng.permutation(shuffled[reg].values)
            fc = model.predict(shuffled[["ds"] + regs])
            yhat = np.expm1(fc["yhat"].values) if log_y else fc["yhat"].values
            deltas.append(mape(test_df["y"].values, yhat) - base_mape)
        results.append({
            "feature": reg,
            "importance": round(np.mean(deltas), 3),
            "std": round(np.std(deltas), 3),
        })

    return pd.DataFrame(results).sort_values("importance", ascending=False).reset_index(drop=True)


def make_plot():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Importance (Permutation) — Test Window", fontsize=13)

    PRETTY = {
        "apparent_temperature_max": "Apparent Temp Max",
        "precipitation_sum":        "Precipitation Sum",
        "precipitation_hours":      "Precipitation Hours",
        "sunshine_duration":        "Sunshine Duration",
        "wind_speed_10m_max":       "Wind Speed",
        "daylight_duration":        "Daylight Duration",
        "footfall_actual":          "Footfall (actual)",
        "footfall_yoy":             "Footfall (YoY)",
        "temp_sq":                  "Temp² (nonlinear)",
        "rainy_day":                "Rainy Day (binary)",
        "precip_sq":                "Precip Hours²",
    }

    for ax, (location, label) in zip(axes, [
        ("borough",       "Borough"),
        ("covent_garden", "Covent Garden"),
    ]):
        imp = permutation_importance(location)
        imp["label"] = imp["feature"].map(PRETTY).fillna(imp["feature"])

        colors = ["#d73027" if v > 0 else "#4575b4" for v in imp["importance"]]
        bars = ax.barh(imp["label"][::-1], imp["importance"][::-1],
                       xerr=imp["std"][::-1], color=colors[::-1],
                       alpha=0.85, capsize=3, error_kw={"linewidth": 0.8})

        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("MAPE increase when shuffled (pp)", fontsize=9)
        ax.set_title(f"{label}  (baseline MAPE shown in subtitle)", fontsize=10)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.25)

        # Annotate each bar with the value
        for i, (v, e) in enumerate(zip(imp["importance"][::-1], imp["std"][::-1])):
            ax.text(v + e + 0.05, i, f"{v:+.2f}pp", va="center", fontsize=7.5, color="#333")

    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    for loc in ["borough", "covent_garden"]:
        print(f"\n{loc.upper()} — permutation importance:")
        imp = permutation_importance(loc)
        imp["label"] = imp["feature"].map({
            "apparent_temperature_max": "Apparent Temp Max",
            "precipitation_sum":        "Precipitation Sum",
            "precipitation_hours":      "Precipitation Hours",
            "sunshine_duration":        "Sunshine Duration",
            "wind_speed_10m_max":       "Wind Speed",
            "daylight_duration":        "Daylight Duration",
            "footfall_actual":          "Footfall (actual)",
            "footfall_yoy":             "Footfall (YoY)",
            "temp_sq":                  "Temp² (nonlinear)",
            "rainy_day":                "Rainy Day (binary)",
            "precip_sq":                "Precip Hours²",
        }).fillna(imp["feature"])
        print(imp[["label", "importance", "std"]].to_string(index=False))

    make_plot()
