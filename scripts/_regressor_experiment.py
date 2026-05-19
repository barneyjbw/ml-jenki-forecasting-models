"""
Test tightened regressor priors + reduced regressor set on Battersea + Spitalfields.

For each location, fits two variants:
  A) Current: all 10 regressors, default prior_scale (10)
  B) Proposed: 5 regressors, prior_scale=1.0

Reports:
  - Test MAPE (out-of-sample on last 7 days)
  - 28-day forecast: trend, regressor offset, yhat range, max/min ratio
  - Component magnitudes (looking for the "huge cancelling components" pathology)
"""
from __future__ import annotations
import os
os.environ.setdefault("DATA_SOURCE", "gcs")
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from datetime import date, timedelta
from prophet import Prophet
import holidays
from dateutil.easter import easter

from src.training.data_prep import load_training_data
from src.training.london_calendar import get_london_events_df


PROPOSED_REGRESSORS = [
    "apparent_temperature_max",
    "precipitation_sum",
    "footfall_actual",
    "footfall_yoy",
    "network_momentum",
]

CURRENT_REGRESSORS = [
    "apparent_temperature_max", "precipitation_sum", "precipitation_hours",
    "sunshine_duration", "wind_speed_10m_max", "daylight_duration",
    "footfall_actual", "footfall_yoy", "temp_sq",
    "network_momentum",
]


def _holidays_df(location: str) -> pd.DataFrame:
    years = range(2024, 2030)
    uk = holidays.country_holidays("GB", subdiv="ENG", years=years)
    rows = [{"ds": str(d), "holiday": name} for d, name in uk.items()]
    for year in years:
        rows.append({"ds": str(easter(year) - timedelta(weeks=3)), "holiday": "Mothering Sunday"})
    return pd.concat([pd.DataFrame(rows), get_london_events_df(years, location=location)], ignore_index=True).drop_duplicates(subset=["ds", "holiday"])


def build_model(location: str, regressors: list[str], prior_scale: float, cps: float) -> Prophet:
    m = Prophet(
        holidays=_holidays_df(location),
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=cps,
        uncertainty_samples=0,
    )
    for reg in regressors:
        m.add_regressor(reg, prior_scale=prior_scale)
    return m


def evaluate(location: str, df: pd.DataFrame, regressors: list[str], prior_scale: float, cps: float, label: str):
    ew = 7
    test_df = df.iloc[-ew:].copy()
    train_df = df.iloc[:-ew].copy()

    m = build_model(location, regressors, prior_scale, cps)
    m.fit(train_df)

    # In-distribution test MAPE (last 7 training days, OOS)
    fc_test = m.predict(test_df[["ds"] + regressors])
    yhat_test = np.maximum(fc_test["yhat"].values, 0)
    actual = test_df["y"].values
    test_mape = np.mean(np.abs((actual - yhat_test) / np.where(actual == 0, 1, actual))) * 100

    # Forecast next 28 days for shape inspection — use the LAST 28 days of training data
    # to provide regressor values (simulating recent conditions)
    horizon_df = df.iloc[-28:][["ds"] + regressors].copy()
    horizon_df["ds"] = pd.to_datetime(horizon_df["ds"]) + pd.Timedelta(days=28)
    fc_h = m.predict(horizon_df)
    yhat_h = np.maximum(fc_h["yhat"].values, 0)
    trend_h = fc_h["trend"].values
    extra_h = fc_h.get("extra_regressors_additive", pd.Series([0] * len(fc_h))).values

    print(f"  [{label}] regressors={len(regressors)}, prior_scale={prior_scale}, cps={cps}")
    print(f"    test MAPE (last 7 days):  {test_mape:.2f}%")
    print(f"    forecast trend range:     {trend_h.min():+,.0f}  ->  {trend_h.max():+,.0f}")
    print(f"    forecast reg-offset range: {extra_h.min():+,.0f}  ->  {extra_h.max():+,.0f}")
    print(f"    forecast yhat range:       {yhat_h.min():,.0f}  ->  {yhat_h.max():,.0f}")
    print(f"    yhat max/min ratio:        {yhat_h.max() / max(yhat_h.min(), 1):.1f}x")
    print(f"    sample yhat (28 days):     {[int(v) for v in yhat_h[:14]]}")
    print()
    return {"test_mape": test_mape, "yhat_max": yhat_h.max(), "yhat_min": yhat_h.min(),
            "trend_range": (trend_h.min(), trend_h.max()),
            "extra_range": (extra_h.min(), extra_h.max())}


def main():
    for location in ["battersea", "spitalfields"]:
        print(f"=== {location.upper()} ===")
        df = load_training_data(location)
        print(f"  training days: {len(df)}  ({df['ds'].min().date()} -> {df['ds'].max().date()})")
        recent = df.tail(7)["y"].values
        print(f"  recent 7d actuals: {[int(v) for v in recent]}  median={int(np.median(recent))}")
        print()

        a = evaluate(location, df, CURRENT_REGRESSORS,  prior_scale=10.0, cps=0.1,  label="A: current (10 regs, loose prior)")
        d = evaluate(location, df, CURRENT_REGRESSORS,  prior_scale=1.0,  cps=0.1,  label="D: all 10 regs, tight prior only")
        e = evaluate(location, df, CURRENT_REGRESSORS,  prior_scale=1.0,  cps=0.05, label="E: all 10 regs, tight prior + stiff trend")
        b = evaluate(location, df, PROPOSED_REGRESSORS, prior_scale=1.0,  cps=0.05, label="B: 5 regs + tight prior + stiff trend")

        print()


if __name__ == "__main__":
    main()
