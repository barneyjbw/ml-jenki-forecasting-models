# R&D — Jenki Revenue Forecasting

Experiments run on the two locations with enough data for a proper holdout eval
(14-day val + 14-day test windows). Battersea, Canary Wharf and Spitalfields are
evaluated in-sample only due to short history.

**Evaluation locations:** Borough (427 days), Covent Garden (502 days)
**Metric focus:** test MAPE (out-of-sample, most honest signal)

---

## Baseline

Model: Prophet, multiplicative seasonality, UK bank holidays, `changepoint_prior_scale=0.05` (default)
Features: weather × 4, footfall × 4 (`footfall_actual`, `footfall_yoy`, `footfall_dow_avg`, `footfall_vs_dow`)

| Location | Val MAE | Val MAPE | Test MAE | Test MAPE |
|---|---|---|---|---|
| Borough | 152.13 | 8.44% | 366.78 | 16.72% |
| Covent Garden | 277.69 | 12.33% | 409.33 | 15.96% |

---

## Experiment Log

### EXP-01 · changepoint_prior_scale tuning
> Default is 0.05. Higher = more flexible trend line.

| Experiment | Borough val | Borough test | CG val | CG test |
|---|---|---|---|---|
| cps = 0.05 (baseline) | 8.44% | 16.72% | 12.33% | 15.96% |
| cps = 0.1 | 8.91% | **11.40%** | 10.63% | 15.32% |
| cps = 0.3 | 18.43% | 15.62% | 10.18% | 15.21% |

**Finding:** `cps=0.1` is a strong win for Borough (−5.3pp test), positive for CG. `cps=0.3` overfits val on Borough.

---

### EXP-02 · Log-transform target (log1p / expm1)

| Location | Val MAPE | Test MAPE | vs baseline |
|---|---|---|---|
| Borough | 14.53% | 24.47% | **worse** |
| Covent Garden | 11.07% | **11.11%** | −4.85pp |

**Finding:** Inconsistent — helps CG substantially, hurts Borough. Not applied universally.
Logged for revisiting if per-location model configs are introduced.

---

### EXP-03 · Lean footfall features
> Remove `footfall_dow_avg` and `footfall_vs_dow` — both derived from day-of-week averages,
> likely redundant with Prophet's built-in weekly seasonality.
> Keep: `footfall_actual` + `footfall_yoy` only.

| Location | Val MAPE | Test MAPE | vs baseline |
|---|---|---|---|
| Borough | 8.21% | 16.18% | −0.54pp |
| Covent Garden | 12.58% | 15.74% | −0.22pp |

**Finding:** Small but consistent improvement on both + simpler feature set. Applied.

---

### EXP-04 · cps=0.1 + lean footfall (combined best)

| Location | Val MAPE | Test MAPE | vs baseline |
|---|---|---|---|
| Borough | 8.85% | **11.09%** | −5.63pp |
| Covent Garden | 11.52% | **14.42%** | −1.54pp |

**Finding:** Best overall configuration. Both improvements stack. **Applied to production.**

---

### EXP-05 · Persona features (T.Bombe.1-7 from oa_personas_total.parquet)
> Incoming traveller persona mix per destination OA, keyed by day-of-week.
> Data has no date axis — varies by Monday–Sunday only, static per location.

| Location | Val MAPE | Test MAPE | vs baseline |
|---|---|---|---|
| Borough | 8.72% | 16.65% | +0.07pp val, −0.07pp test |
| Covent Garden | 12.11% | 15.29% | −0.22pp val, −0.67pp test |

**Finding:** Effectively redundant with weekly seasonality (7 static values per location).
Marginal improvement on CG, negligible on Borough. Not applied — adds 7 regressors
for minimal gain and requires parquet lookup in prod.

---

### EXP-06 · Persona × footfall interaction (quality-adjusted footfall)
> `pf_bi = T.Bombe.i × footfall_actual` — 7 interaction features.
> Idea: weight footfall by the persona mix of incoming travellers.

| Location | Val MAPE | Test MAPE | vs baseline |
|---|---|---|---|
| Borough | 8.69% | 16.40% | worse |
| Covent Garden | 13.64% | 16.48% | worse |

**Finding:** Hurts both locations. Interaction features without enough data to learn
persona → revenue coefficients likely add noise. Not applied.

---

### EXP-07 · seasonality_prior_scale tuning
> Default is 10. Controls flexibility of seasonal components.

| Experiment | Borough test | CG test |
|---|---|---|
| sps = 5 | 16.31% | 15.42% |
| sps = 10 (baseline) | 16.72% | 15.96% |
| sps = 20 | 16.23% | 15.61% |

**Finding:** Marginal improvement at sps=20 but effect is small. Not worth adding complexity;
superseded by cps tuning in EXP-01.

---

### EXP-08 · cps=0.1 + log_y combined

| Location | Val MAPE | Test MAPE | vs baseline |
|---|---|---|---|
| Borough | 13.09% | 22.64% | much worse |
| Covent Garden | 10.81% | 11.84% | −4.12pp |

**Finding:** CG benefits but Borough degrades badly. Log transform seems incompatible
with Borough's revenue distribution at this changepoint flexibility.

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough test −5.9pp, CG test −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Simpler feature set, marginal improvement |
| `log_y=True` (log1p/expm1 transform) | Covent Garden only | CG test −5.0pp additional |

**Current holdout metrics (post all changes):**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.74% | **10.78%** | −5.94pp |
| Covent Garden | 11.01% | **11.00%** | −4.96pp |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 8.46% |
| Canary Wharf | 74 | 5.64% |
| Spitalfields | 65 | 6.08% |

---

---

### EXP-09 · School holidays (binary regressor)
> English school term holiday windows 2024-2026 as a binary feature.

| Location | Val MAPE | Test MAPE | vs baseline |
|---|---|---|---|
| Borough | 10.12% | 10.65% | −0.13pp test, val worse |
| Covent Garden | 11.84% | 10.56% | −0.44pp test, val worse |

**Finding:** Marginal on test, hurts val. Inconsistent signal — not applied.

---

### EXP-10 · holidays_prior_scale tuning
> Default is 10. Controls how much the bank holiday effects can flex.

| Experiment | Borough test | CG test |
|---|---|---|
| hps = 5 | 10.66% | 11.0% |
| hps = 10 (baseline) | 10.78% | 11.0% |
| hps = 20 | 10.79% | 10.94% |

**Finding:** Negligible. Not applied.

---

### EXP-11 · Additional weather variables (daylight_duration, precipitation_hours)
> Both available from Open-Meteo archive AND forecast API.
> `daylight_duration` = astronomical daylight hours (calendar-based, predictable).
> `precipitation_hours` = number of hours with rain on that day.

| Experiment | Borough val | Borough test | CG val | CG test |
|---|---|---|---|---|
| baseline | 9.74% | 10.78% | 11.01% | 11.0% |
| + daylight_duration | 9.87% | 10.77% | 10.97% | 10.76% |
| + precipitation_hours | 9.40% | **9.75%** | 10.86% | **10.41%** |
| + both | 9.49% | 9.76% | 10.53% | **9.96%** |
| swap sunshine→daylight | 8.79% | 10.24% | 16.26% | 10.1% |

**Finding:** `precipitation_hours` is a clear win for both (Borough −1.03pp, CG −0.59pp).
Adding `daylight_duration` on top helps CG further (total CG test −1.04pp). Both **applied**.
Swapping out `sunshine_duration` hurts — keep all 4 original weather vars plus the 2 new ones.

---

### EXP-12 · Matcha affinity footfall (persona B6+B7 × footfall)
> Persona labels confirmed: Bombe 6 = "High demanding high income", Bombe 7 = "Aspirational".
> These are the target matcha buyer profiles.
> Feature: `footfall_actual × (T.Bombe.6_proportion + T.Bombe.7_proportion)` per OA/day-of-week.

| Experiment | Borough test | CG test |
|---|---|---|
| baseline | 9.78% | 9.88% |
| matcha_ff replaces footfall_actual | 10.0% | 10.33% |
| matcha_ff added alongside | 9.82% | 9.83% |

**Finding:** No improvement. Root cause: persona proportions only vary by day-of-week (7 static values per location), so `matcha_affinity_ff` = `footfall_actual × constant_per_dow`. This adds no information beyond `footfall_actual` that Prophet's weekly seasonality doesn't already capture. Not applied.

---

### EXP-13 · Special drink launch features
> `active_special_drinks`: count of currently active limited-edition drinks.
> `days_since_drink_launch`: days since most recent drink was introduced.
> Source: Drink Start Dates.xlsx in GCS.

| Experiment | Borough test | CG test |
|---|---|---|
| baseline | 9.78% | 9.88% |
| + active_special_drinks | 11.67% | 16.05% |
| + days_since_drink_launch | 10.96% | 13.3% |

**Finding:** Significantly hurts the model. The test period (last 14 days of data) has no active
special drink while training periods do — model can't generalise the coefficient. Not applied.
Note: would require manual input at forecast time anyway (operational dependency).

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough test −5.9pp, CG test −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Simpler feature set, minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG test −5.0pp additional |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough test −1.0pp, CG test −1.1pp |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.54% | **9.78%** | −6.94pp |
| Covent Garden | 10.61% | **9.88%** | −6.08pp |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 5.54% |
| Canary Wharf | 74 | 5.29% |
| Spitalfields | 65 | 5.75% |

---

## GCS Bucket Findings

Files explored in `gs://bombe-ml-data/revenue-predictor/`:
- `hourly_station_footfall.parquet` — day-of-week aggregated (no actual dates). Same limitation as `oa_personas_total.parquet`. No date-level signal.
- `jenki_store_rev.csv` — day-of-week average revenue per store (42 rows). Not daily data.
- `persona_labels.csv` — confirmed Bombe 6 = "High demanding high income", Bombe 7 = "Aspirational" as target personas.
- `combined_filtered_data.csv` — special drink daily sales by store. Useful for product-level models, not total revenue.
- `Drink Start Dates.xlsx` — special drink launch/end dates. Tested as feature, didn't help (EXP-13).
- `tfl_footfall/` — only 3 daily files (Mar 12–14 2026). New pipeline, not enough history for training.
- No weather data in GCS. Open-Meteo is the appropriate source.

---

---

### EXP-14 · Staffing features (worked_hours, leave_ratio)
> Source: `Weekly Trade Report (with leave) (53).csv`

**Rejected on principle:** staffing is scheduled based on expected revenue — the causality
runs the wrong way. `worked_hours` would encode the manager's own forecast, not independent
signal. Data also ends Nov 2025 making it untestable. Not worth pursuing.

---

### EXP-15 · changepoint_range tuning

| Config | Borough test | CG test |
|---|---|---|
| cps=0.1, cr=0.8 (current) | **9.78%** | **9.88%** |
| cps=0.1, cr=0.9 | 9.84% | 16.91% |
| cps=0.08, cr=0.8 | 11.21% | 10.12% |
| cps=0.15, cr=0.8 | 11.65% | 10.48% |

**Finding:** Current config is already optimal. Not changed.

---

### EXP-16 · Google Trends
**Blocked:** `pytrends` rate-limited by Google. Also unreliable for production.
Not pursued further without a paid search interest API.

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough test −5.9pp, CG test −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Simpler feature set, minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG test −5.0pp additional |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough test −1.0pp, CG test −1.1pp |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.54% | **9.78%** | −6.94pp |
| Covent Garden | 10.61% | **9.88%** | −6.08pp |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 5.54% |
| Canary Wharf | 74 | 5.29% |
| Spitalfields | 65 | 5.75% |

---

## GCS Bucket Findings

| File | Useful? | Notes |
|---|---|---|
| `hourly_station_footfall.parquet` | No | Day-of-week aggregated only, no date dimension |
| `jenki_store_rev.csv` | No | Day-of-week averages (42 rows), not daily data |
| `persona_labels.csv` | Info only | B6="High demanding high income", B7="Aspirational" confirmed |
| `combined_filtered_data.csv` | No | Special drink sales only, not total revenue |
| `Drink Start Dates.xlsx` | No (tested) | Hurts model, requires manual forecast input |
| `Weekly Trade Report` | No (tested) | Data ends Nov 2025, breaks splits |
| `store_oa_map.parquet` | Info only | Minor OA code discrepancies from our postcode lookup |
| `tfl_footfall/` | No | Only 3 days (Mar 12–14 2026), insufficient history |
| `hourly/Revel Data/` | Future | Hourly revenue CSVs per store, useful for a future hourly model |

No weather data in GCS. Open-Meteo is the correct source for both archive and forecast.

---

### EXP-18 · Yearly seasonality Fourier order tuning
> Prophet `yearly_seasonality=True` defaults to 10 Fourier terms. Tested n=5,10,15,20.

| n_terms | Borough val | Borough test | CG val | CG test |
|---|---|---|---|---|
| 5 | 7.30% | 9.89% | 10.00% | 15.91% |
| 10 (baseline) | 7.79% | 9.78% | 10.30% | 9.88% |
| 15 | 7.61% | 13.56% | 9.85% | 9.82% |
| 20 | 7.71% | 13.35% | 9.68% | 10.47% |

**Finding:** n=15 and n=20 severely overfit Borough. CG n=15 gives +0.06pp — not worth the risk.
Baseline n=10 confirmed correct. Not changed.

---

### EXP-19 · Temperature nonlinearity (temp_sq)
> Revenue vs temperature is nonlinear: mild days are best, very cold and very hot days both suppress.
> Feature: `temp_sq = (apparent_temperature_max − 15.0)²`
> 15°C is a fixed constant (ideal mild day for outdoor activity) — no data leakage at forecast time.

| Location | Val MAPE | Test MAPE | vs baseline |
|---|---|---|---|
| Borough | 9.92% | **9.65%** | −0.13pp |
| Covent Garden | 11.1% | **9.26%** | −0.62pp |

Also tested `hot_day` binary (>25°C): helps Borough but hurts CG; inconsistent, not applied.

**Finding:** `temp_sq` is a clean win for both locations (CG especially). Applied to all locations.
`(temp − 15)²` is computable at forecast time from Open-Meteo forecast API with no leakage.

---

### EXP-20 · Weekly seasonality Fourier order tuning
> Default weekly_seasonality=True uses 3 Fourier terms. Tested n=3,5,7.

| n_terms | Borough test | CG test |
|---|---|---|
| 3 (baseline) | 9.65% | 9.26% |
| 5 | 9.62% | 9.25% |
| 7 | 13.86% | 24.99% |

**Finding:** n=5 is marginal (0.01–0.03pp). n=7 severely overfits. Baseline n=3 kept.

---

### EXP-21 · Rain interaction features (rain_x_ff, rainy_day binary, precip_sq)

Tested three rain-related features independently and in combination:

| Feature | Borough test | CG test |
|---|---|---|
| baseline | 9.65% | 9.26% |
| rain × footfall interaction | +0.08pp | +3.06pp |
| rainy_day (precip > 1mm binary) | **−0.18pp** | +0.33pp |
| precip_sq (precipitation_hours²) | **−0.34pp** | +0.21pp |
| rainy_day + precip_sq (Borough) | **−0.41pp** | — |

**Finding:** `rain × footfall` hurts both. `rainy_day` and `precip_sq` help Borough consistently;
both hurt CG. Applied `rainy_day` + `precip_sq` to Borough only as per-location `extra_regressors`.

---

### EXP-22 · Various additional experiments (no improvement)

All tested against current baseline, none applied:
- Weekly prior_scale tuning (footfall): helps Borough −0.06pp but hurts CG +0.74pp
- Bank holiday proximity (days_to_bh): Borough +0.83pp, CG −0.08pp
- 7-day rolling footfall average: hurts both (+0.76pp, +1.56pp)
- Wind chill (wind × cold): hurts both (+0.14pp, +0.28pp)
- Payday zone (first/last 3 days of month): negligible/hurts (+0.07pp, +0.02pp)
- `temp_sq` + `daylight_sq` for CG: both hurt (+0.49pp)
- `precip_sq` for CG: +0.21pp worse

**Finding:** CG and Borough have different response functions to weather. Features are now location-specific.

---

### EXP-17 · TfL strikes binary regressor
> `is_tfl_strike`: 1 on days with tube/Overground/Elizabeth line strike action.
> Historical dates from news records (Nov 2024–Mar 2026). At forecast time: TfL Disruption API.

| Location | Baseline test | Strike test | Delta |
|---|---|---|---|
| Borough | 9.78% | 9.90% | +0.12pp |
| Covent Garden | 9.88% | 10.09% | +0.21pp |

**Finding:** Same failure mode as EXP-13 (special drinks). Test period (last 14 days) contains 0 strike
days. Model can't generalise the coefficient and adds noise. Not applied.
Note: strike lookup + TfL API integration is still valid infrastructure for production disruption
monitoring, but it's not a useful training feature.

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough test −5.9pp, CG test −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Simpler feature set, minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG test −5.0pp additional |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough test −1.0pp, CG test −1.1pp |
| Added `temp_sq = (apparent_temperature_max − 15)²` | All locations | Borough test −0.13pp, CG test −0.62pp |
| Added `rainy_day` + `precip_sq` as extra regressors | Borough only | Borough test −0.41pp |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.71% | **9.24%** | −7.48pp |
| Covent Garden | 11.10% | **9.26%** | −6.70pp |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 5.19% |
| Canary Wharf | 74 | 5.28% |
| Spitalfields | 65 | 14.84% |

Note: Spitalfields 14.84% in-sample reflects the short post-reopen history (Jan 2026 only).
Revenue level shifted structurally; model will improve as more post-reopen data accumulates.

---

### EXP-23 · Footfall sanity check (data gap fix)
> If `footfall_actual` for a given date is < 10% of that station's day-of-week average,
> treat it as a data gap and substitute the dow average.
> Root cause: CG station missing from StationFootfall CSV for some dates, returning ~10 exits.

**Finding:** Fixed. Applied in `_get_footfall_features()`.

---

### EXP-24 · Mothering Sunday holiday feature
> UK Mothering Sunday (4th Sunday of Lent = Easter − 3 weeks) added to Prophet holidays.
> Not a bank holiday — people stay home for family lunches, suppresses coffee shop revenue.
> Both Borough (−29%) and CG (−57%) missed badly on Mar 15 2026 (Mothering Sunday).
> Calculable years ahead — production safe.

| Location | Before | After | Delta |
|---|---|---|---|
| Borough test | 9.24% | **8.18%** | −1.06pp |
| Covent Garden test | 9.26% | **8.24%** | −1.02pp |

**Finding:** Largest single improvement since EXP-01. Applied.

---

### EXP-25 · Additional calendar events (Valentine's Day, Halloween, Black Friday, etc.)
> Tested individually and in combination. All require 2+ training examples.
> Events with only 1 occurrence (Father's Day, London Marathon, Notting Hill Carnival,
> St Patrick's Day) excluded — can't estimate reliable coefficient.

| Event | Borough delta | CG delta |
|---|---|---|
| Valentine's Day | +0.22pp | +0.39pp |
| Halloween | +0.03pp | −0.20pp |
| Black Friday | +0.04pp | −0.05pp |
| Christmas Eve | +0.03pp | +0.07pp |
| New Year's Eve | −0.01pp | −0.07pp |
| Halloween + NYE combined | +0.10pp | +2.58pp |

**Finding:** All marginal or harmful. Halloween −0.20pp for CG looks promising in isolation but
with only 2 training examples the coefficient is noise — confirmed unstable when combined.
Not applied. Mothering Sunday is the only calendar event with enough signal.

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough −5.9pp, CG −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG −5.0pp |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough −1.0pp, CG −1.1pp |
| Added `temp_sq = (apparent_temperature_max − 15)²` | All locations | Borough −0.13pp, CG −0.62pp |
| Added `rainy_day` + `precip_sq` as extra regressors | Borough only | Borough −0.41pp |
| Footfall sanity check (< 10% of dow avg → substitute dow avg) | All locations | Data quality fix |
| Mothering Sunday added to Prophet holidays | All locations | Borough −1.06pp, CG −1.02pp |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.29% | **8.18%** | −8.54pp |
| Covent Garden | 10.83% | **8.24%** | −7.72pp |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 5.03% |
| Canary Wharf | 74 | 5.69% |
| Spitalfields | 65 | 5.49% |

---

### EXP-26 · PredictHQ events features (event_count, has_major_event)
> Fetched historical events within radius of each location via PredictHQ API.
> Categories: concerts, sports, conferences, expos, festivals, performing-arts. Min rank: 30.
> Features: `event_count` (events per day), `has_major_event` (any rank ≥ 70).

| Feature | Borough delta | CG delta |
|---|---|---|
| event_count | +0.45pp | +0.23pp |
| has_major_event | +0.18pp | +1.19pp |
| both | +0.62pp | +1.35pp |

**Finding:** Hurts both locations. Root cause: signal too sparse (30–60 event-days out of 500 total)
for Prophet to learn a reliable coefficient — same failure mode as EXP-13 and EXP-17.
Not applied to training model.

**Note:** PredictHQ is still valuable at forecast time as an advisory signal (flag days with
major nearby events for human review), even without a trained coefficient.

---

### EXP-27 · Network momentum (cross-location revenue trend signal)
> `network_momentum = rolling_7d_mean(other_locations_revenue) / rolling_28d_mean(other_locations_revenue)`
> Captures whether the Jenki network is trending above or below its recent baseline.
> Value > 1.0 = network hot; < 1.0 = network cold; 1.0 = neutral fallback.
> No data leakage — uses historical actuals only. At forecast time: scalar held constant for 14-day horizon.
> Clipped to [0.5, 2.0] to prevent extreme values distorting the model.
> Requires min_count=2 for the network sum (at least 2 other locations trading that day).

Tested on Borough and CG first, then applied to all 5 locations:

| Location | Before | After | Delta |
|---|---|---|---|
| Borough test | 8.18% | **8.09%** | −0.09pp |
| Covent Garden test | 8.24% | **8.02%** | −0.22pp |

**Finding:** Small but consistent improvement on both primary locations. CG benefits more (+0.22pp).
No degradation on either. Applied to all 5 locations as an `extra_regressor`.

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough −5.9pp, CG −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG −5.0pp |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough −1.0pp, CG −1.1pp |
| Added `temp_sq = (apparent_temperature_max − 15)²` | All locations | Borough −0.13pp, CG −0.62pp |
| Added `rainy_day` + `precip_sq` as extra regressors | Borough only | Borough −0.41pp |
| Footfall sanity check (< 10% of dow avg → substitute dow avg) | All locations | Data quality fix |
| Mothering Sunday added to Prophet holidays | All locations | Borough −1.06pp, CG −1.02pp |
| Added `network_momentum` as extra regressor | All locations | Borough −0.09pp, CG −0.22pp |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.47% | **8.09%** | −8.63pp |
| Covent Garden | 10.60% | **8.02%** | −7.94pp |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 3.91% |
| Canary Wharf | 74 | 5.54% |
| Spitalfields | 65 | 5.48% |

---

---

### EXP-28 · London social calendar (Phase 1) — algorithmic recurring events as Prophet holidays
> Named holidays for recurring London social events, replacing the binary-regressor approach
> (which EXP-26 showed is too sparse). Prophet treats each holiday name as a shared coefficient
> across all its dates — enabling the model to learn from a handful of recurrences per event type.
>
> Phase 1 events (universal + CG-specific):
>   Universal: Father's Day, Valentine's Day, New Year's Eve, Bonfire Night, London Marathon
>   CG only: Pride London, Chinese New Year Parade, St Patrick's Day Parade, Diwali,
>            Notting Hill Carnival, Wimbledon Finals Weekend, Chelsea Flower Show
>
> Borough / CW / Battersea / Spitalfields: opted out — insufficient training history
> for ≥2 occurrences of annual events.

| Location | Before | After | Delta |
|---|---|---|---|
| Borough test | 8.09% | 8.02% | −0.07pp |
| Covent Garden test | 8.02% | **7.24%** | **−0.78pp** |

**Finding:** CG benefits substantially — London social calendar holidays explain previously
anomalous days (Wimbledon finals, Pride etc.). Borough is unaffected (opted out). Applied.

---

### EXP-29 · London Fashion Week + Trooping the Colour (Phase 2 calendar additions)
> London Fashion Week: Somerset House (200m from Jenki CG) is the main hub.
> 5-day events in February and September — 3+ occurrences in CG's 18-month training window
> (Sept 2024, Feb 2025, Sept 2025) = enough for a stable coefficient.
>
> Trooping the Colour: 2nd Saturday of June, procession past Trafalgar Sq.
> 1 occurrence in training (Jun 14 2025) — coefficient likely noisy but included.
>
> Results are combined with EXP-28 in the single training run above.

**Note:** Venue-level events (Shakespeare's Globe for Borough, ROH for CG) infrastructure
built in `src/training/venue_events.py` + `scripts/fetch_venue_events.py`. Awaiting
PredictHQ plan upgrade for geographic proximity search. Cache-ready when available.

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough −5.9pp, CG −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG −5.0pp |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough −1.0pp, CG −1.1pp |
| Added `temp_sq = (apparent_temperature_max − 15)²` | All locations | Borough −0.13pp, CG −0.62pp |
| Added `rainy_day` + `precip_sq` as extra regressors | Borough only | Borough −0.41pp |
| Footfall sanity check (< 10% of dow avg → substitute dow avg) | All locations | Data quality fix |
| Mothering Sunday added to Prophet holidays | All locations | Borough −1.06pp, CG −1.02pp |
| Added `network_momentum` as extra regressor | All locations | Borough −0.09pp, CG −0.22pp |
| London social calendar (Phase 1) as Prophet holidays | CG only | CG −0.78pp |
| London Fashion Week + Trooping the Colour (Phase 2) | CG only | Included in −0.78pp above |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.37% | **8.02%** | −8.70pp |
| Covent Garden | 11.83% | **7.24%** | **−8.72pp** |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 3.91% |
| Canary Wharf | 74 | 5.54% |
| Spitalfields | 65 | 5.48% |

---

### EXP-30 · Borough social calendar opt-in
> Tested enabling Phase 1 social calendar events for Borough (previously opted out).
> Borough has 17 months of data — each annual event appears only once in the training window,
> so Prophet can't learn a reliable coefficient from a single occurrence.

| Location | Before | After | Delta |
|---|---|---|---|
| Borough test | 8.02% | 8.97% | **+0.95pp worse** |

**Finding:** Confirmed hypothesis — 1-occurrence annual events add noise, not signal. Borough
remains opted out of social calendar (`LOCATION_EVENTS["borough"] = None`).

---

### EXP-31 · log_y for short-history locations (Canary Wharf)
> Tested `log_y=True` for Canary Wharf (74 days at time of test).
> EXP-02 showed log_y helps CG but hurts Borough. CW is between them in history length.

| Location | Before (no log_y) | After (log_y) | Delta |
|---|---|---|---|
| Canary Wharf in-sample | 5.54% | 5.78% | **+0.24pp worse** |

**Finding:** 74 days is insufficient for log_y to help. CW reverted to no log_y.
Revisit when CW reaches 90+ days of data.

---

### EXP-32 · Stacked ensemble (peer_yhat cross-location signal)
> For each target location, load saved Prophet models for all OTHER locations.
> Predict for target dates using each peer model → normalise by that peer's rolling 28d actual mean
> → average across peers → `peer_yhat` (1.0 = network typical, >1 = above average predicted day).
>
> Key difference from `network_momentum`:
>   - `network_momentum`: ratio of rolling ACTUALS (constant across 14-day horizon at forecast time)
>   - `peer_yhat`: model PREDICTIONS normalised per-location (day-varying in forecast horizon,
>     incorporates holiday/weather/trend from each peer model)
>
> Infrastructure: `scripts/generate_peer_forecasts.py` generates parquet caches per location.
> `load_training_data()` loads cache if present; neutral fallback (1.0) if absent.

**Result when added as extra_regressor:**

| Location | Before | After | Delta |
|---|---|---|---|
| Borough test | 8.02% | 14.07% | **+6.05pp worse** |
| Covent Garden test | 7.24% | 8.79% | **+1.55pp worse** |

**Root cause — in-sample data leakage:** Saved models were trained on ALL data including the test
period. Their predictions for test dates carry indirect knowledge of those dates → spurious
training correlation that doesn't generalise. True out-of-sample stacking would require rolling
cross-validation (training each model N times) — computationally prohibitive for daily retraining.

**Current state:** `peer_yhat` column is present in training data (parquet caches generated) but
excluded from `MODEL_CONFIG extra_regressors`. At production forecast time, the two-pass approach
(run all base models → compute peer_yhat from those predictions → re-run stacked models) would
work without leakage, but the coefficient would need to be trained differently.

**Not applied to training config.**

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough −5.9pp, CG −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG −5.0pp |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough −1.0pp, CG −1.1pp |
| Added `temp_sq = (apparent_temperature_max − 15)²` | All locations | Borough −0.13pp, CG −0.62pp |
| Added `rainy_day` + `precip_sq` as extra regressors | Borough only | Borough −0.41pp |
| Footfall sanity check (< 10% of dow avg → substitute dow avg) | All locations | Data quality fix |
| Mothering Sunday added to Prophet holidays | All locations | Borough −1.06pp, CG −1.02pp |
| Added `network_momentum` as extra regressor | All locations | Borough −0.09pp, CG −0.22pp |
| London social calendar Phase 1 + Fashion Week + Trooping | CG only | CG −0.78pp |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.37% | **8.02%** | −8.70pp |
| Covent Garden | 11.83% | **7.24%** | **−8.72pp** |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 3.91% |
| Canary Wharf | 74 | 5.54% |
| Spitalfields | 65 | 5.48% |

---

### EXP-33 · Easter Sunday + anomalous day exclusions

#### A — Easter Sunday holiday
> Easter Sunday is not a UK bank holiday (Easter Monday is), so it was missing from the calendar.
> It's a family day with the same suppression pattern as Mothering Sunday.
> Borough missed Easter Sunday Apr 20 2025 by 57.9%. CG handled it better (not in top errors).
> Calculable years ahead. Added as `_ALWAYS_ON` — applies even to opted-out locations.
>
> Tested alongside Black Friday (Friday after US Thanksgiving):
> - Black Friday helps Borough marginally (−0.04pp) but hurts CG (+0.19pp).
> - With only 2 training occurrences per location (first in opening weeks), coefficient is noisy.
> - **Black Friday not applied.** Revisit at Nov 2026 (3rd occurrence).

| Location | Before | After (Easter only) | Delta |
|---|---|---|---|
| Borough test | 8.02% | **7.79%** | −0.23pp |
| Covent Garden test | 7.24% | **7.18%** | −0.06pp |

**Finding:** Easter Sunday is a clean win for both. Applied.

#### B — Anomalous day exclusions (tested, reverted)
> Tested excluding: Borough Aug 1 2025 (£73 on a Friday), CG Aug 28 + Sep 5 2024 (£10 days).
> All three caused test regressions (+0.73pp Borough, +13pp CG) when excluded.
> Root cause: Prophet averages anomalous days into the smooth fit rather than memorising them.
> Removing them shifts the trend/seasonality estimates in ways that hurt March 2026 predictions.
> **All anomalous day exclusions reverted.** Battersea first-day exclusion (Jan 18 2026) retained
> as it was already applied before this experiment.

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough −5.9pp, CG −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG −5.0pp |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough −1.0pp, CG −1.1pp |
| Added `temp_sq = (apparent_temperature_max − 15)²` | All locations | Borough −0.13pp, CG −0.62pp |
| Added `rainy_day` + `precip_sq` as extra regressors | Borough only | Borough −0.41pp |
| Footfall sanity check (< 10% of dow avg → substitute dow avg) | All locations | Data quality fix |
| Mothering Sunday added to Prophet holidays | All locations | Borough −1.06pp, CG −1.02pp |
| Added `network_momentum` as extra regressor | All locations | Borough −0.09pp, CG −0.22pp |
| London social calendar Phase 1 + Fashion Week + Trooping | CG only | CG −0.78pp |
| Easter Sunday added to Prophet holidays | All locations | Borough −0.23pp, CG −0.06pp |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.46% | **7.79%** | −8.93pp |
| Covent Garden | 11.80% | **7.18%** | **−8.78pp** |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 3.91% |
| Canary Wharf | 74 | 5.54% |
| Spitalfields | 65 | 5.48% |

---

### EXP-34 · Christmas Eve holiday

> Christmas Eve (Dec 24) is not a UK bank holiday — absent from the model's holiday calendar.
> Stores close early (typically noon–2pm); model consistently over-predicts CG on Dec 24.
> CG: Dec 24 2024 = 40.6% over-pred. Dec 24 2025 also in training → 2 occurrences.
> Added as `_ALWAYS_ON` alongside Easter Sunday (applies to all locations).

| Location | Before | After | Delta |
|---|---|---|---|
| Borough test | 7.79% | **7.66%** | −0.13pp |
| Covent Garden test | 7.18% | **7.00%** | −0.18pp |

Val also improved: Borough 9.46% → 9.53% (marginal), CG 11.80% → 11.07% (−0.73pp).

**Finding:** Clean win for both. Applied.

---

### EXP-35 · Shakespeare's Globe Season Active (binary regressor for Borough)

> Globe open-air season runs ~April 23 to ~October 26. Thousands of theatre-goers pass through
> Borough Market before/after performances — sustained 6-month footfall uplift.
> Modelled as `globe_season_active` binary regressor (1 = season active, 0 = winter).
> Only 1 Globe season in Borough's training data (Apr–Oct 2025).

| Config | Borough test |
|---|---|
| Christmas Eve only | 7.66% |
| Christmas Eve + Globe Season | 7.72% |

**Finding:** Globe Season adds noise (+0.06pp). Single occurrence insufficient for reliable coefficient.
Test period is March (outside Globe season), so there's no direct improvement path.
**Not applied.** Revisit when Borough has 2 Globe seasons (Oct 2026).

---

## Applied Changes (current production config)

| Change | Applied to | Impact |
|---|---|---|
| `changepoint_prior_scale=0.1` | All locations | Borough −5.9pp, CG −5.0pp |
| Removed `footfall_dow_avg`, `footfall_vs_dow` | All locations | Minor improvement |
| `log_y=True` (log1p/expm1) | Covent Garden only | CG −5.0pp |
| Added `precipitation_hours`, `daylight_duration` | All locations | Borough −1.0pp, CG −1.1pp |
| Added `temp_sq = (apparent_temperature_max − 15)²` | All locations | Borough −0.13pp, CG −0.62pp |
| Added `rainy_day` + `precip_sq` as extra regressors | Borough only | Borough −0.41pp |
| Footfall sanity check (< 10% of dow avg → substitute dow avg) | All locations | Data quality fix |
| Mothering Sunday added to Prophet holidays | All locations | Borough −1.06pp, CG −1.02pp |
| Added `network_momentum` as extra regressor | All locations | Borough −0.09pp, CG −0.22pp |
| London social calendar Phase 1 + Fashion Week + Trooping | CG only | CG −0.78pp |
| Easter Sunday added to Prophet holidays | All locations | Borough −0.23pp, CG −0.06pp |
| Christmas Eve added to Prophet holidays | All locations | Borough −0.13pp, CG −0.18pp |

**Current holdout metrics:**

| Location | Val MAPE | Test MAPE | vs original baseline |
|---|---|---|---|
| Borough | 9.53% | **7.66%** | −9.06pp |
| Covent Garden | 11.07% | **7.00%** | **−8.96pp** |

In-sample only (short history):

| Location | Days | In-sample MAPE |
|---|---|---|
| Battersea | 52 | 3.91% |
| Canary Wharf | 74 | 5.54% |
| Spitalfields | 65 | 5.48% |

---

### EXP-36 · Autumn Half-Term (CG) + New Year's Eve (Borough)

#### A — Autumn Half-Term window for CG
> CG October MAPE is 13.5%, partly driven by Oct 1-12 over-prediction (post-September drop).
> Tested last week of October (Oct 27-31) as a named holiday for CG, to capture half-term uplift.
> Only 2 half-term windows in training (Oct 2024, Oct 2025).

| Location | Before | After | Delta |
|---|---|---|---|
| Covent Garden val | 11.07% | 12.23% | **+1.16pp worse** |
| Covent Garden test | 7.00% | 7.22% | **+0.22pp worse** |

**Finding:** Hurts CG. Half-term uplift signal is too noisy with 2 occurrences.
Not applied. October over-prediction remains a data-limited problem.

#### B — New Year's Eve for Borough
> Borough opts out of the social calendar. NYE is in `_universal` but not applied to Borough.
> Tested adding NYE specifically to Borough (2 occurrences: Dec 31 2024, Dec 31 2025).

| Location | Before | After | Delta |
|---|---|---|---|
| Borough val | 9.53% | 9.52% | −0.01pp |
| Borough test | 7.66% | 7.66% | **neutral** |

**Finding:** No change. NYE has no measurable effect on Borough revenue in the training data.
Not applied.

---

### EXP-37 · Valentine's Day + Bonfire Night for Borough

> Borough opts out of the full social calendar (EXP-30 confirmed +0.95pp). But specific events
> with 2 occurrences in Borough's training window are now testable individually.
> - Bonfire Night: Nov 5 2024, Nov 5 2025 (2 occurrences ✓)
> - Valentine's Day: Feb 14 2025, Feb 14 2026 (2 occurrences ✓)
> Added to `_ALWAYS_ON` (same mechanism as Easter Sunday/Christmas Eve).

| Config | Borough val | Borough test | Delta |
|---|---|---|---|
| Baseline | 9.53% | 7.66% | — |
| + Valentine's Day + Bonfire Night | 9.48% | 7.77% | **+0.11pp worse** |
| + Bonfire Night only | 9.43% | 7.65% | −0.01pp (noise) |

**Finding:** Combined hurts; Bonfire Night alone is neutral (within rounding noise). Valentine's Day
repeats EXP-25 pattern — even with 2 occurrences, the coefficient is noisy for Borough.
**Not applied.** Reverted to baseline.

---

## Data-Limited Frontier Assessment (March 2026)

After 37 experiments, the models have been improved by ~9pp test MAPE each:
- Borough: 16.72% → **7.66%** (−9.06pp)
- Covent Garden: 15.96% → **7.00%** (−8.96pp)

The remaining errors are largely **data-limited** — not solvable with more features or calendar events
until more training history accumulates. Specific blockers:

| Improvement | What's needed | When |
|---|---|---|
| Borough social calendar (Marathon, Father's Day, NHC) | 2nd annual occurrence | ~Jun 2026 |
| Shakespeare's Globe Season for Borough | 2nd Globe season | Oct 2026 |
| Black Friday revisit | 3rd occurrence | Nov 2026 |
| LFW coefficient stability for CG | 3rd Sep LFW | Sep 2026 |
| September cluster (Borough Sep 17% MAPE) | 2nd September for modelling | Sep 2026 |
| October over-prediction (CG Oct 13.5% MAPE) | 3rd October to stabilise | Oct 2026 |

**Near-term actions that remain valid:**
- Retrain all 5 models monthly as new Revel data arrives
- Venue events (Phase 2) once PredictHQ plan upgrades
- Per-location log_y for Battersea/Spitalfields/CW once 90+ days

---

## To Test Next (future sessions)

| Idea | Producible? | Notes |
|---|---|---|
| Borough Market recurring events | Research needed | Sep 26–30 2025 cluster (20–40% under-pred, 5 days) — Totally Thames ends Sep 30, Woolmen's Sheep Drive Sep 28. Monitor Sep 2026 to confirm if recurring. |
| Shakespeare's Globe Season (Borough) | Yes (Oct 2026) | 1 season in training — noisy. Revisit when Borough has 2 seasons (Oct 2026). |
| Black Friday | Yes (Nov 2026) | Revisit with 3rd occurrence; currently hurts CG, marginal for Borough |
| London Fashion Week coefficient stability | Monitoring | 2024 effect much larger than 2025 (store newness effect). Will stabilise by Sep 2026 (3rd occurrence). |
| Borough social calendar (annual events at 2 occurrences) | Yes (mid-2026) | Marathon, Father's Day, Carnival all hit 2 occurrences ~Jun 2026 |
| Google Trends | Conditional | Needs reliable API; pytrends is blocked/unreliable |
| Per-location log_y expansion | Yes | Revisit Battersea/Spitalfields/CW once 90+ days |
| Hourly revenue model | Yes | Hourly Revel CSVs in GCS back to ~Sep 2025; separate model |
| Venue events (Phase 2) | Blocked | `venue_events.py` + `fetch_venue_events.py` built; needs PredictHQ plan upgrade for `within` geo filter |
| Stacked ensemble (production two-pass) | Yes | Run base models → compute peer_yhat → re-run; valid at inference time but needs separate coefficient training strategy |
