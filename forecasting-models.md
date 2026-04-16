# Jenki Revenue Forecasting System

## Overview

This system predicts daily revenue for each Jenki location across London, up to 14 days ahead. It retrains itself every morning using the latest sales, weather, footfall, and event data, then publishes forecasts with confidence bounds. Fully automated - no manual intervention required.

Each location gets its own model, tuned to its specific trading patterns. If a model degrades or data looks suspicious, the system catches it automatically and keeps the previous working version in place.

| Location | Forecast Horizon | Accuracy |
|----------|-----------------|----------|
| Borough Market | 14 days | 93.9% |
| Covent Garden | 14 days | 92.5% |
| Canary Wharf | 7 days | 91.3% |
| Spitalfields | 7 days | 89.1% |
| Battersea | 7 days | 82.4% |

*Newer locations have less history to learn from and improve as more trading days accumulate.*

---

## Data Sources

The system combines six distinct real-time data sources per location:

**1. Point-of-Sale Revenue** - Daily till revenue per location, ingested automatically each morning.

**2. Weather (6 variables)** - Feels-like temperature, rainfall, hours of rain, sunshine duration, wind speed, and daylight hours. Fetched for each location's exact coordinates from the Open-Meteo API. Historical data for training, forecast data for predictions.

**3. TfL Station Footfall** - London Underground exit counts from the nearest major station to each location (London Bridge for Borough, Liverpool Street for Spitalfields, Covent Garden for Covent Garden, etc.). Captures real-world area activity.

**4. TfL Disruption Detection** - Monitors TfL lines serving each location for planned closures, strikes, and suspensions. During training, the system identifies historical disruption days from footfall anomalies. At inference time, it checks the live TfL API for upcoming disruptions.

**5. London Events Calendar** - 15+ recurring events with algorithmically computed dates that project into the future automatically: UK bank holidays, London Marathon, Pride, Notting Hill Carnival, Chinese New Year, Wimbledon, Shakespeare's Globe season, Chelsea Flower Show, London Fashion Week, Diwali, and more. Each event is assigned only to the locations it demonstrably affects.

**6. Cross-Location Network Momentum** - A signal that captures whether the Jenki network as a whole is trading above or below its recent baseline. If one location starts trending up, that momentum feeds into the other models before their own revenue data confirms the shift.

The system also engineers derived features from raw data - temperature sensitivity curves, rain intensity effects, and seasonal indicators.

---

## Daily Pipeline

Every morning at 6:00 AM, the full pipeline runs automatically:

```
 6:00 AM    Scheduled trigger
    |
    v
 INGEST     Pulls latest sales, weather, footfall, and disruption data
    |
    v
 QUALITY    Screens for anomalies, closures, and data gaps
    |        - Revenue spikes > 4x normal are quarantined
    |        - Revenue drops > 40% pause retraining
    |        - Missing days are interpolated
    v
 RETRAIN    Fits a fresh model per location on all available history
    |
    v
 VALIDATE   Tests the new model against a held-out evaluation window
    |        - Compares accuracy to the currently deployed version
    |        - Rejects if accuracy drops by more than 1.5pp
    |        - Runs a smoke test (3-day forward prediction sanity check)
    v
 PUBLISH    Uploads forecast CSVs with daily predictions + confidence bounds
    |
    v
 ALERT      Posts summary to Slack with accuracy metrics per location
```

Each location runs independently. If one fails, the others continue unaffected.

---

## Quality Gates

Five layers of automated protection prevent bad predictions from reaching production:

| Gate | Function |
|------|----------|
| **Anomaly detection** | Revenue spikes > 4x rolling average are automatically removed from training and logged |
| **Structural break detection** | Revenue drops > 40% pause retraining for that location pending review |
| **Validation gate** | New model rejected if accuracy drops more than 1.5 percentage points vs the deployed version |
| **Smoke test** | 3-day forward prediction checked for NaN, all-zero, or negative outputs before deployment |
| **Sanity limit** | Any model below 50% accuracy is automatically rejected |

If any gate triggers, the previous working model stays in place and a diagnostic alert is sent to Slack.

---

## Forecast Output

One CSV per location, published daily:

```
date,predicted_revenue,lower_bound,upper_bound
2026-04-08,2907.46,2316.17,3498.74
2026-04-09,2838.86,2165.95,3511.77
2026-04-10,2321.84,1605.72,3037.95
```

Confidence bounds are computed from actual historical prediction errors grouped by day of week, providing approximately 90% coverage.

---

## Architecture

```
+------------------+     +------------------+     +-----------------+
|   Data Sources   |     |  Daily Pipeline  |     |    Outputs      |
+------------------+     +------------------+     +-----------------+
|                  |     |                  |     |                 |
|  Point-of-Sale   |---->|  Ingest + Clean  |---->|  Forecast CSVs  |
|  (daily revenue) |     |                  |     |  (per location) |
|                  |     |  Quality Gates   |     |                 |
|  Weather API     |---->|  (5 checks)      |---->|  Model Registry |
|  (6 variables)   |     |                  |     |  (versioned)    |
|                  |     |  Retrain + Test  |     |                 |
|  TfL Footfall    |---->|  (per location)  |---->|  Slack Alerts   |
|  (station exits) |     |                  |     |  (daily summary)|
|                  |     |  Validate + Gate |     |                 |
|  TfL Disruptions |---->|  (vs. deployed)  |     |  Quarantine Log |
|  (strikes/works) |     |                  |     |  (anomalies)    |
|                  |     |  Promote + Ship  |     |                 |
|  London Events   |---->|  (if improved)   |     |                 |
|  (15+ events)    |     |                  |     |                 |
|                  |     |                  |     |                 |
|  Network Signal  |---->|                  |     |                 |
|  (cross-location)|     |                  |     |                 |
+------------------+     +------------------+     +-----------------+

Fully serverless | Runs daily at 6:00 AM | ~10 min end-to-end
```

---

## Accuracy Over Time

Model accuracy improves as locations accumulate more trading history:

| History | Expected Accuracy | What the Model Learns |
|---------|-------------------|----------------------|
| 0-3 months | 80-85% | Weekly patterns, weather effects |
| 3-6 months | 85-90% | Stable trend, seasonal shifts |
| 6-12 months | 90-93% | Partial yearly seasonality, holiday effects |
| 12+ months | 92-95% | Full yearly cycle (Borough and Covent Garden are here) |

The system adapts its model configuration as locations mature - newer locations use simpler, more robust settings that automatically graduate to more sophisticated configurations as history grows.

---

## Monitoring

The system posts to Slack after every run:

```
Daily retrain complete - 2026-04-10 06:09 UTC
[pass] borough       - 93.9% accuracy
[pass] covent_garden - 92.5% accuracy
[pass] canary_wharf  - 91.3% accuracy
[pass] spitalfields  - 89.1% accuracy
[pass] battersea     - 82.4% accuracy
```

Alerts fire automatically for: failed retrains, accuracy regressions, data anomalies, structural breaks, and missing forecasts.
