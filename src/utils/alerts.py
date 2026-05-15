"""
Slack alerting via Incoming Webhook.

Set SLACK_WEBHOOK_URL in your environment (or .env):
    export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ

If the env var is absent, alerts are logged locally and silently skipped.
"""

import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.utils.logging import get_logger

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = get_logger(__name__)

_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def _post(payload: dict) -> None:
    if not _WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL not set — alert not sent")
        return
    try:
        resp = requests.post(_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Slack alert failed: {e}")


def alert_retrain_failure(location: str, error: str) -> None:
    _post({
        "text": (
            f":x: *Retrain failed* — `{location}`\n"
            f"```{error[:500]}```\n"
            f"_Serving previous model. Check GCS for details._"
        )
    })


def alert_validation_gate(location: str, new_mape: float, prev_mape: float) -> None:
    _post({
        "text": (
            f":warning: *Validation gate failed* — `{location}`\n"
            f"New MAPE: `{new_mape:.2f}%`  |  Previous MAPE: `{prev_mape:.2f}%`\n"
            f"_Model not promoted. Previous version still serving._"
        )
    })


def alert_data_quarantine(location: str, dates: list[str], reason: str) -> None:
    date_list = ", ".join(dates[:5]) + (" ..." if len(dates) > 5 else "")
    _post({
        "text": (
            f":microscope: *Data quarantined* — `{location}`\n"
            f"Dates: `{date_list}`\n"
            f"Reason: {reason}\n"
            f"_Review: gs://jenki-forecast/quarantine/{location}/_"
        )
    })


def alert_structural_break(location: str, recent_avg: float, baseline_avg: float) -> None:
    pct_str = (
        f"`-{(baseline_avg - recent_avg) / baseline_avg * 100:.0f}%`"
        if baseline_avg > 0 else "(baseline was zero)"
    )
    _post({
        "text": (
            f":rotating_light: *Structural break detected* — `{location}`\n"
            f"7-day avg: `£{recent_avg:,.0f}`  |  28-day baseline: `£{baseline_avg:,.0f}`  "
            f"({pct_str})\n"
            f"_Retraining paused for this location pending review._"
        )
    })


def alert_forecast_stale(location: str, last_date: str) -> None:
    _post({
        "text": (
            f":clock3: *Stale forecast* — `{location}`\n"
            f"No forecast generated today. Serving last available: `{last_date}`."
        )
    })


def alert_retrain_success(results: dict[str, dict]) -> None:
    """Summary message after a full retrain run. results = {loc: {mape, promoted}}."""
    lines = []
    for loc, r in results.items():
        icon = ":white_check_mark:" if r["promoted"] else ":x:"
        lines.append(f"{icon} `{loc}` — MAPE `{r['mape']:.2f}%`")
    _post({
        "text": (
            f":moneybag: *Revenue retrain complete* — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            + "\n".join(lines)
        )
    })


def alert_daily_summary(
    revenue_results: dict[str, dict] | None = None,
    sku_results: dict[str, dict] | None = None,
) -> None:
    """Single consolidated message covering both forecasts (when both run in the same job)."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    sections = []

    if revenue_results:
        lines = []
        for loc, r in revenue_results.items():
            icon = ":white_check_mark:" if r.get("promoted") else ":x:"
            mape = r.get("mape", -1)
            lines.append(f"{icon} `{loc}` — MAPE `{mape:.1f}%`")
        sections.append(":moneybag: *Revenue model*\n" + "\n".join(lines))

    if sku_results:
        lines = []
        for loc, r in sku_results.items():
            if isinstance(r.get("promoted"), dict):
                icon = ":white_check_mark:" if r["promoted"].get("1hr") else ":x:"
                wmape = r.get("wmape", {}).get("1hr", -1)
                n_skus = r.get("metrics", {}).get("1hr", {}).get("n_skus", 0)
                lines.append(f"{icon} `{loc}` — WMAPE `{wmape:.1f}%` ({n_skus} SKUs)")
        sections.append(":package: *SKU model (1hr)*\n" + "\n".join(lines))

    if not sections:
        return

    _post({
        "text": f":repeat: *Daily forecast run complete* — {timestamp}\n\n"
                + "\n\n".join(sections)
    })
