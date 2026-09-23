"""Rule-based anomaly detection for parsed and recomputed metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def detect_anomalies(
    report: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    returns: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Detect reporting inconsistencies, extreme returns and volatility breaks.

    Rules intentionally remain explainable.  They are designed to prompt human
    review, not to replace a statistical significance framework.
    """
    issues: list[dict[str, str]] = []
    parsed = report.get("parsed_metrics", {})
    max_drawdown = parsed.get("max_drawdown", math.nan)
    cumulative_return = parsed.get("cumulative_return", math.nan)
    volatility = parsed.get("annual_volatility", math.nan)
    win_days = parsed.get("win_days", math.nan)

    if not math.isnan(max_drawdown) and max_drawdown > 0:
        issues.append({"severity": "error", "message": "Max drawdown should be non-positive"})
    if not math.isnan(cumulative_return) and not math.isnan(volatility) and abs(cumulative_return) > 10 * volatility:
        issues.append({"severity": "warning", "message": "Cumulative return is extreme relative to annual volatility"})
    if not math.isnan(win_days) and win_days > 0.75:
        issues.append({"severity": "warning", "message": "Unusually high daily win rate; inspect survivorship or lookahead bias"})

    if metrics and returns is not None:
        strategy = returns["strategy"].dropna()
        rolling_std = strategy.rolling(20).std() * math.sqrt(252)
        baseline = float(rolling_std.median())
        recent = float(rolling_std.tail(20).mean())
        if baseline and recent > baseline * 2:
            issues.append({"severity": "warning", "message": f"Recent volatility is {recent / baseline:.1f}x the historical median"})
        z_scores = (strategy - strategy.mean()) / strategy.std(ddof=1)
        outliers = strategy[z_scores.abs() > 5]
        if not outliers.empty:
            issues.append(
                {
                    "severity": "warning",
                    "message": f"{len(outliers)} daily returns exceed 5 standard deviations",
                }
            )

    return {"issues": issues, "checked": "report+metrics+returns" if returns is not None else "report-only"}
