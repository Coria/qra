"""Lightweight return attribution using only data carried in the bundle."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def run_attribution(returns: pd.DataFrame | None) -> dict[str, Any]:
    """Compute monthly contribution, drawdown timing and simple exposures.

    Factor columns named ``factor_*`` are fit with OLS period returns.  When no
    factors exist but a benchmark exists, the function estimates benchmark
    beta.  This is intentionally simpler than Brinson or Barra attribution.
    """
    if returns is None:
        return {
            "available": False,
            "reason": "raw returns are required; the QuantStats HTML does not embed the return series",
        }

    strategy = returns["strategy"].dropna()
    if strategy.empty:
        return {"available": False, "reason": "empty return series"}

    monthly = (1 + strategy).resample("ME").prod() - 1
    positive = monthly[monthly > 0]
    negative = monthly[monthly < 0]
    drawdown = (1 + strategy).cumprod()
    drawdown = drawdown / drawdown.cummax() - 1
    worst_drawdown_date = drawdown.idxmin()

    attribution: dict[str, Any] = {
        "available": True,
        "monthly_contribution": {
            "positive_months": int(len(positive)),
            "negative_months": int(len(negative)),
            "positive_month_mean": float(positive.mean()) if not positive.empty else 0.0,
            "negative_month_mean": float(negative.mean()) if not negative.empty else 0.0,
        },
        "worst_drawdown_date": worst_drawdown_date.isoformat(),
    }

    factor_columns = [column for column in returns.columns if column.startswith("factor_")]
    if factor_columns:
        aligned = returns[[*factor_columns, "strategy"]].dropna()
        x = np.column_stack([np.ones(len(aligned)), *[aligned[column] for column in factor_columns]])
        y = aligned["strategy"].to_numpy()
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        attribution["factor_exposures"] = {
            factor: float(value) for factor, value in zip(factor_columns, coefficients[1:])
        }
    elif "benchmark" in returns.columns:
        aligned = returns[["strategy", "benchmark"]].dropna()
        covariance = np.cov(aligned["strategy"], aligned["benchmark"], ddof=1)
        variance = float(np.var(aligned["benchmark"], ddof=1))
        beta = float(covariance[0, 1] / variance) if variance else math.nan
        attribution["benchmark_beta"] = beta
    return attribution
