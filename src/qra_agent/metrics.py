"""Return-series loading and QuantStats-compatible metric recalculation.

Inputs are simple arithmetic period returns, not equity NAV levels.  The
annualization factor is taken from the report (typically 252 daily or 12
monthly observations).  Sharpe and Sortino use the same annualized excess-mean
convention as QuantStats; VaR and Expected Shortfall use a 95% daily Gaussian
VaR cutoff.  Keeping these assumptions explicit is important when comparing the
result with report values.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"strategy"}
OPTIONAL_COLUMNS = {"date", "benchmark"}


def load_returns(returns_path: str | Path) -> pd.DataFrame:
    """Load a returns table and normalize it to a datetime-indexed frame."""
    path = Path(returns_path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError("returns file must contain a 'strategy' returns column")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date").set_index("date")
    else:
        frame = frame.sort_index()
        frame.index = pd.to_datetime(frame.index)
    return frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def _annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    """Compound period returns and annualize using the report frequency."""
    count = max(1, len(returns))
    growth = float((1 + returns).prod())
    return math.copysign(abs(growth) ** (periods_per_year / count) - 1, growth)


def _drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown from the compounded equity curve."""
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def _consecutive_counts(returns: pd.Series) -> tuple[int, int]:
    """Count the longest consecutive win and loss streaks."""
    wins = losses = max_wins = max_losses = 0
    for value in returns:
        if value > 0:
            wins += 1
            losses = 0
        elif value < 0:
            losses += 1
            wins = 0
        else:
            wins = losses = 0
        max_wins = max(max_wins, wins)
        max_losses = max(max_losses, losses)
    return max_wins, max_losses


def calculate_metrics(
    returns: pd.DataFrame | pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Recalculate performance and risk metrics from arithmetic returns."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame("strategy")
    strategy = returns["strategy"].dropna()
    if strategy.empty:
        raise ValueError("strategy return series is empty")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    excess = strategy - risk_free_rate / periods_per_year
    volatility = float(strategy.std(ddof=1)) * math.sqrt(periods_per_year)
    excess_annualized_return = float(excess.mean()) * periods_per_year
    sharpe = excess_annualized_return / volatility if volatility else math.nan
    downside_deviation = float(np.sqrt(np.square(np.minimum(excess, 0.0)).mean() * periods_per_year))
    sortino = excess_annualized_return / downside_deviation if downside_deviation else math.nan
    drawdown = _drawdown_series(strategy)
    max_drawdown = float(drawdown.min())
    cagr = _annualized_return(strategy, periods_per_year)
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else math.nan
    max_wins, max_losses = _consecutive_counts(strategy)
    wins = strategy[strategy > 0]
    losses = strategy[strategy < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if float(losses.sum()) != 0 else math.nan
    daily_var = float(strategy.mean() - 1.6448536269514722 * strategy.std(ddof=1))
    cvar = float(strategy[strategy <= daily_var].mean()) if (strategy <= daily_var).any() else daily_var

    result: dict[str, Any] = {
        "cumulative_return": float((1 + strategy).prod() - 1),
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "daily_value_at_risk": daily_var,
        "expected_shortfall": cvar,
        "profit_factor": profit_factor,
        "win_days": float((strategy > 0).mean()),
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "observation_count": int(len(strategy)),
        "period_start": strategy.index.min().isoformat(),
        "period_end": strategy.index.max().isoformat(),
    }

    if "benchmark" in returns.columns:
        benchmark = returns["benchmark"].dropna()
        aligned = returns[["strategy", "benchmark"]].dropna()
        if not aligned.empty:
            covariance = np.cov(aligned["strategy"], aligned["benchmark"], ddof=1)
            benchmark_variance = float(np.var(aligned["benchmark"], ddof=1))
            beta = float(covariance[0, 1] / benchmark_variance) if benchmark_variance else math.nan
            correlation = float(aligned["strategy"].corr(aligned["benchmark"]))
            alpha = cagr - _annualized_return(aligned["benchmark"], periods_per_year) * beta
            result.update(
                {
                    "benchmark_cagr": _annualized_return(benchmark, periods_per_year),
                    "beta": beta,
                    "alpha": alpha,
                    "correlation": correlation,
                }
            )
    return result
