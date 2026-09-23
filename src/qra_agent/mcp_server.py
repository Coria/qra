"""FastMCP tools for other agents and workflow engines.

Each tool is independently callable so an external strategy-generation agent
can parse a report, recompute metrics, inspect orders, validate a bundle, or
ask for a structured analysis without owning QRA's internals.
"""

from __future__ import annotations

from fastmcp import FastMCP

from .anomaly import detect_anomalies
from .attribution import run_attribution
from .data_bundle import (
    bundle_info,
    bundle_orders_frame,
    bundle_report_path,
    bundle_returns_frame,
    load_data_bundle,
    validate_data_bundle,
)
from .metrics import calculate_metrics, load_returns
from .orders import analyze_orders, load_orders
from .report_parser import parse_report


app = FastMCP("qra")


@app.tool
def parse_backtest_report(report_path: str) -> dict:
    """Parse QuantStats report provenance and key metrics."""
    return parse_report(report_path)


@app.tool
def recompute_backtest_metrics(
    returns_path: str,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Recalculate metrics from a raw returns file."""
    return calculate_metrics(load_returns(returns_path), periods_per_year, risk_free_rate)


@app.tool
def generate_backtest_summary(report_path: str, returns_path: str | None = None) -> dict:
    """Parse a report and optionally attach recomputed metrics/anomalies."""
    report = parse_report(report_path)
    metrics = None
    returns = None
    if returns_path:
        returns = load_returns(returns_path)
        metrics = calculate_metrics(returns, report["periods_per_year"], report["risk_free_rate"])
    return {
        "report": report,
        "recalculated_metrics": metrics,
        "anomalies": detect_anomalies(report, metrics, returns),
    }


@app.tool
def analyze_backtest_orders(orders_path: str) -> dict:
    """Analyze order activity and FIFO round trips."""
    return analyze_orders(load_orders(orders_path))


@app.tool
def run_return_attribution(returns_path: str) -> dict:
    """Run simple monthly and factor/benchmark attribution."""
    return run_attribution(load_returns(returns_path))


@app.tool
def validate_backtest_bundle(bundle_path: str) -> dict:
    """Validate a bundle without loading its full report/datasets."""
    import json
    from pathlib import Path

    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    info = bundle_info(bundle)
    info["validation_errors"] = validate_data_bundle(bundle)
    return info


@app.tool
def analyze_backtest_bundle(bundle_path: str) -> dict:
    """Parse and analyze a complete portable backtest bundle."""
    bundle = load_data_bundle(bundle_path)
    report = parse_report(bundle_report_path(bundle))
    returns = bundle_returns_frame(bundle)
    metrics = calculate_metrics(returns, report["periods_per_year"], report["risk_free_rate"]) if returns is not None else None
    orders = bundle_orders_frame(bundle)
    return {
        "bundle": bundle_info(bundle),
        "report": report,
        "recalculated_metrics": metrics,
        "anomalies": detect_anomalies(report, metrics, returns),
        "orders_analysis": analyze_orders(orders) if orders is not None else {"available": False},
    }


def run() -> None:
    """Start the MCP stdio server."""
    app.run()
