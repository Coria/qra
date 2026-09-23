"""Reproducibility checks for a backtest report.

The check asks whether the reviewer has enough information to reproduce the
result: report provenance, strategy code/data versions, strategy parameters,
costs, universe, rebalance rule, and raw returns.  Missing any blocker is not
just a warning; the graph routes to the reproducibility/rejection branch.
"""

from __future__ import annotations

from typing import Any


REQUIRED_METADATA = {
    "strategy_id": "strategy identifier",
    "data_version": "market data version",
    "strategy_version": "strategy code version",
    "parameters": "strategy parameters",
    "transaction_cost": "transaction cost assumptions",
    "universe": "instrument universe",
    "rebalance_rule": "rebalance rule",
}


def _is_missing(value: Any) -> bool:
    """Treat blank, empty containers and unspecified values as missing."""
    return value is None or value == "" or value == [] or value == {} or value == "auto"


def check_integrity(
    report: dict[str, Any],
    metadata: dict[str, Any],
    returns_path: str | None = None,
    *,
    has_raw_returns: bool = False,
) -> dict[str, Any]:
    """Return missing fields and human-readable blockers."""
    missing = []
    for key, label in REQUIRED_METADATA.items():
        value = metadata.get(key)
        if _is_missing(value):
            missing.append(label)
    if not has_raw_returns and not returns_path:
        missing.append("raw returns data")
    if not report.get("report_version"):
        missing.append("report generator version")
    if not report.get("generated_at"):
        missing.append("report generation timestamp")
    if not report.get("period_start") or not report.get("period_end"):
        missing.append("period start/end")

    blockers = [{"field": key, "reason": f"missing {label}"} for key, label in REQUIRED_METADATA.items()]
    blockers = [item for item in blockers if item["field"] in _missing_fields(metadata)]
    if not has_raw_returns and not returns_path:
        blockers.append({"field": "returns", "reason": "raw return series is required to recompute metrics"})
    return {
        "passed": not missing,
        "missing": sorted(set(missing)),
        "blockers": blockers,
    }


def _missing_fields(metadata: dict[str, Any]) -> list[str]:
    """Return required metadata keys whose values are missing."""
    return [key for key in REQUIRED_METADATA if _is_missing(metadata.get(key))]
