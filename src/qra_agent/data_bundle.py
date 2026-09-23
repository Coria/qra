from __future__ import annotations

"""Loading and validation for ``qra.backtest_bundle/v1``.

A bundle is the portable contract between a backtest engine and this review
agent.  It carries report metadata, reproducibility metadata, and optional
tabular datasets.  Relative dataset paths are resolved against the JSON file's
directory, which makes a bundle easy to move together with its data files.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .orders import COLUMN_ALIASES, normalize_orders


BUNDLE_SCHEMA_VERSION = "qra.backtest_bundle/v1"
DATASET_REQUIRED_FIELDS = {
    "returns": {"date", "strategy"},
    "benchmark": {"date", "benchmark"},
    "orders": {"timestamp", "symbol", "size", "price", "side"},
    "positions": {"date", "symbol", "quantity", "market_price"},
    "nav": {"date", "nav"},
    "market_data": {"date", "symbol", "close"},
    "corporate_actions": {"date", "symbol", "action"},
}
KNOWN_DATASETS = set(DATASET_REQUIRED_FIELDS)
REPORT_FORMATS = {"quantstats_html"}
TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet", ".pq"}


def load_data_bundle(path: str | Path) -> dict[str, Any]:
    """Load a bundle, validate its contract, and resolve its base directory."""
    bundle_path = Path(path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    errors = validate_data_bundle(bundle)
    if errors:
        raise ValueError(f"invalid backtest bundle: {'; '.join(errors)}")
    bundle = dict(bundle)
    bundle["_bundle_path"] = str(bundle_path.resolve())
    return bundle


def validate_data_bundle(bundle: dict[str, Any]) -> list[str]:
    """Return human-readable schema errors without raising on the first issue."""
    errors: list[str] = []
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {BUNDLE_SCHEMA_VERSION}")

    report = bundle.get("report")
    if not isinstance(report, dict):
        errors.append("report must be an object")
    else:
        if not report.get("path"):
            errors.append("report.path is required")
        if report.get("format") not in REPORT_FORMATS:
            errors.append("report.format must be quantstats_html")

    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
    else:
        for key in {
            "strategy_id",
            "strategy_version",
            "data_version",
            "universe",
            "parameters",
            "transaction_cost",
            "rebalance_rule",
        }:
            if key not in metadata or metadata[key] in (None, "", [], {}):
                errors.append(f"metadata.{key} is required")

    datasets = bundle.get("datasets")
    if not isinstance(datasets, dict):
        errors.append("datasets must be an object")
        return errors
    unknown = set(datasets).difference(KNOWN_DATASETS)
    if unknown:
        errors.append(f"unknown datasets: {', '.join(sorted(unknown))}")
    for name, descriptor in datasets.items():
        if not isinstance(descriptor, dict):
            errors.append(f"datasets.{name} must be an object")
            continue
        has_path = bool(descriptor.get("path"))
        has_records = isinstance(descriptor.get("records"), list)
        if has_path == has_records:
            errors.append(f"datasets.{name} must define exactly one of path or records")
        elif has_path and not descriptor.get("format"):
            errors.append(f"datasets.{name}.format is required with path")
        elif has_records:
            required = DATASET_REQUIRED_FIELDS.get(name, set())
            first = descriptor["records"][0] if descriptor["records"] else {}
            missing = sorted(required.difference(first))
            if missing:
                errors.append(f"datasets.{name}.records missing fields: {', '.join(missing)}")
    return errors


def bundle_dataset_frame(bundle: dict[str, Any], name: str) -> pd.DataFrame | None:
    """Load one named dataset as a normalized, lower-column DataFrame."""
    descriptor = bundle.get("datasets", {}).get(name)
    if descriptor is None:
        return None
    if "records" in descriptor:
        frame = pd.DataFrame.from_records(descriptor["records"])
    else:
        frame = _load_tabular(_resolve_bundle_path(bundle, descriptor["path"]))
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if name == "orders":
        frame = frame.rename(columns=COLUMN_ALIASES)
    date_format = descriptor.get("date_format")
    for column in ("date", "timestamp"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], format=date_format, errors="raise")
    missing = DATASET_REQUIRED_FIELDS.get(name, set()).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {', '.join(sorted(missing))}")
    return frame


def bundle_returns_frame(bundle: dict[str, Any]) -> pd.DataFrame | None:
    """Load returns and align an optional benchmark dataset by date."""
    returns = bundle_dataset_frame(bundle, "returns")
    benchmark = bundle_dataset_frame(bundle, "benchmark")
    if returns is None and benchmark is None:
        return None
    if returns is None:
        raise ValueError("returns dataset is required when benchmark dataset is provided")
    returns = returns.copy()
    if "date" in returns.columns:
        returns = returns.sort_values("date").set_index("date")
    else:
        returns.index = pd.to_datetime(returns.index)
        returns = returns.sort_index()
    numeric_columns = [column for column in returns.columns if column != "date"]
    returns[numeric_columns] = returns[numeric_columns].apply(pd.to_numeric, errors="coerce")
    returns = returns.dropna(subset=["strategy"])

    if benchmark is not None:
        benchmark = benchmark[["date", "benchmark"]].sort_values("date")
        benchmark["benchmark"] = pd.to_numeric(benchmark["benchmark"], errors="coerce")
        returns = returns.join(benchmark.set_index("date"), how="outer")
        if "strategy" in returns.columns:
            returns["strategy"] = returns["strategy"].fillna(0.0)
    return returns.dropna(how="all")


def bundle_orders_frame(bundle: dict[str, Any]) -> pd.DataFrame | None:
    """Return the normalized orders dataset when the bundle supplies one."""
    orders = bundle_dataset_frame(bundle, "orders")
    return normalize_orders(orders) if orders is not None else None


def bundle_report_path(bundle: dict[str, Any]) -> str:
    """Resolve the QuantStats HTML path against the bundle directory."""
    return str(_resolve_bundle_path(bundle, bundle["report"]["path"]))


def bundle_info(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    """Expose schema, source path and report path in the final review JSON."""
    if bundle is None:
        return None
    return {
        "schema_version": bundle.get("schema_version"),
        "bundle_id": bundle.get("bundle_id"),
        "generated_at": bundle.get("generated_at"),
        "engine": bundle.get("engine"),
        "engine_version": bundle.get("engine_version"),
        "path": bundle.get("_bundle_path"),
        "datasets": sorted(bundle.get("datasets", {})),
    }


def _resolve_bundle_path(bundle: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path as bundle-relative when possible."""
    path = Path(value)
    if path.is_absolute() or bundle.get("_bundle_path") is None:
        return path
    return (Path(bundle["_bundle_path"]).parent / path).resolve()


def _load_tabular(path: Path) -> pd.DataFrame:
    """Load CSV, Excel or Parquet data using the file suffix."""
    suffix = path.suffix.lower()
    if suffix not in TABULAR_SUFFIXES:
        raise ValueError(f"unsupported dataset file format: {path.suffix}")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)
