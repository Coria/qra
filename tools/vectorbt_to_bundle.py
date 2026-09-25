from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
REQUIRED_METADATA_FIELDS = {
    "strategy_id",
    "strategy_version",
    "data_version",
    "universe",
    "parameters",
    "transaction_cost",
    "rebalance_rule",
}
ORDER_COLUMN_MAP = {
    "order_id": "order_id",
    "id": "order_id",
    "column": "symbol",
    "symbol": "symbol",
    "asset": "symbol",
    "timestamp": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
    "size": "size",
    "quantity": "size",
    "qty": "size",
    "price": "price",
    "fees": "fees",
    "fee": "fees",
    "commission": "fees",
    "side": "side",
}


def portfolio_to_bundle(
    portfolio: Any,
    report_path: str | Path,
    metadata: dict[str, Any],
    *,
    close: pd.DataFrame | pd.Series | None = None,
    returns_source: str = "value",
    benchmark: pd.DataFrame | pd.Series | None = None,
    datasets: dict[str, pd.DataFrame] | None = None,
    generate_report: bool = False,
    inline: bool = False,
    output_dir: Path | None = None,
    bundle_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Convert a vectorbt Portfolio object to a qra.backtest_bundle/v1 dictionary.

    When ``inline=False`` the returned second value is the dataset directory.
    The bundle then references CSV files inside that directory.
    """
    if returns_source not in {"value", "portfolio_returns"}:
        raise ValueError("returns_source must be 'value' or 'portfolio_returns'")

    report_path = Path(report_path)
    if not report_path.exists():
        if not generate_report:
            raise FileNotFoundError(f"QuantStats report does not exist: {report_path}")
        _generate_quantstats_report(portfolio, report_path)

    nav = _portfolio_value(portfolio)
    if nav is None or nav.empty:
        raise ValueError("portfolio has no value/nav series")
    nav_frame = pd.DataFrame({"date": nav.index, "nav": nav.astype(float)})

    if returns_source == "value":
        strategy_returns = nav.sort_index().pct_change(fill_method=None).fillna(0.0)
    else:
        strategy_returns = _portfolio_returns(portfolio)

    returns_frame = pd.DataFrame({"date": strategy_returns.index, "strategy": strategy_returns.astype(float)})
    orders = _orders_frame(portfolio)
    normalized_close = _close_frame(close if close is not None else _portfolio_close(portfolio))
    positions = _positions_frame(orders, normalized_close, nav.index)
    market_data = normalized_close.copy() if normalized_close is not None else None

    bundle_dir = Path(output_dir) if output_dir is not None else Path(report_path).resolve().parent
    dataset_dir = bundle_dir / "datasets"
    frames: dict[str, pd.DataFrame | None] = {
        "returns": returns_frame,
        "orders": orders,
        "positions": positions,
        "nav": nav_frame,
        "market_data": market_data,
    }
    frames.update(datasets or {})
    if benchmark is not None and "benchmark" not in frames:
        frames["benchmark"] = _benchmark_frame(benchmark)
    descriptors: dict[str, Any] = {}
    for name, frame in frames.items():
        if frame is None:
            continue
        _validate_dataset_columns(name, frame)
        descriptors[name] = _dataset_descriptor(frame, name, dataset_dir, inline=inline)

    bundle: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "report": _report_descriptor(report_path, bundle_dir),
        "metadata": metadata,
        "datasets": descriptors,
    }
    bundle.update(bundle_options or {})
    errors = validate_data_bundle(bundle)
    if errors:
        raise ValueError("invalid backtest bundle: " + "; ".join(errors))
    return bundle, (None if inline else dataset_dir)


def write_bundle(
    portfolio: Any,
    report_path: str | Path,
    metadata: dict[str, Any],
    output_path: str | Path,
    *,
    close: pd.DataFrame | pd.Series | None = None,
    returns_source: str = "value",
    benchmark: pd.DataFrame | pd.Series | None = None,
    datasets: dict[str, pd.DataFrame] | None = None,
    generate_report: bool = False,
    inline: bool = False,
    bundle_options: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle, dataset_dir = portfolio_to_bundle(
        portfolio,
        report_path=report_path,
        metadata=metadata,
        close=close,
        returns_source=returns_source,
        benchmark=benchmark,
        datasets=datasets,
        generate_report=generate_report,
        inline=inline,
        output_dir=output_path.parent,
        bundle_options=bundle_options,
    )
    if not inline:
        if dataset_dir is None:
            raise ValueError("dataset directory is required for non-inline bundles")
        dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output_path


def validate_data_bundle(bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {BUNDLE_SCHEMA_VERSION}")

    report = bundle.get("report")
    if not isinstance(report, dict) or not report.get("path"):
        errors.append("report.path is required")
    if not isinstance(report, dict) or report.get("format") != "quantstats_html":
        errors.append("report.format must be quantstats_html")

    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
    else:
        errors.extend(
            f"metadata.{field} is required"
            for field in REQUIRED_METADATA_FIELDS
            if field not in metadata or metadata[field] in (None, "", [], {})
        )

    datasets = bundle.get("datasets")
    if not isinstance(datasets, dict):
        errors.append("datasets must be an object")
        return errors
    unknown = set(datasets).difference(DATASET_REQUIRED_FIELDS)
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
        elif has_records:
            required = DATASET_REQUIRED_FIELDS.get(name, set())
            first = descriptor["records"][0] if descriptor["records"] else {}
            missing = sorted(required.difference(first))
            if missing:
                errors.append(f"datasets.{name}.records missing fields: {', '.join(missing)}")
    return errors


def _validate_dataset_columns(name: str, frame: pd.DataFrame) -> None:
    required = DATASET_REQUIRED_FIELDS.get(name, set())
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {', '.join(sorted(missing))}")


def _portfolio_value(portfolio: Any) -> pd.Series | None:
    value = _call(portfolio, "value")
    if value is None:
        value = _call(portfolio, "asset_value")
    if value is None:
        return None
    return _as_series(value).astype(float).sort_index()


def _portfolio_returns(portfolio: Any) -> pd.Series:
    value = _call(portfolio, "returns")
    if value is None:
        raise ValueError("portfolio does not expose returns()")
    series = _as_series(value)
    if series.empty:
        raise ValueError("portfolio returns are empty")
    return series.astype(float).sort_index()


def _orders_frame(portfolio: Any) -> pd.DataFrame | None:
    orders = _attr(portfolio, "orders")
    if orders is None:
        return None
    frame = _call(orders, "records_readable")
    if frame is None:
        raw = _attr(orders, "records")
        if raw is None:
            return None
        frame = pd.DataFrame(raw)
    frame = _normalize_columns(frame)
    frame = frame.rename(columns=ORDER_COLUMN_MAP)
    if "timestamp" not in frame.columns and "idx" in frame.columns:
        frame["timestamp"] = frame["idx"].map(_lookup_date(portfolio))
    if "symbol" not in frame.columns and "col" in frame.columns:
        columns = _columns(portfolio)
        frame["symbol"] = frame["col"].map(lambda value: str(columns[value]) if value < len(columns) else str(value))
    if "side" in frame.columns:
        frame["side"] = frame["side"].map(_normalize_side)
    required = {"timestamp", "symbol", "size", "price", "side"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"vectorbt orders are missing fields: {', '.join(sorted(missing))}")
    if "fees" not in frame.columns:
        frame["fees"] = 0.0
    if "order_id" not in frame.columns:
        frame["order_id"] = range(len(frame))
    columns = ["timestamp", "symbol", "size", "price", "side", "fees", "order_id"]
    frame = frame[columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for column in ("size", "price", "fees"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    return frame.sort_values(["timestamp", "order_id"], kind="stable").reset_index(drop=True)


def _positions_frame(
    orders: pd.DataFrame | None,
    close: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    if orders is None:
        return None
    signed = orders.copy()
    side_sign = signed["side"].map({"Buy": 1.0, "Sell": -1.0})
    if side_sign.isna().any():
        raise ValueError("orders contain unsupported sides")
    signed["quantity"] = signed["size"] * side_sign
    daily = (
        signed.groupby([signed["timestamp"].dt.normalize(), "symbol"])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )
    if daily.empty:
        return pd.DataFrame(columns=["date", "symbol", "quantity", "market_price"])
    daily["quantity"] = daily.groupby("symbol", group_keys=False)["quantity"].cumsum()
    wide = daily.pivot(index="date", columns="symbol", values="quantity").reindex(dates).fillna(0.0)
    long = wide.melt(ignore_index=False, var_name="symbol", value_name="quantity").reset_index()
    long = long[long["quantity"].abs() > 1e-12].copy()
    if long.empty:
        return pd.DataFrame(columns=["date", "symbol", "quantity", "market_price"])
    if close is not None:
        long["market_price"] = _lookup_market_price(long, close)
    else:
        last_prices = (
            orders.sort_values("timestamp")
            .groupby("symbol")["price"]
            .last()
            .to_dict()
        )
        long["market_price"] = long["symbol"].map(last_prices).astype(float)
    return long.sort_values(["date", "symbol"]).reset_index(drop=True)


def _close_frame(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.Series):
        frame = value.to_frame("close")
        frame.index.name = "date"
        frame["symbol"] = "strategy"
        return frame.reset_index()[["date", "symbol", "close"]]
    if not isinstance(value, pd.DataFrame):
        return None
    if {"date", "symbol", "close"}.issubset(value.columns):
        optional = [column for column in ("open", "high", "low") if column in value.columns]
        frame = value[["date", "symbol", "close", *optional]].copy()
    else:
        reset = value.reset_index()
        date_column = reset.columns[0]
        frame = reset.melt(id_vars=date_column, var_name="symbol", value_name="close")
        frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    for column in ("open", "high", "low"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.dropna(subset=["close"]).sort_values(["date", "symbol"])


def _lookup_market_price(positions: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    close = close.sort_values("date")
    return positions.apply(lambda row: _last_close(close, row["symbol"], row["date"]), axis=1)


def _last_close(close: pd.DataFrame, symbol: str, date: pd.Timestamp) -> float | None:
    subset = close[(close["symbol"] == symbol) & (close["date"] <= date)]
    if subset.empty:
        return None
    return float(subset.sort_values("date")["close"].iloc[-1])


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).strip().lower().replace(" ", "_").replace("-", "_") for column in frame.columns}
    return frame.rename(columns=renamed)


def _normalize_side(value: Any) -> str:
    if value in {0, "Buy", "buy", "B", "Long", "long"}:
        return "Buy"
    if value in {1, "Sell", "sell", "S", "Short", "short"}:
        return "Sell"
    raise ValueError(f"unsupported order side: {value}")


def _dataset_descriptor(
    frame: pd.DataFrame,
    name: str,
    dataset_dir: Path,
    *,
    inline: bool,
) -> dict[str, Any]:
    if inline:
        return {"records": _records(frame)}
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{name}.csv"
    frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    relative_path = path.relative_to(dataset_dir.parent).as_posix()
    return {"path": relative_path, "format": "csv"}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
            elif isinstance(value, np.generic):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return records


def _report_descriptor(report_path: Path, bundle_dir: Path) -> dict[str, Any]:
    resolved = report_path.resolve()
    try:
        path = resolved.relative_to(bundle_dir.resolve()).as_posix()
    except ValueError:
        path = resolved.as_posix()
    return {"path": path, "format": "quantstats_html"}


def _generate_quantstats_report(portfolio: Any, report_path: Path) -> None:
    try:
        import quantstats as qs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Generate report requires quantstats: pip install quantstats") from exc
    returns = _portfolio_returns(portfolio)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(returns, output=str(report_path), title="VectorBT Backtest")


def _attr(value: Any, name: str) -> Any:
    return getattr(value, name, None)


def _call(value: Any, name: str) -> Any:
    method = _attr(value, name)
    if callable(method):
        return method()
    return method


def _as_series(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.copy()
    elif isinstance(value, pd.DataFrame):
        if "strategy" in value.columns:
            series = value["strategy"].copy()
        elif value.shape[1] == 1:
            series = value.iloc[:, 0].copy()
        else:
            series = value.sum(axis=1)
    else:
        series = pd.Series(value)
    series.index = pd.to_datetime(series.index, errors="raise")
    series.index.name = series.index.name or "date"
    return series.sort_index()


def _lookup_date(portfolio: Any):
    dates = _attr(portfolio, "index")
    if dates is None:
        wrapper = _attr(portfolio, "wrapper")
        dates = _attr(wrapper, "index") if wrapper is not None else None
    if dates is None:
        return lambda value: pd.to_datetime(value)
    dates = pd.to_datetime(pd.Index(dates), errors="raise")
    return lambda value: dates[int(value)]


def _columns(portfolio: Any) -> list[Any]:
    columns = _attr(portfolio, "columns")
    if columns is None:
        wrapper = _attr(portfolio, "wrapper")
        columns = _attr(wrapper, "columns") if wrapper is not None else None
    return list(columns) if columns is not None else []


def _portfolio_close(portfolio: Any) -> Any:
    close = _attr(portfolio, "close")
    if close is None:
        return None
    return close() if callable(close) else close


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_close(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    return _close_frame(frame)


def _benchmark_frame(value: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Normalize benchmark prices or returns to date,benchmark returns."""
    if isinstance(value, pd.Series):
        index = pd.to_datetime(value.index, errors="raise")
        frame = pd.DataFrame({"date": index, "benchmark": pd.to_numeric(value, errors="coerce")})
    else:
        frame = _normalize_columns(value.copy())
        if "date" not in frame.columns:
            frame = frame.reset_index().rename(columns={"index": "date"})
        if "benchmark" not in frame.columns:
            value_columns = [column for column in frame.columns if column != "date"]
            if len(value_columns) != 1:
                raise ValueError("benchmark dataset must have a 'benchmark' column or exactly one value column")
            frame = frame.rename(columns={value_columns[0]: "benchmark"})
        frame = pd.DataFrame({
            "date": pd.to_datetime(frame["date"], errors="raise"),
            "benchmark": pd.to_numeric(frame["benchmark"], errors="coerce"),
        })
    values = frame["benchmark"]
    if float(values.abs().quantile(0.95)) > 0.5:
        frame["benchmark"] = values.pct_change(fill_method=None).fillna(0.0)
    return (
        frame.dropna(subset=["date", "benchmark"])
        .sort_values("date", kind="stable")
        .reset_index(drop=True)[["date", "benchmark"]]
    )


def _load_benchmark(path: Path) -> pd.DataFrame:
    return _benchmark_frame(pd.read_csv(path, comment="#"))


def _load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _datasets(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for item in args.dataset:
        if "=" not in item:
            raise ValueError("--dataset must use NAME=PATH, for example returns=returns.csv")
        name, path = item.split("=", 1)
        if name not in DATASET_REQUIRED_FIELDS:
            raise ValueError(f"unknown dataset name: {name}")
        datasets[name] = _load_dataset(path)
    return datasets


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if args.metadata:
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    overrides = {
        "strategy_id": args.strategy_id,
        "strategy_version": args.strategy_version,
        "data_version": args.data_version,
        "universe": args.universe.split(",") if args.universe else None,
        "parameters": json.loads(args.parameters) if args.parameters else None,
        "transaction_cost": json.loads(args.transaction_cost) if args.transaction_cost else None,
        "rebalance_rule": args.rebalance_rule,
    }
    metadata.update({key: value for key, value in overrides.items() if value is not None})
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a pickled vectorbt Portfolio to a QRA backtest bundle")
    parser.add_argument("--portfolio", required=True, help="Pickle file containing a vectorbt Portfolio")
    parser.add_argument("--report", required=True, help="QuantStats HTML report path")
    parser.add_argument("--generate-report", action="store_true", help="Create the report with quantstats if it does not exist")
    parser.add_argument("--output", required=True, help="Output bundle JSON path")
    parser.add_argument("--close", help="Optional wide or long close-price CSV")
    parser.add_argument("--benchmark", help="Optional benchmark returns CSV with date,benchmark")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional process dataset in NAME=PATH form; repeat for returns, orders, positions, nav, benchmark, market_data, corporate_actions",
    )
    parser.add_argument("--returns-source", choices=["value", "portfolio_returns"], default="value")
    parser.add_argument("--inline", action="store_true", help="Embed datasets in JSON instead of writing CSV files")
    parser.add_argument("--metadata", help="Optional complete metadata JSON file")
    parser.add_argument("--strategy-id")
    parser.add_argument("--strategy-version")
    parser.add_argument("--data-version")
    parser.add_argument("--universe", help="Comma-separated instrument symbols")
    parser.add_argument("--parameters", help="JSON object of strategy parameters")
    parser.add_argument("--transaction-cost", help="JSON object of transaction costs")
    parser.add_argument("--rebalance-rule")
    parser.add_argument("--bundle-id")
    parser.add_argument("--engine", default="vectorbt")
    parser.add_argument("--engine-version")
    parser.add_argument("--currency", default="CNY")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--calendar")
    return parser


def main() -> None:
    args = _parser().parse_args()
    portfolio = _load_pickle(Path(args.portfolio))
    close = _load_close(Path(args.close) if args.close else None)
    datasets = _datasets(args)
    metadata = _metadata(args)
    bundle_options = {
        "bundle_id": args.bundle_id,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "currency": args.currency,
        "timezone": args.timezone,
        "calendar": args.calendar,
    }
    bundle_options = {key: value for key, value in bundle_options.items() if value is not None}
    output_path = write_bundle(
        portfolio,
        args.report,
        metadata,
        args.output,
        close=close,
        returns_source=args.returns_source,
        benchmark=_load_benchmark(Path(args.benchmark)) if args.benchmark else None,
        datasets=datasets,
        generate_report=args.generate_report,
        inline=args.inline,
        bundle_options=bundle_options,
    )
    derived = {"benchmark"} if args.benchmark else set()
    dataset_names = ", ".join(sorted(set(datasets) | derived)) if datasets or derived else "portfolio-derived"
    print(f"Bundle written to {output_path}")
    print(f"Datasets: {dataset_names}")


if __name__ == "__main__":
    main()


