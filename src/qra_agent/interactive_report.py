from __future__ import annotations

"""Build a self-contained interactive HTML analysis from review state."""

from collections import deque
import json
import math
from typing import Any

import pandas as pd
import numpy as np


TRADING_DAYS_PER_YEAR = 252


def _date_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _number(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def _periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return TRADING_DAYS_PER_YEAR
    median_days = float(pd.Series((index[1:] - index[:-1]).days).median())
    if median_days <= 1.5:
        return 252
    if median_days <= 8:
        return 52
    if median_days <= 31:
        return 12
    return 1


def _equity_curve(returns: pd.Series) -> list[dict[str, Any]]:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1.0 + clean).cumprod()
    return [{"date": _date_text(date), "value": _number(value, 6)} for date, value in equity.items()]


def _drawdown(returns: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    points = [{"date": _date_text(date), "value": _number(value, 6)} for date, value in drawdown.items()]
    trough_date = drawdown.idxmin()
    peak_date = equity.loc[:trough_date].idxmax()
    recovery = drawdown.loc[trough_date:]
    recovery = recovery[recovery >= -1e-12]
    summary = {
        "max_drawdown": _number(float(drawdown.min())),
        "peak_date": _date_text(peak_date),
        "trough_date": _date_text(trough_date),
        "recovery_date": _date_text(recovery.index[0]) if len(recovery) else None,
        "current_drawdown": _number(float(drawdown.iloc[-1])),
    }
    return points, summary


def _metric_block(returns: pd.Series, periods_per_year: int) -> dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1.0 + clean).cumprod()
    years = max(1e-12, len(clean) / periods_per_year)
    volatility = float(clean.std(ddof=0) * (periods_per_year**0.5)) if len(clean) > 1 else 0.0
    downside = clean[clean < 0]
    downside_volatility = float(downside.std(ddof=0) * (periods_per_year**0.5)) if len(downside) else 0.0
    mean_return = float(clean.mean() * periods_per_year)
    return {
        "start_date": _date_text(clean.index[0]) if len(clean) else None,
        "end_date": _date_text(clean.index[-1]) if len(clean) else None,
        "cumulative_return": _number(float(equity.iloc[-1] - 1.0)),
        "cagr": _number(float(equity.iloc[-1] ** (1.0 / years) - 1.0)),
        "annual_volatility": _number(volatility),
        "sharpe_ratio": _number(mean_return / volatility) if volatility else None,
        "sortino_ratio": _number(mean_return / downside_volatility) if downside_volatility else None,
    }


def _series_by_name(state: dict[str, Any], bundle: dict[str, Any] | None) -> dict[str, pd.Series]:
    frames: dict[str, pd.Series] = {}
    returns_frame = state.get("returns_frame")
    if returns_frame is not None and not getattr(returns_frame, "empty", True):
        for column in ("strategy", "benchmark"):
            if column in returns_frame.columns:
                series = pd.to_numeric(returns_frame[column], errors="coerce").dropna()
                if not series.empty:
                    frames[column] = series

    if "strategy" not in frames and state.get("returns_path"):
        from .metrics import load_returns

        returns_frame = load_returns(state["returns_path"])
        if returns_frame is not None and not returns_frame.empty:
            for column in ("strategy", "benchmark"):
                if column in returns_frame.columns:
                    series = pd.to_numeric(returns_frame[column], errors="coerce").dropna()
                    if not series.empty:
                        frames[column] = series

    if bundle and "strategy" not in frames:
        from .data_bundle import bundle_dataset_frame

        returns_dataset = bundle_dataset_frame(bundle, "returns")
        if returns_dataset is not None and not returns_dataset.empty:
            strategy = pd.to_numeric(returns_dataset["strategy"], errors="coerce").dropna()
            dates = pd.to_datetime(returns_dataset.loc[strategy.index, "date"])
            frames["strategy"] = pd.Series(strategy.values, index=dates).sort_index()

    if bundle and "benchmark" not in frames:
        from .data_bundle import benchmark_to_returns, bundle_dataset_frame

        benchmark_frame = bundle_dataset_frame(bundle, "benchmark")
        if benchmark_frame is not None and not benchmark_frame.empty:
            values = pd.to_numeric(benchmark_frame["benchmark"], errors="coerce").dropna()
            dates = pd.to_datetime(benchmark_frame.loc[values.index, "date"])
            frames["benchmark"] = benchmark_to_returns(pd.Series(values.values, index=dates)).sort_index()
    return frames


def _nav_analysis(state: dict[str, Any], bundle: dict[str, Any] | None) -> dict[str, Any]:
    frames = _series_by_name(state, bundle)
    if "strategy" not in frames:
        return {"available": False, "reason": "returns data is unavailable"}
    all_index = frames["strategy"].index
    for series in frames.values():
        all_index = all_index.union(series.index)
    aligned = {name: series.reindex(all_index, fill_value=0.0) for name, series in frames.items()}
    periods_per_year = _periods_per_year(all_index)
    return {
        "available": True,
        "periods_per_year": periods_per_year,
        "series": {name: {"metrics": _metric_block(series, periods_per_year)} for name, series in aligned.items()},
        "nav": {name: _equity_curve(series) for name, series in aligned.items()},
    }


def _drawdown_analysis(state: dict[str, Any], bundle: dict[str, Any] | None) -> dict[str, Any]:
    frames = _series_by_name(state, bundle)
    if "strategy" not in frames:
        return {"available": False, "reason": "returns data is unavailable"}
    return {
        "available": True,
        "series": {name: dict(zip(("points", "summary"), _drawdown(series))) for name, series in frames.items()},
    }


def _load_market_prices(bundle: dict[str, Any] | None, orders: pd.DataFrame | None) -> pd.DataFrame | None:
    """Load held-symbol prices and optional OHLC, avoiding a full market_data load."""
    if bundle is None or orders is None or orders.empty:
        return None
    from .data_bundle import _resolve_bundle_path, bundle_dataset_frame

    descriptor = bundle.get("datasets", {}).get("market_data")
    if descriptor is None:
        return None
    start = pd.to_datetime(orders["timestamp"]).min().normalize()
    nav_frame = bundle_dataset_frame(bundle, "nav")
    end = pd.to_datetime(nav_frame["date"]).max() if nav_frame is not None and not nav_frame.empty else pd.to_datetime(orders["timestamp"]).max()
    symbols = set(orders["symbol"].astype(str))
    path = _resolve_bundle_path(bundle, descriptor["path"])
    frames: list[pd.DataFrame] = []
    if path.suffix.lower() == ".csv":
        chunks = pd.read_csv(
            path,
            usecols=lambda column: column.strip().lower() in {"date", "symbol", "open", "high", "low", "close"},
            chunksize=250_000,
        )
        for chunk in chunks:
            chunk.columns = [column.strip().lower() for column in chunk.columns]
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk["close"] = pd.to_numeric(chunk["close"], errors="coerce")
            chunk = chunk.dropna().query("date >= @start and date <= @end and symbol in @symbols")
            if not chunk.empty:
                frames.append(chunk)
    else:
        frame = bundle_dataset_frame(bundle, "market_data")
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    available = [column for column in ("date", "symbol", "open", "high", "low", "close") if column in combined.columns]
    frame = combined[available].copy().rename(columns={"close": "price"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    for column in ("open", "high", "low"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().query("date >= @start and date <= @end and symbol in @symbols")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _current_prices(state: dict[str, Any], bundle: dict[str, Any] | None, market_prices: pd.DataFrame | None) -> dict[str, float]:
    prices: dict[str, float] = {}
    if market_prices is not None and not market_prices.empty:
        ordered = market_prices.sort_values("date").dropna(subset=["price"])
        prices.update(ordered.groupby("symbol", sort=False)["price"].last().astype(float).to_dict())
    if bundle:
        from .data_bundle import bundle_dataset_frame

        positions = bundle_dataset_frame(bundle, "positions")
        if positions is not None and not positions.empty:
            ordered = positions.sort_values("date").dropna(subset=["market_price"])
            prices.update(ordered.groupby("symbol", sort=False)["market_price"].last().astype(float).to_dict())
    if not prices and state.get("orders_path"):
        from .orders import load_orders

        orders = load_orders(state["orders_path"])
        if orders is not None and not orders.empty:
            ordered = orders.sort_values("timestamp")
            prices.update(ordered.groupby("symbol", sort=False)["price"].last().astype(float).to_dict())
    return prices


def _symbol_price_analysis(market_prices: pd.DataFrame | None, orders: pd.DataFrame | None, timeline: dict[str, Any]) -> dict[str, Any]:
    """Build compact per-symbol price series for the interactive detail chart."""
    symbols = timeline.get("symbols", []) if timeline.get("available") else []
    if not symbols:
        return {"available": False, "reason": "timeline data is unavailable", "symbols": {}}
    if market_prices is None or market_prices.empty:
        if orders is None or orders.empty:
            return {"available": False, "reason": "market price data is unavailable", "symbols": {}}
        market_prices = pd.DataFrame({
            "date": pd.to_datetime(orders["timestamp"]).dt.normalize(),
            "symbol": orders["symbol"].astype(str),
            "price": pd.to_numeric(orders["price"], errors="coerce"),
        })

    symbols_out: dict[str, Any] = {}
    for row in symbols:
        symbol = row["symbol"]
        subset = market_prices[
            (market_prices["symbol"] == symbol)
            & (market_prices["date"] >= pd.to_datetime(row["first_date"]))
            & (market_prices["date"] <= pd.to_datetime(row["last_date"]))
        ].sort_values("date").drop_duplicates("date", keep="last")
        if subset.empty:
            continue
        if len(subset) > 240:
            positions = np.unique(np.linspace(0, len(subset) - 1, 240, dtype=int))
            subset = subset.iloc[positions]
        candles = {"open", "high", "low", "price"}.issubset(subset.columns)
        points: list[dict[str, Any]] = []
        for item in subset.itertuples(index=False):
            point: dict[str, Any] = {
                "date": _date_text(item.date),
                "price": _number(getattr(item, "price"), 6),
            }
            if candles:
                point.update({
                    "open": _number(item.open, 6),
                    "high": _number(item.high, 6),
                    "low": _number(item.low, 6),
                        "close": _number(item.price, 6),
                })
            points.append(point)
        symbols_out[symbol] = {"candles": candles, "points": points}
    return {
        "available": bool(symbols_out),
        "reason": "" if symbols_out else "market price data is unavailable",
        "symbols": symbols_out,
    }
def _orders_with_actual_sizes(orders: pd.DataFrame, positions: pd.DataFrame | None) -> pd.DataFrame:
    """Convert percentage order sizes to actual quantities using position snapshots."""
    orders = orders.copy()
    if positions is None or positions.empty:
        return orders
    snapshot = positions.copy()
    snapshot["date"] = pd.to_datetime(snapshot["date"]).dt.normalize()
    snapshot = snapshot.sort_values("date").groupby(["date", "symbol"], as_index=False)["quantity"].last()
    quantities = snapshot.set_index(["date", "symbol"])["quantity"].astype(float).to_dict()
    sort_columns = ["symbol", "timestamp"]
    if "order_id" in orders.columns:
        sort_columns.append("order_id")
    normalized = orders.sort_values(sort_columns, kind="stable")
    actual_sizes: list[float] = []
    for symbol, group in normalized.groupby("symbol", sort=False):
        previous_quantity = 0.0
        for date, rows in group.groupby(pd.to_datetime(group["timestamp"]).dt.normalize(), sort=True):
            quantity_after = float(quantities.get((date, symbol), 0.0))
            quantity_delta = quantity_after - previous_quantity
            requested = float(rows["size"].abs().sum())
            scale = abs(quantity_delta) / requested if requested else 0.0
            actual_sizes.extend((float(row.size) * scale for row in rows.itertuples(index=False)))
            previous_quantity = quantity_after
    normalized["size"] = actual_sizes
    final_sort_columns = ["timestamp"]
    if "order_id" in orders.columns:
        final_sort_columns.append("order_id")
    orders = normalized.sort_values(final_sort_columns, kind="stable").reset_index(drop=True)
    return orders


def _position_intervals(orders: pd.DataFrame, round_trips: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    realized_by_sell: dict[tuple[str, str], float] = {}
    for trade in round_trips or []:
        key = (trade["symbol"], trade["close_date"])
        realized_by_sell[key] = realized_by_sell.get(key, 0.0) + float(trade["net_pnl"])
    columns = ["symbol", "timestamp"]
    if "order_id" in orders.columns:
        columns.append("order_id")
    for symbol, group in orders.sort_values(columns, kind="stable").groupby("symbol", sort=False):
        quantity = 0.0
        previous = None
        active_intervals: list[int] = []
        for row in group.itertuples(index=False):
            if previous is not None and abs(quantity) > 1e-12:
                active_intervals.append(len(intervals))
                intervals.append({
                    "symbol": symbol,
                    "start": _date_text(previous["timestamp"]),
                    "end": _date_text(row.timestamp),
                    "position_after": _number(quantity, 6),
                    "side": "Buy" if quantity > 0 else "Sell",
                })
            quantity += row.size if row.side == "Buy" else -row.size
            if row.side == "Sell" and abs(quantity) <= 1e-12:
                closed_pnl = realized_by_sell.get((symbol, _date_text(row.timestamp)))
                if closed_pnl is not None:
                    for interval_index in active_intervals:
                        intervals[interval_index]["closed_pnl"] = _number(closed_pnl, 2)
                active_intervals = []
            previous = {"timestamp": row.timestamp}
        if previous is not None and abs(quantity) > 1e-12:
                intervals.append({
                    "symbol": symbol,
                    "start": _date_text(previous["timestamp"]),
                    "end": _date_text(group["timestamp"].max()),
                    "position_after": _number(quantity, 6),
                    "side": "Buy" if quantity > 0 else "Sell",
                })
    return intervals


def _fifo_records(orders: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    open_lots: dict[str, deque[dict[str, Any]]] = {}
    round_trips: list[dict[str, Any]] = []
    unmatched_sells: list[dict[str, Any]] = []
    columns = ["symbol", "timestamp"]
    if "order_id" in orders.columns:
        columns.append("order_id")
    for row in orders.sort_values(columns, kind="stable").itertuples(index=False):
        if row.side == "Buy":
            open_lots.setdefault(row.symbol, deque()).append({
                "open_date": row.timestamp,
                "quantity": float(row.size),
                "remaining_quantity": float(row.size),
                "price": float(row.price),
                "fee": float(row.fees),
            })
            continue
        remaining = float(row.size)
        sell_fee = float(row.fees)
        while remaining > 1e-12 and open_lots.get(row.symbol):
            lot = open_lots[row.symbol][0]
            if lot["remaining_quantity"] <= 1e-12:
                open_lots[row.symbol].popleft()
                continue
            matched = min(remaining, lot["remaining_quantity"])
            cost_basis = matched * lot["price"]
            sell_notional = matched * row.price
            buy_fee = lot["fee"] * matched / lot["quantity"]
            allocated_sell_fee = sell_fee * matched / float(row.size) if row.size else 0.0
            net_pnl = sell_notional - cost_basis - buy_fee - allocated_sell_fee
            round_trips.append({
                "symbol": row.symbol,
                "open_date": _date_text(lot["open_date"]),
                "close_date": _date_text(row.timestamp),
                "quantity": _number(matched, 6),
                "buy_price": _number(lot["price"], 6),
                "sell_price": _number(float(row.price), 6),
                "cost_basis": _number(cost_basis, 2),
                "sell_notional": _number(sell_notional, 2),
                "fees": _number(buy_fee + allocated_sell_fee, 2),
                "net_pnl": _number(net_pnl, 2),
                "return_on_cost": _number(net_pnl / cost_basis if cost_basis else 0.0),
                "holding_days": float((row.timestamp - lot["open_date"]).days),
            })
            lot["remaining_quantity"] -= matched
            remaining -= matched
        if remaining > 1e-12:
            unmatched_sells.append({
                "symbol": row.symbol,
                "date": _date_text(row.timestamp),
                "quantity": _number(remaining, 6),
            })
    open_positions: list[dict[str, Any]] = []
    for symbol, lots in open_lots.items():
        for lot in lots:
            if lot["remaining_quantity"] <= 1e-12:
                continue
            share = lot["remaining_quantity"] / lot["quantity"]
            open_positions.append({
                "symbol": symbol,
                "open_date": _date_text(lot["open_date"]),
                "quantity": _number(lot["remaining_quantity"], 6),
                "price": _number(lot["price"], 6),
                "cost_basis": _number(lot["remaining_quantity"] * lot["price"] + lot["fee"] * share, 2),
            })
    return round_trips, unmatched_sells, open_positions


def _timeline_analysis(orders: pd.DataFrame, prices: dict[str, float]) -> dict[str, Any]:
    round_trips, _, open_positions = _fifo_records(orders)
    events = [
        {
            "date": _date_text(row.timestamp),
            "symbol": row.symbol,
            "side": row.side,
            "quantity": _number(row.size, 6),
            "price": _number(row.price, 6),
            "notional": _number(row.size * row.price, 2),
            "fees": _number(row.fees, 2),
        }
        for row in orders.itertuples(index=False)
    ]
    grouped: dict[str, dict[str, Any]] = {}
    round_trip_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for trade in round_trips:
        round_trip_by_symbol.setdefault(trade["symbol"], []).append(trade)
    for symbol, group in orders.groupby("symbol", sort=False):
        buys = group[group["side"] == "Buy"]
        sells = group[group["side"] == "Sell"]
        symbol_trades = round_trip_by_symbol.get(symbol, [])
        symbol_open = [item for item in open_positions if item["symbol"] == symbol]
        open_quantity = float(sum(item["quantity"] for item in symbol_open))
        open_cost = float(sum(item["cost_basis"] for item in symbol_open))
        current_price = prices.get(symbol)
        unrealized_pnl = open_quantity * current_price - open_cost if current_price is not None and open_quantity else None
        grouped[symbol] = {
            "symbol": symbol,
            "first_date": _date_text(group["timestamp"].min()),
            "last_date": _date_text(group["timestamp"].max()),
            "buy_count": int(len(buys)),
            "sell_count": int(len(sells)),
            "buy_quantity": _number(float(buys["size"].sum())),
            "sell_quantity": _number(float(sells["size"].sum())),
            "buy_notional": _number(float((buys["size"] * buys["price"]).sum()), 2),
            "sell_notional": _number(float((sells["size"] * sells["price"]).sum()), 2),
            "fees": _number(float(group["fees"].sum()), 2),
            "realized_pnl": _number(sum(float(trade["net_pnl"]) for trade in symbol_trades), 2),
            "closed_trade_count": len(symbol_trades),
            "open_quantity": _number(open_quantity, 6),
            "open_cost_basis": _number(open_cost, 2),
            "current_price": _number(current_price, 6),
            "unrealized_pnl": _number(unrealized_pnl, 2),
            "fully_closed": open_quantity <= 1e-12,
        }
    return {
        "available": True,
        "start_date": _date_text(orders["timestamp"].min()),
        "end_date": _date_text(orders["timestamp"].max()),
        "symbols": list(grouped.values()),
        "events": events,
        "intervals": _position_intervals(orders, round_trips),
        "trades": round_trips,
        "open_positions": open_positions,
    }


def _position_weight_analysis(bundle: dict[str, Any] | None, orders: pd.DataFrame | None, market_prices: pd.DataFrame | None) -> dict[str, Any]:
    if bundle is None or orders is None or orders.empty:
        return {"available": False, "reason": "orders or bundle data is unavailable"}
    from .data_bundle import bundle_dataset_frame

    nav_frame = bundle_dataset_frame(bundle, "nav")
    if nav_frame is None or nav_frame.empty:
        returns_frame = bundle_dataset_frame(bundle, "returns")
        if returns_frame is None or returns_frame.empty:
            return {"available": False, "reason": "nav data is unavailable"}
        dates = pd.DatetimeIndex(pd.to_datetime(returns_frame["date"])).sort_values()
        nav = pd.Series(1.0, index=dates)
    else:
        nav_frame = nav_frame.sort_values("date")
        dates = pd.DatetimeIndex(pd.to_datetime(nav_frame["date"]))
        nav = pd.Series(pd.to_numeric(nav_frame["nav"], errors="coerce").values, index=dates).ffill().bfill()

    signed = orders.copy()
    signed["quantity"] = signed["size"] * signed["side"].map({"Buy": 1.0, "Sell": -1.0})
    daily = signed.groupby([pd.to_datetime(signed["timestamp"]).dt.normalize(), "symbol"])["quantity"].sum()
    quantity_wide = daily.groupby(level="symbol").cumsum().unstack("symbol").sort_index()
    quantity_wide = quantity_wide.reindex(dates).ffill().fillna(0.0)

    positions = bundle_dataset_frame(bundle, "positions")
    price_frames: list[pd.DataFrame] = []
    if positions is not None and not positions.empty:
        price_frames.append(positions.pivot(index="date", columns="symbol", values="market_price"))
    if market_prices is not None and not market_prices.empty:
        price_frames.append(market_prices.pivot(index="date", columns="symbol", values="price"))
    if price_frames:
        price_wide = pd.concat(price_frames).sort_index()
        price_wide = price_wide.groupby(level=0).last()
        price_wide = price_wide.loc[:, ~price_wide.columns.duplicated()]
        price_wide = price_wide.reindex(dates).ffill().bfill()
    else:
        last_prices = orders.sort_values("timestamp").groupby("symbol")["price"].last()
        price_wide = pd.DataFrame({symbol: last_prices.get(symbol) for symbol in quantity_wide.columns}, index=dates)

    value = quantity_wide * price_wide.reindex(columns=quantity_wide.columns).fillna(0.0)
    value = value.fillna(0.0)
    weights = value.div(nav.abs().replace(0.0, pd.NA), axis=0).fillna(0.0).clip(lower=0.0)
    total = weights.sum(axis=1)
    symbols = sorted(weights.columns, key=lambda column: (-float(weights[column].max()), str(column)))
    rows = []
    for date, row in weights.iterrows():
        nonzero = {symbol: _number(float(row[symbol]), 6) for symbol in symbols if abs(float(row[symbol])) > 1e-8}
        rows.append({
            "date": _date_text(date),
            "total": _number(float(total.loc[date]), 6),
            "holdings": int((row.abs() > 1e-8).sum()),
            "weights": nonzero,
        })
    return {
        "available": True,
        "symbols": symbols,
        "rows": rows,
        "summary": {
            "average_weight": _number(float(total.mean())),
            "max_weight": _number(float(total.max())),
            "max_weight_date": _date_text(total.idxmax()) if not total.empty else None,
            "final_weight": _number(float(total.iloc[-1])) if len(total) else None,
            "average_symbol_count": _number(float((value.abs() > 1e-8).sum(axis=1).mean()), 1),
            "max_symbol_count": int((value.abs() > 1e-8).sum(axis=1).max()) if len(value) else 0,
        },
    }


def _daily_pnl_analysis(state: dict[str, Any], bundle: dict[str, Any] | None) -> dict[str, Any]:
    nav = None
    if bundle:
        from .data_bundle import bundle_dataset_frame

        nav_frame = bundle_dataset_frame(bundle, "nav")
        if nav_frame is not None and not nav_frame.empty:
            nav_frame = nav_frame.sort_values("date")
            nav = pd.Series(
                pd.to_numeric(nav_frame["nav"], errors="coerce").values,
                index=pd.to_datetime(nav_frame["date"]),
            ).dropna()
            pnl = nav.diff()
            if len(pnl):
                pnl.iloc[0] = 0.0

    if nav is None:
        frames = _series_by_name(state, bundle)
        if "strategy" not in frames:
            return {"available": False, "reason": "returns or nav data is unavailable"}
        returns = frames["strategy"].fillna(0.0)
        equity = (1.0 + returns).cumprod()
        pnl = equity.diff()
        if len(pnl):
            pnl.iloc[0] = float(equity.iloc[0] - 1.0)

    clean = pnl.dropna()
    mean = float(clean.mean()) if len(clean) else 0.0
    std = float(clean.std(ddof=0)) if len(clean) > 1 else 0.0
    points = [{"date": _date_text(date), "value": _number(value, 2)} for date, value in clean.items()]
    sigma_counts = {
        str(int(multiple)): int((clean.abs() > std * multiple).sum()) if std else 0
        for multiple in (1.0, 2.0, 3.0, 4.0, 5.0)
    }
    return {
        "available": True,
        "points": points,
        "mean": _number(mean, 2),
        "std": _number(std, 2),
        "sigma_counts": sigma_counts,
    }


def _concentration_analysis(timeline: dict[str, Any]) -> dict[str, Any]:
    if not timeline.get("available"):
        return {"available": False, "reason": "orders data is unavailable"}
    trade_items = []
    for trade in timeline["trades"]:
        pnl = float(trade.get("net_pnl") or 0.0)
        trade_items.append({
            "label": f"{trade['symbol']} {trade['close_date']}",
            "symbol": trade["symbol"],
            "pnl": _number(pnl, 2),
            "holding_days": trade.get("holding_days"),
        })
    symbol_items = []
    for symbol in timeline["symbols"]:
        pnl = float(symbol.get("realized_pnl") or 0.0)
        if float(symbol.get("open_quantity") or 0.0) > 0 and symbol.get("unrealized_pnl") is not None:
            pnl += float(symbol["unrealized_pnl"])
        symbol_items.append({"label": symbol["symbol"], "symbol": symbol["symbol"], "pnl": _number(pnl, 2)})

    result: dict[str, Any] = {"available": True, "items": {}}
    for kind, items in (("trade", trade_items), ("symbol", symbol_items)):
        ranked = sorted(items, key=lambda item: -abs(float(item.get("pnl") or 0.0)))
        absolute_total = sum(abs(float(item.get("pnl") or 0.0)) for item in ranked)
        cumulative = 0.0
        for rank, item in enumerate(ranked, start=1):
            absolute = abs(float(item.get("pnl") or 0.0))
            share = absolute / absolute_total if absolute_total else 0.0
            cumulative += share
            item["rank"] = rank
            item["share"] = _number(share, 6)
            item["cumulative_share"] = _number(cumulative, 6)
        result["items"][kind] = ranked
    positive = sum(float(item.get("pnl") or 0.0) > 0 for item in symbol_items)
    return {
        **result,
        "metrics": {
            "trade_top_1": _number(result["items"]["trade"][0]["cumulative_share"]) if trade_items else None,
            "trade_top_5": _number(result["items"]["trade"][4]["cumulative_share"]) if len(trade_items) >= 5 else None,
            "trade_top_10": _number(result["items"]["trade"][9]["cumulative_share"]) if len(trade_items) >= 10 else None,
            "positive_symbol_count": positive,
            "negative_symbol_count": len(symbol_items) - positive,
        },
    }


def _position_analysis(timeline: dict[str, Any], weights: dict[str, Any]) -> dict[str, Any]:
    if not timeline.get("available"):
        return {"available": False, "reason": "orders data is unavailable", "weights": weights}
    symbols = timeline["symbols"]
    open_symbols = [row for row in symbols if float(row.get("open_quantity") or 0.0) > 0]
    realized = [float(row.get("realized_pnl") or 0.0) for row in symbols]
    unrealized = [float(row.get("unrealized_pnl") or 0.0) for row in open_symbols]
    return {
        "available": True,
        "weights": weights,
        "summary": {
            "symbol_count": len(symbols),
            "open_symbol_count": len(open_symbols),
            "closed_trade_count": sum(int(row.get("closed_trade_count") or 0) for row in symbols),
            "realized_pnl": _number(sum(realized), 2),
            "unrealized_pnl": _number(sum(unrealized), 2) if unrealized else None,
            "open_cost_basis": _number(sum(float(row.get("open_cost_basis") or 0.0) for row in open_symbols), 2),
        },
    }


def build_interactive_data(state: dict[str, Any]) -> dict[str, Any]:
    """Create the payload used by the standalone interactive analysis page."""
    bundle = state.get("bundle")
    metadata = state.get("metadata") or {}
    report = state.get("report") or {}
    orders_frame = state.get("orders_frame")
    has_orders = orders_frame is not None and not getattr(orders_frame, "empty", True)
    market_prices = _load_market_prices(bundle, orders_frame)
    prices = _current_prices(state, bundle, market_prices)
    if bundle:
        from .data_bundle import bundle_dataset_frame

        positions = bundle_dataset_frame(bundle, "positions")
    else:
        positions = None
    orders_frame = _orders_with_actual_sizes(orders_frame, positions) if has_orders else orders_frame
    timeline = _timeline_analysis(orders_frame, prices) if has_orders else {"available": False, "reason": "orders data is unavailable"}
    daily_weights = _position_weight_analysis(bundle, orders_frame, market_prices)
    return {
        "schema_version": "qra.interactive_analysis/v1",
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "meta": {
            "title": str(metadata.get("strategy_id") or report.get("strategy_id") or "Interactive Analysis"),
            "strategy_id": metadata.get("strategy_id"),
            "strategy_version": metadata.get("strategy_version"),
            "benchmark": report.get("benchmark"),
        },
        "nav_analysis": _nav_analysis(state, bundle),
        "drawdown_analysis": _drawdown_analysis(state, bundle),
        "position_analysis": _position_analysis(timeline, daily_weights),
        "daily_pnl": _daily_pnl_analysis(state, bundle),
        "concentration": _concentration_analysis(timeline),
        "timeline": timeline,
        "symbol_prices": _symbol_price_analysis(market_prices, orders_frame, timeline),
        "bundle": {
            "schema_version": bundle.get("schema_version"),
            "bundle_id": bundle.get("bundle_id"),
            "datasets": sorted(bundle.get("datasets", {})) if bundle else [],
        } if bundle else None,
    }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · 互动分析</title>
<style>
:root{--bg:#f1f5f9;--panel:#fff;--text:#0f172a;--muted:#64748b;--line:#e2e8f0;--blue:#2563eb;--green:#16a34a;--red:#dc2626}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,"Microsoft YaHei",system-ui,sans-serif}
.layout{max-width:1560px;margin:auto;padding:18px}.hero{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
h1,h2,h3{margin:0}h1{font-size:24px}h2{font-size:18px;margin-bottom:10px}a{color:var(--blue);text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.col-12{grid-column:span 12}.col-8{grid-column:span 8}.col-4{grid-column:span 4}.col-6{grid-column:span 6}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 4px 16px #0f172a08}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin-bottom:12px}
.card{border:1px solid var(--line);border-radius:9px;padding:9px 10px;background:#f8fafc}.label{font-size:11px;color:var(--muted)}.value{font-size:17px;font-weight:650;white-space:nowrap}.delta.pos,.pos{color:var(--green)}.delta.neg,.neg{color:var(--red)}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}.btn,.select,input{border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 8px;font:inherit}.btn{cursor:pointer}.btn:hover{background:#eff6ff}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:99px;margin-right:4px}
.chart{border:1px solid var(--line);border-radius:9px;background:#fff;overflow:hidden}.chart svg{display:block;width:100%;cursor:grab}.chart svg.dragging{cursor:grabbing}.overview{margin-top:8px;border:1px solid var(--line);border-radius:9px;background:#f8fafc;overflow:hidden}.range-buttons{display:flex;gap:5px;padding:7px 8px;border-bottom:1px solid var(--line);background:#fff}.range-btn{border:1px solid var(--line);background:#fff;border-radius:6px;padding:3px 8px;font-size:11px;cursor:pointer}.range-btn.active,.range-btn:hover{border-color:#2563eb;color:#2563eb;background:#eff6ff}.overview svg{display:block;width:100%;height:88px;cursor:pointer}.overview .range-handle{cursor:ew-resize;filter:drop-shadow(0 1px 2px #0f172a33)}.overview .range-window{cursor:grab}.overview .range-window.dragging{cursor:grabbing}.zoom-controls{display:flex;gap:5px}.zoom-controls .btn{min-width:30px;padding:4px 7px}.zoom-band{fill:#2563eb18;stroke:#2563eb;stroke-width:1;stroke-dasharray:4 3;pointer-events:none}.subchart-title{font-size:13px;font-weight:650;margin:12px 0 8px}.hint{font-size:11px;color:var(--muted);margin-top:5px}.crosshair{pointer-events:none}.crosshair line{stroke:#ef4444;stroke-width:1.4;stroke-dasharray:5 4}.crosshair .badge{fill:#0f172a;stroke:#0f172a}.crosshair text{fill:#fff;font-size:11px}.row-selected{fill:#2563eb14}
#nav-chart,#dd-chart,#pnl-chart{height:430px}#position-chart{height:480px}#holdings-chart{height:260px}#gantt-chart{height:auto;min-height:240px}.table-scroll{overflow:auto;max-height:390px;border:1px solid var(--line);border-radius:8px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:#f8fafc;z-index:1}.muted{color:var(--muted);text-align:center!important}.empty{padding:28px;color:var(--muted);text-align:center;border:1px dashed var(--line);border-radius:9px}
#tooltip{position:fixed;z-index:99;pointer-events:none;background:#0f172ad9;color:#fff;padding:7px 9px;border-radius:7px;font-size:12px;opacity:0;max-width:360px}
.section{scroll-margin-top:16px}.footer{color:var(--muted);text-align:center;padding:18px}.tick{fill:#64748b;font-size:11px}.ref{stroke-dasharray:6 5;stroke-width:1.4}.zero{stroke:#94a3b8;stroke-width:1}
@media(max-width:1050px){.col-8,.col-4,.col-6{grid-column:span 12}.cards{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div id="tooltip"></div><div class="layout">
<div class="hero"><div><h1 id="title">__TITLE__</h1><div class="muted" id="subtitle"></div></div><a href="#" onclick="scrollTo({top:0,behavior:'smooth'});return false">返回顶部</a></div>

<section id="section-overview" class="section"><div class="panel"><div class="cards" id="overview-cards"></div><div class="legend" id="bundle-info"></div></div></section>

<section id="section-nav" class="section" style="margin-top:14px"><div class="panel">
<h2>净值分析</h2><div class="cards" id="nav-summary"></div>
<div class="toolbar"><button class="btn" data-reset="nav-chart">重置缩放</button><label><input type="checkbox" id="show-strategy" checked> 策略</label><label><input type="checkbox" id="show-benchmark" checked> Benchmark</label><span class="hint">拖拽平移 · 下方缩略图缩放 · 双击重置</span></div>
<div class="chart"><svg id="nav-chart"></svg></div></div></section>

<section id="section-drawdown" class="section" style="margin-top:14px"><div class="panel">
<h2>回撤分析</h2><div class="cards" id="dd-summary"></div>
<div class="toolbar"><button class="btn" data-reset="dd-chart">重置缩放</button><label><input type="checkbox" id="dd-strategy" checked> 策略</label><label><input type="checkbox" id="dd-benchmark" checked> Benchmark</label></div>
<div class="chart"><svg id="dd-chart"></svg></div></div></section>

<section id="section-position" class="section" style="margin-top:14px"><div class="panel">
<h2>仓位分析 · 每日市值占账户净值比例</h2><div class="cards" id="position-summary"></div>
<div class="toolbar"><button class="btn" data-reset="position-chart">重置缩放</button><select class="select" id="position-mode"><option value="stacked">堆叠面积</option><option value="total">总仓位</option></select><div class="legend" id="position-legend"></div></div>
<div class="chart"><svg id="position-chart"></svg></div><div class="hint">面积从下到上表示标的权重；鼠标悬浮显示当日主要持仓。</div>
<div class="subchart-title">&#27599;&#26085;&#25345;&#26377;&#26631;&#30340;&#25968;&#37327;</div><div class="chart"><svg id="holdings-chart"></svg></div><div class="hint">&#26609;&#39640;&#20026;&#24403;&#26085;&#25345;&#20179;&#26631;&#30340;&#25968;&#37327;&#65307;&#25903;&#25345;&#29420;&#31435;&#32553;&#25918;&#21644;&#21313;&#23383;&#32447;&#26597;&#30475;&#12290;</div></div></section>

<section id="section-pnl" class="section" style="margin-top:14px"><div class="panel">
<h2>每日盈亏与标准差</h2><div class="cards" id="pnl-summary"></div>
<div class="toolbar"><button class="btn" data-reset="pnl-chart">重置缩放</button><label><input type="checkbox" id="show-1sigma" checked> ±1σ</label><label><input type="checkbox" id="show-2sigma" checked> ±2σ</label><label><input type="checkbox" id="show-3sigma" checked> ±3σ</label><label><input type="checkbox" id="show-4sigma" checked> ±4σ</label><label><input type="checkbox" id="show-5sigma" checked> ±5σ</label></div>
<div class="chart"><svg id="pnl-chart"></svg></div></div></section>

<section id="section-concentration" class="section" style="margin-top:14px"><div class="panel">
<h2>盈亏 Top-K 集中度</h2><div class="cards" id="concentration-summary"></div>
<div class="toolbar"><select class="select" id="conc-kind"><option value="trade">单笔交易</option><option value="symbol">单标的</option></select><label>Top <input id="conc-k" type="range" min="1" max="20" value="10"></label><span id="conc-k-label" class="muted">10</span></div>
<div class="chart"><svg id="concentration-chart"></svg></div><div class="hint">按盈亏绝对值排序；柱体为单笔盈亏，黄色折线为累计占比，用于识别肥尾或长尾。</div></div></section>

<section id="section-timeline" class="section" style="margin-top:14px"><div class="panel">
<h2>标的持仓时间线</h2><div class="cards" id="timeline-summary"></div>
<div class="toolbar"><select class="select" id="timeline-symbol" title="选择标的"></select><input id="timeline-search" placeholder="搜索标的"><label><input type="checkbox" id="filter-buy" checked> 买入</label><label><input type="checkbox" id="filter-sell" checked> 卖出</label><button class="btn" data-reset="gantt-chart">重置缩放</button></div>
<div class="legend"><span><span class="dot" style="background:#16a34a"></span>已清仓 · 盈利</span><span><span class="dot" style="background:#dc2626"></span>已清仓 · 亏损</span><span><span class="dot" style="background:#2563eb"></span>仍持仓</span><span><span class="dot" style="background:#7c3aed"></span>增持段</span></div>
<div class="chart" style="margin-top:8px;max-height:720px;overflow:auto"><svg id="gantt-chart"></svg></div><div class="hint">拖拽平移 · 下方缩略图缩放 · 点击标的查看交易明细。</div>
<div id="symbol-detail" style="margin-top:12px"></div></div></section>

<div class="footer">Generated __GENERATED__ · Self-contained interactive report</div>
</div>
<script>
const data=__PAYLOAD__,D={};
function renderSymbolPrice(symbol){$('timeline-symbol').value=symbol;requestAnimationFrame(drawSymbolPrice)}
const $=id=>document.getElementById(id),dt=v=>v?new Date(v.replace(" ","T")):null,fmt=v=>v===null||v===undefined||Number.isNaN(+v)?"-":(+v).toLocaleString("en-US",{maximumFractionDigits:2}),pct=v=>v===null||v===undefined?"-":(100*v).toLocaleString("en-US",{maximumFractionDigits:2})+"%",pc=v=>+v>0?"pos":+v<0?"neg":"",money=v=>v===null||v===undefined?"-":(+v).toLocaleString("en-US",{maximumFractionDigits:0});
const show=(e,html)=>{const t=$("tooltip");t.innerHTML=html;t.style.opacity=1;const x=Math.min(e.clientX+14,innerWidth-360);t.style.left=x+"px";t.style.top=(e.clientY+16)+"px"},hide=()=>$("tooltip").style.opacity=0;
const card=(label,value,cls="")=>`<div class="card"><div class="label">${label}</div><div class="value ${cls}">${value}</div></div>`;
const PALETTE=["#2563eb","#16a34a","#dc2626","#7c3aed","#ea580c","#0891b2","#be185d","#65a30d","#7e22ce","#0f766e","#a16207","#475569"];
function hideCross(svg){svg.querySelector(".crosshair")?.remove();hide()}
function setCross(svg,e,geometry,value,format){const rect=svg.getBoundingClientRect(),px=(e.clientX-rect.left)*geometry.W/rect.width,py=(e.clientY-rect.top)*geometry.H/rect.height,x=Math.max(geometry.L,Math.min(geometry.W-geometry.R,px)),y=Math.max(geometry.T,Math.min(geometry.H-geometry.B,py)),NS="http://www.w3.org/2000/svg";let group=svg.querySelector(".crosshair");if(!group){group=document.createElementNS(NS,"g");group.setAttribute("class","crosshair");["cross-x","cross-y"].forEach(name=>{const line=document.createElementNS(NS,"line");line.setAttribute("class",name);group.appendChild(line)});["x-badge","y-badge"].forEach(name=>{const badge=document.createElementNS(NS,"rect");badge.setAttribute("class","badge "+name);badge.setAttribute("rx","5");group.appendChild(badge)});["x-text","y-text"].forEach(name=>{const text=document.createElementNS(NS,"text");text.setAttribute("class",name);if(name==="x-text")text.setAttribute("text-anchor","middle");group.appendChild(text)});svg.appendChild(group)}const time=svg._zoomState?xToTime(x,svg._zoomState,geometry.L,geometry.W,geometry.R):null,xt=time?time.toLocaleDateString("en-CA"):"-",yt=format(value),xb=Math.max(geometry.L,Math.min(geometry.W-geometry.R-104,x-52)),yb=Math.max(geometry.T+2,Math.min(geometry.H-geometry.B-24,y-11));const parts={xe:group.querySelector(".cross-x"),ye:group.querySelector(".cross-y"),xrect:group.querySelector(".x-badge"),xtext:group.querySelector(".x-text"),yrect:group.querySelector(".y-badge"),ytext:group.querySelector(".y-text")};parts.xe.setAttribute("x1",x);parts.xe.setAttribute("x2",x);parts.xe.setAttribute("y1",geometry.T);parts.xe.setAttribute("y2",geometry.H-geometry.B);parts.ye.setAttribute("x1",geometry.L);parts.ye.setAttribute("x2",geometry.W-geometry.R);parts.ye.setAttribute("y1",y);parts.ye.setAttribute("y2",y);parts.xrect.setAttribute("x",xb);parts.xrect.setAttribute("y",geometry.H-geometry.B-28);parts.xrect.setAttribute("width",104);parts.xrect.setAttribute("height",22);parts.xtext.setAttribute("x",xb+52);parts.xtext.setAttribute("y",geometry.H-geometry.B-13);parts.xtext.textContent=xt;parts.yrect.setAttribute("x",2);parts.yrect.setAttribute("y",yb);parts.yrect.setAttribute("width",78);parts.yrect.setAttribute("height",22);parts.ytext.setAttribute("x",6);parts.ytext.setAttribute("y",yb+15);parts.ytext.textContent=yt;return{x:time,y:value}}
function xToTime(px,state,L,W,R=16){const ratio=Math.max(0,Math.min(1,(px-L)/(W-L-R)));return new Date(state.s.getTime()+ratio*(state.e-state.s))}
function addZoomControls(id,redraw){const svg=$(id),toolbar=svg.closest(".panel")?.querySelector(".toolbar");if(!toolbar||toolbar.querySelector(`[data-controls="${id}"]`))return;const box=document.createElement("div");box.className="zoom-controls";box.dataset.controls=id;box.innerHTML='<button class="btn" data-zoom="out" title="缩小">−</button><button class="btn" data-zoom="in" title="放大">+</button><button class="btn" data-pan="left" title="向左平移">◀</button><button class="btn" data-pan="right" title="向右平移">▶</button>';toolbar.appendChild(box);box.querySelector('[data-zoom="out"]').onclick=()=>zoomView(id,redraw,.7);box.querySelector('[data-zoom="in"]').onclick=()=>zoomView(id,redraw,1.4);box.querySelector('[data-pan="left"]').onclick=()=>panView(id,redraw,-1);box.querySelector('[data-pan="right"]').onclick=()=>panView(id,redraw,1)}
function zoomView(id,redraw,factor){const state=D[id];if(!state)return;const full={s:new Date(state.fullS),e:new Date(state.fullE)},center=(state.s.getTime()+state.e.getTime())/2;let span=(state.e-state.s)*factor;span=Math.max(7*864e5,(full.e-full.s)/1000,span);state.s=new Date(center-span/2);state.e=new Date(center+span/2);clampZoom(state);redraw()}
function panView(id,redraw,direction){const state=D[id];if(!state)return;const shift=(state.e-state.s)*.2*direction;state.s=new Date(+state.s+shift);state.e=new Date(+state.e+shift);clampZoom(state);redraw()}
function clampZoom(state){const full={s:new Date(state.fullS),e:new Date(state.fullE)};if(state.s<full.s){const span=state.e-state.s;state.s=full.s;state.e=new Date(full.s.getTime()+span)}if(state.e>full.e){const span=state.e-state.s;state.e=full.e;state.s=new Date(full.e.getTime()-span)}}
function bindZoom(id,state,redraw,geometry={W:1520,L:56,R:16}){const svg=$(id);svg._zoomGeom=geometry;svg._zoomState=state;svg._zoomRedraw=redraw;if(svg._zoomListeners)return;svg._zoomListeners=true;let drag=null;
svg.addEventListener("pointerdown",e=>{if(e.button!==0)return;const state=svg._zoomState,geom=svg._zoomGeom,rect=svg.getBoundingClientRect(),ratio=Math.max(0,Math.min(1,((e.clientX-rect.left)*geom.W/rect.width-geom.L)/(geom.W-geom.L-geom.R)));drag={ratio,s:new Date(state.s),e:new Date(state.e),moved:false};svg.setPointerCapture(e.pointerId);svg.classList.add("dragging")});svg.addEventListener("pointermove",e=>{if(!drag)return;const state=svg._zoomState,geom=svg._zoomGeom,rect=svg.getBoundingClientRect(),ratio=Math.max(0,Math.min(1,((e.clientX-rect.left)*geom.W/rect.width-geom.L)/(geom.W-geom.L-geom.R)));if(Math.abs(e.clientX-(drag._lastX??e.clientX))>2)drag.moved=true;drag._lastX=e.clientX;const shift=(ratio-drag.ratio)*(drag.e-drag.s);state.s=new Date(drag.s.getTime()-shift);state.e=new Date(drag.e.getTime()-shift);clampZoom(state);if(drag.moved)redraw()});svg.addEventListener("pointerup",e=>{if(!drag)return;svg._suppressClick=drag.moved;drag=null;svg.classList.remove("dragging")});svg.addEventListener("dblclick",()=>{const state=svg._zoomState,redraw=svg._zoomRedraw;state.s=new Date(state.fullS);state.e=new Date(state.fullE);redraw()});svg.addEventListener("pointercancel",()=>{drag=null;svg.classList.remove("dragging")});addZoomControls(id,()=>svg._zoomRedraw())}
 function ensureOverview(id){const svg=$(id),container=svg.parentElement;let wrap=document.querySelector(`[data-overview="${id}"]`);if(!wrap){wrap=document.createElement("div");wrap.className="overview";wrap.dataset.overview=id;wrap.innerHTML='<div class="range-buttons"></div><svg></svg>';container.after(wrap)}return wrap.querySelector("svg")}
 function overviewFloor(state){const full=new Date(state.fullE)-new Date(state.fullS);return Math.max(7*864e5,full/1000)}
 function rangeLabel(t){const d=new Date(t);return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`}
 function drawOverview(osvg){const ov=osvg._overview;if(!ov)return;const {points,getter,state,options}=ov,W=1520,H=88,L=64,R=64,T=12,B=26;osvg.setAttribute("viewBox",`0 0 ${W} ${H}`);const full={s:new Date(state.fullS),e:new Date(state.fullE)},x=t=>L+Math.max(0,Math.min(1,(new Date(t)-full.s)/(full.e-full.s)))*(W-L-R);let mini=`<rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="#eef2f7" rx="4"/>`;const values=points.map(getter).filter(Number.isFinite);if(values.length){let min=Math.min(...values),max=Math.max(...values);if(min===max){min-=1;max+=1}const pad=(max-min)*.08,lo=min-pad,hi=max+pad,y=v=>T+(1-(v-lo)/(hi-lo))*(H-T-B),line=points.map((p,i)=>`${i?"L":"M"}${x(p.date)},${y(getter(p))}`).join("");if(options.area!==false&&points.length>1)mini+=`<path d="${line}L${x(points.at(-1).date)},${y(lo)}L${x(points[0].date)},${y(lo)}Z" fill="${options.color||"#2563eb"}22" stroke="none"/>`;mini+=`<path d="${line}" fill="none" stroke="${options.color||"#2563eb"}" stroke-width="1.7" vector-effect="non-scaling-stroke"/>`}const sx=x(state.s),ex=x(state.e);if(sx>L)mini+=`<rect x="${L}" y="${T}" width="${sx-L}" height="${H-T-B}" fill="#f8fafce6"/>`;if(ex<W-R)mini+=`<rect x="${ex}" y="${T}" width="${W-R-ex}" height="${H-T-B}" fill="#f8fafce6"/>`;for(let i=0;i<=4;i++){const t=full.s.getTime()+(full.e-full.s)*i/5,xx=x(new Date(t));mini+=`<line x1="${xx}" y1="${T}" x2="${xx}" y2="${H-B}" stroke="#dbe2ea"/><text class="tick" x="${xx}" y="${H-7}" text-anchor="middle">${rangeLabel(t)}</text>`}mini+=`<rect class="range-window" x="${sx}" y="${T}" width="${Math.max(3,ex-sx)}" height="${H-T-B}" rx="4" fill="#2563eb18" stroke="#2563eb" stroke-width="1.3"/><rect class="range-handle" data-side="left" x="${sx-3}" y="${T-2}" width="7" height="${H-T-B+4}" rx="3" fill="#2563eb"/><rect class="range-handle" data-side="right" x="${ex-4}" y="${T-2}" width="7" height="${H-T-B+4}" rx="3" fill="#2563eb"/>`;osvg.innerHTML=mini;const buttons=osvg.closest(".overview")?.querySelectorAll(".range-btn"),fullSpan=full.e-full.s,span=state.e-state.s;if(buttons)buttons.forEach(btn=>{const days=btn.dataset.days;btn.classList.toggle("active",days===""?span>=fullSpan-2*864e5:Math.abs(span-(+days)*864e5)<2*864e5)})}
 function renderOverview(id,points,getter,state,redraw,options={}){const osvg=ensureOverview(id);osvg._overview={points,getter,state,redraw,options};if(!osvg._rangeInit){osvg._rangeInit=true;const wrap=osvg.closest(".overview"),box=wrap?.querySelector(".range-buttons");if(box){[["3M","91"],["6M","182"],["1Y","365"],["3Y","1095"],["5Y","1825"],["ALL",""]].forEach(([label,days])=>{const btn=document.createElement("button");btn.className="range-btn";btn.type="button";btn.textContent=label;btn.dataset.days=days;btn.onclick=()=>{const ov=osvg._overview,full={s:new Date(ov.state.fullS),e:new Date(ov.state.fullE)},floor=overviewFloor(ov.state);if(days){ov.state.e=new Date(full.e);ov.state.s=new Date(Math.max(full.s.getTime(),full.e.getTime()-(+days)*864e5))}else{ov.state.s=new Date(full.s);ov.state.e=new Date(full.e)}const span=ov.state.e-ov.state.s;if(span<floor)ov.state.s=new Date(ov.state.e.getTime()-floor);clampZoom(ov.state);drawOverview(osvg);ov.redraw()};box.appendChild(btn)})}const local=e=>{const rect=osvg.getBoundingClientRect(),geometry={W:1520,L:64,R:64};return Math.max(geometry.L,Math.min(1520-geometry.R,(e.clientX-rect.left)*1520/rect.width))},dateAt=e=>{const ov=osvg._overview,full={s:new Date(ov.state.fullS),e:new Date(ov.state.fullE)},ratio=(local(e)-64)/(1520-64-64);return new Date(full.s.getTime()+Math.max(0,Math.min(1,ratio))*(full.e-full.s))};osvg.addEventListener("pointerdown",e=>{if(e.button!==0)return;const ov=osvg._overview,state=ov.state,sx=64+(new Date(state.s)-new Date(state.fullS))/(new Date(state.fullE)-new Date(state.fullS))*(1520-64-64),ex=64+(new Date(state.e)-new Date(state.fullS))/(new Date(state.fullE)-new Date(state.fullS))*(1520-64-64),px=local(e);let mode="new";if(Math.abs(px-sx)<=12)mode="left";else if(Math.abs(px-ex)<=12)mode="right";else if(px>=sx&&px<=ex)mode="move";osvg._drag={mode,t:dateAt(e),s:new Date(state.s),e:new Date(state.e),moved:false};osvg.classList.add("range-dragging");osvg.setPointerCapture(e.pointerId);e.preventDefault()});osvg.addEventListener("pointermove",e=>{const drag=osvg._drag;if(!drag)return;const ov=osvg._overview,state=ov.state,t=dateAt(e),floor=overviewFloor(state);if(drag.mode==="left")state.s=new Date(Math.min(+t,+state.e-floor));else if(drag.mode==="right")state.e=new Date(Math.max(+t,+state.s+floor));else if(drag.mode==="move"){const shift=+t-+drag.t;state.s=new Date(+drag.s+shift);state.e=new Date(+drag.e+shift)}else{state.s=new Date(Math.min(+drag.t,+t));state.e=new Date(Math.max(+drag.t,+t,floor))}clampZoom(state);drag.moved=true;drawOverview(osvg);ov.redraw()});const stop=()=>{if(!osvg._drag)return;osvg._drag=null;osvg.classList.remove("range-dragging");drawOverview(osvg)};osvg.addEventListener("pointerup",stop);osvg.addEventListener("pointercancel",stop);osvg.addEventListener("lostpointercapture",stop);osvg.addEventListener("dblclick",()=>{const ov=osvg._overview;ov.state.s=new Date(ov.state.fullS);ov.state.e=new Date(ov.state.fullE);drawOverview(osvg);ov.redraw()})}drawOverview(osvg)}
function timeFrame(state,points){return points.filter(p=>{const t=dt(p.date);return t>=state.s&&t<=state.e})}
function yScale(values,T,H,B,includeZero=true){let min=Math.min(...values),max=Math.max(...values);if(includeZero){min=Math.min(0,min);max=Math.max(0,max)}if(min===max){min-=1;max+=1}const pad=(max-min)*.08;min-=pad;max+=pad;return[v=>T+(1-(v-min)/(max-min))*(H-T-B),min,max]}
function drawTimeSeries(id,points,defs,opts={}){const svg=$(id),W=1520,H=+svg.dataset.height||430,L=56,R=16,T=22,B=44;svg.dataset.height=H;svg.setAttribute("viewBox",`0 0 ${W} ${H}`);D[id]=D[id]||{s:new Date(opts.fullS),e:new Date(opts.fullE),fullS:new Date(opts.fullS),fullE:new Date(opts.fullE)};const state=D[id],view=timeFrame(state,points);
if(!view.length){svg.innerHTML=`<text x="180" y="180" class="tick">No data in current range</text>`;return}
const names=defs.filter(d=>d.visible).map(d=>d.name),vals=view.flatMap(p=>names.map(n=>p[n])).filter(Number.isFinite);const[y,min,max]=yScale(vals,T,H,B,opts.includeZero!==false);const x=t=>L+(t-state.s)/(state.e-state.s)*(W-L-R);let axis="";for(let i=0;i<=5;i++){const v=min+(max-min)*i/5,yy=y(v);axis+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e2e8f0"/><text class="tick" x="${L-8}" y="${yy+4}" text-anchor="end">${opts.yFormat?opts.yFormat(v):v.toFixed(2)}</text>`}
const step=Math.max(1,Math.floor(view.length/7));view.forEach((p,i)=>{if(i%step===0)axis+=`<text class="tick" x="${x(dt(p.date))}" y="${H-14}" text-anchor="middle">${p.date}</text>`});
let marks="";if(opts.zero&&min<0&&max>0)marks+=`<line class="zero" x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}"/>`;const paths=defs.filter(d=>d.visible).map(def=>{const pts=view.filter(p=>Number.isFinite(p[def.name]));if(!pts.length)return"";if(opts.area&&def.name===opts.area){const line=pts.map((p,i)=>`${i?"L":"M"}${x(dt(p.date))},${y(p[def.name])}`).join("");return`<path d="${line}L${x(dt(pts.at(-1).date))},${y(0)}L${x(dt(pts[0].date))},${y(0)}Z" fill="${def.color}22" stroke="none"/><path d="${line}" fill="none" stroke="${def.color}" stroke-width="2.4" vector-effect="non-scaling-stroke"/>`}return`<path d="${pts.map((p,i)=>`${i?"L":"M"}${x(dt(p.date))},${y(p[def.name])}`).join("")}" fill="none" stroke="${def.color}" stroke-width="2.4" vector-effect="non-scaling-stroke"/>`}).join("");
svg.innerHTML=axis+marks+paths+`<line class="cursor" y1="${T}" y2="${H-B}" stroke="#64748b" stroke-dasharray="3 3" opacity="0"/>`;const redraw=()=>drawTimeSeries(id,points,defs,opts);bindZoom(id,state,redraw,{W,L,R});svg._yScale={min,max,T,B,H};svg.onmousemove=e=>{const rect=svg.getBoundingClientRect(),px=(e.clientX-rect.left)*W/rect.width,t=xToTime(px,state,L,W,R),i=Math.max(0,Math.min(view.length-1,view.findIndex(p=>dt(p.date)>=t))),p=view[i],rawY=(e.clientY-rect.top)*H/rect.height,clampedY=Math.max(T,Math.min(H-B,rawY)),yValue=(1-(clampedY-T)/(H-T-B))*(max-min)+min;setCross(svg,e,{W,H,L,R,T,B},yValue,opts.yFormat||(v=>v.toFixed(2)));show(e,`<b>${p.date}</b><br>`+defs.filter(d=>d.visible).map(d=>`${d.label}: ${fmt(p[d.name])}`).join("<br>"))};svg.onmouseleave=()=>hideCross(svg);renderOverview(id,points,p=>{const def=defs.find(d=>d.visible);return def?p[def.name]:null},state,()=>drawTimeSeries(id,points,defs,opts),{color:"#2563eb"})}
function drawPnl(){const p=data.daily_pnl;if(!p.available){$("pnl-summary").innerHTML=`<div class="empty">${p.reason}</div>`;return}const s1=$("show-1sigma").checked,s2=$("show-2sigma").checked,s3=$("show-3sigma").checked,refs=[{value:p.mean,color:"#7c3aed",label:"Mean"}];if(s1)refs.push({value:p.mean+p.std,color:"#f59e0b",label:"+1σ"},{value:p.mean-p.std,color:"#f59e0b",label:"-1σ"});if(s2)refs.push({value:p.mean+2*p.std,color:"#ea580c",label:"+2σ"},{value:p.mean-2*p.std,color:"#ea580c",label:"-2σ"});if(s3)refs.push({value:p.mean+3*p.std,color:"#dc2626",label:"+3σ"},{value:p.mean-3*p.std,color:"#dc2626",label:"-3σ"});if($("show-4sigma").checked)refs.push({value:p.mean+4*p.std,color:"#be123c",label:"+4σ"},{value:p.mean-4*p.std,color:"#be123c",label:"-4σ"});if($("show-5sigma").checked)refs.push({value:p.mean+5*p.std,color:"#7f1d1d",label:"+5σ"},{value:p.mean-5*p.std,color:"#7f1d1d",label:"-5σ"});
const svg=$("pnl-chart"),W=1520,H=430,L=56,R=16,T=22,B=44;svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.dataset.height=H;D["pnl-chart"]=D["pnl-chart"]||{s:dt(p.points[0].date),e:dt(p.points.at(-1).date),fullS:dt(p.points[0].date),fullE:dt(p.points.at(-1).date)};const state=D["pnl-chart"],view=timeFrame(state,p.points);if(!view.length){svg.innerHTML='<text x="180" y="180" class="tick">No data</text>';return}
const absMax=Math.max(...view.map(v=>Math.abs(v.value)),p.std*5)*1.08||1,y=v=>T+(1-(v+absMax)/(2*absMax))*(H-T-B),x=t=>L+(t-state.s)/(state.e-state.s)*(W-L-R),bw=Math.max(1,(W-L-R)/Math.max(1,view.length)*.72);let bars="",axis="";for(let i=0;i<=6;i++){const v=-absMax+2*absMax*i/6,yy=y(v);axis+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e2e8f0"/><text class="tick" x="${L-8}" y="${yy+4}" text-anchor="end">${money(v)}</text>`}const step=Math.max(1,Math.floor(view.length/7));view.forEach((r,i)=>{if(i%step===0)axis+=`<text class="tick" x="${x(dt(r.date))}" y="${H-14}" text-anchor="middle">${r.date}</text>`;const yy=y(r.value),h=Math.abs(yy-y(0));bars+=`<rect x="${x(dt(r.date))-bw/2}" y="${Math.min(yy,y(0))}" width="${bw}" height="${h}" fill="${r.value>=0?"#16a34a":"#dc2626"}" opacity=".82"/>`});
let lines=`<line class="zero" x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}"/>`;refs.forEach(r=>{const yy=y(r.value);lines+=`<line class="ref" x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="${r.color}"/><text class="tick" x="${W-R-3}" y="${yy-4}" text-anchor="end" fill="${r.color}">${r.label}</text>`});svg.innerHTML=axis+lines+bars;bindZoom("pnl-chart",state,drawPnl,{W,L,R});svg.onmousemove=e=>{const rect=svg.getBoundingClientRect(),px=(e.clientX-rect.left)*W/rect.width,t=xToTime(px,state,L,W,R),i=Math.max(0,Math.min(view.length-1,view.findIndex(v=>dt(v.date)>=t))),r=view[i],rawY=(e.clientY-rect.top)*H/rect.height,clampedY=Math.max(T,Math.min(H-B,rawY)),yValue=-absMax+2*absMax*(1-(clampedY-T)/(H-T-B));setCross(svg,e,{W,H,L,R,T,B},yValue,money);show(e,`<b>${r.date}</b><br>Daily PnL: ${money(r.value)}`)};svg.onmouseleave=()=>hideCross(svg);renderOverview("pnl-chart",p.points,r=>r.value,state,drawPnl,{color:"#16a34a"})}
function drawStacked(){const p=data.position_analysis.weights;if(!p.available)return;const mode=$("position-mode").value,selected=p.symbols.slice(0,18),svg=$("position-chart"),W=1520,H=480,L=56,R=16,T=20,B=44;svg.setAttribute("viewBox",`0 0 ${W} ${H}`);D["position-chart"]=D["position-chart"]||{s:dt(p.rows[0].date),e:dt(p.rows.at(-1).date),fullS:dt(p.rows[0].date),fullE:dt(p.rows.at(-1).date)};const state=D["position-chart"],view=timeFrame(state,p.rows);if(!view.length){svg.innerHTML='<text x="180" y="180" class="tick">No data</text>';return}
const stack=selected.map((s,i)=>({...s,color:PALETTE[i%PALETTE.length]}));stack.push({symbol:"__others__",color:"#94a3b8"});const hMax=Math.max(1,...view.map(r=>r.holdings||0)),hy=v=>T+(1-v/Math.ceil(hMax*1.15))*(H-T-B);const weightFor=(r,s)=>s==="__others__"?Math.max(0,r.total-selected.reduce((sum,x)=>sum+(r.weights[x]||0),0)):(r.weights[s]||0);const totals=view.map(r=>mode==="total"?r.total:stack.reduce((sum,s)=>sum+weightFor(r,s.symbol),0)),max=1,y=v=>T+(1-v/max)*(H-T-B),x=t=>L+(t-state.s)/(state.e-state.s)*(W-L-R);let axis="";for(let i=0;i<=5;i++){const v=max*i/5,yy=y(v);axis+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e2e8f0"/><text class="tick" x="${L-8}" y="${yy+4}" text-anchor="end">${(v*100).toFixed(1)}%</text>`}for(let i=0;i<=4;i++){const v=Math.ceil(hMax*1.15)*i/4,yy=hy(v);axis+=`<text class="tick" x="${W-R+4}" y="${yy+4}" fill="#7c3aed">${Math.round(v)}</text>`}const step=Math.max(1,Math.floor(view.length/7));view.forEach((r,i)=>{if(i%step===0)axis+=`<text class="tick" x="${x(dt(r.date))}" y="${H-14}" text-anchor="middle">${r.date}</text>`});
let areas="";if(mode==="stacked"){const bases=new Map(view.map(r=>[r.date,0]));stack.forEach(s=>{const top=view.map(r=>{const base=bases.get(r.date),val=weightFor(r,s.symbol);bases.set(r.date,base+val);return`${x(dt(r.date))},${y(base+val)}`}),bottom=view.slice().reverse().map(r=>`${x(dt(r.date))},${y(bases.get(r.date)-weightFor(r,s.symbol))}`).join("L");areas+=`<path d="M${top.join("L")}L${bottom}Z" fill="${s.color}88" stroke="${s.color}" stroke-width=".7" opacity=".85"/>`});areas+=`<path d="M${view.map((r,i)=>`${i?"L":"M"}${x(dt(r.date))},${y(r.total)}`).join("")}" fill="none" stroke="#0f172a" stroke-width="1.2" stroke-dasharray="4 3"/>`}else areas+=`<path d="M${view.map((r,i)=>`${i?"L":"M"}${x(dt(r.date))},${y(r.total)}`).join("")}" fill="none" stroke="#2563eb" stroke-width="2.4"/>`;
const holdingsPath=`<path d="M${view.map((r,i)=>`${i?"L":"M"}${x(dt(r.date))},${hy(r.holdings||0)}`).join("")}" fill="none" stroke="#7c3aed" stroke-width="1.7" stroke-dasharray="5 4"/>`;svg.innerHTML=axis+areas+holdingsPath;bindZoom("position-chart",state,drawStacked,{W,L,R});svg.onmousemove=e=>{const rect=svg.getBoundingClientRect(),px=(e.clientX-rect.left)*W/rect.width,t=xToTime(px,state,L,W,R),i=Math.max(0,Math.min(view.length-1,view.findIndex(r=>dt(r.date)>=t))),r=view[i],rawY=(e.clientY-rect.top)*H/rect.height,clampedY=Math.max(T,Math.min(H-B,rawY)),yValue=1-(clampedY-T)/(H-T-B),other=Math.max(0,r.total-selected.reduce((sum,s)=>sum+(r.weights[s]||0),0)),entries=Object.entries(r.weights).sort((a,b)=>b[1]-a[1]).slice(0,10);if(other>1e-8)entries.push(["Other holdings",other]);const items=entries.map(([s,v])=>`${s}: ${(v*100).toFixed(2)}%`).join("<br>");setCross(svg,e,{W,H,L,R,T,B},yValue,v=>pct(v));show(e,`<b>${r.date}</b><br>Total: ${(r.total*100).toFixed(2)}% · Holdings: ${fmt(r.holdings)}<br>${items}`)};svg.onmouseleave=()=>hideCross(svg);renderOverview("position-chart",p.rows,r=>r.total,state,drawStacked,{color:"#2563eb"})};function drawHoldings(){const p=data.position_analysis.weights;if(!p.available)return;const svg=$("holdings-chart"),W=1520,H=260,L=56,R=16,T=20,B=44;svg.setAttribute("viewBox",`0 0 ${W} ${H}`);D["holdings-chart"]=D["holdings-chart"]||{s:dt(p.rows[0].date),e:dt(p.rows.at(-1).date),fullS:dt(p.rows[0].date),fullE:dt(p.rows.at(-1).date)};const state=D["holdings-chart"],view=timeFrame(state,p.rows);if(!view.length){svg.innerHTML='<text x="180" y="120" class="tick">No data</text>';return}const maxCount=Math.max(1,...view.map(r=>r.holdings||0)),top=Math.ceil(maxCount*1.08),y=v=>T+(1-v/top)*(H-T-B),x=t=>L+(t-state.s)/(state.e-state.s)*(W-L-R),bw=Math.max(1,(W-L-R)/Math.max(1,view.length)*.78);let axis="",bars="";for(let i=0;i<=4;i++){const v=top*i/4,yy=y(v);axis+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e2e8f0"/><text class="tick" x="${L-8}" y="${yy+4}" text-anchor="end">${Math.round(v)}</text>`}const step=Math.max(1,Math.floor(view.length/7));view.forEach((r,i)=>{if(i%step===0)axis+=`<text class="tick" x="${x(dt(r.date))}" y="${H-14}" text-anchor="middle">${r.date}</text>`;const yy=y(r.holdings||0);bars+=`<rect x="${x(dt(r.date))-bw/2}" y="${yy}" width="${bw}" height="${y(0)-yy}" fill="#7c3aed" opacity=".78"/>`});svg.innerHTML=axis+bars;bindZoom("holdings-chart",state,drawHoldings,{W,L,R});svg.onmousemove=e=>{const rect=svg.getBoundingClientRect(),px=(e.clientX-rect.left)*W/rect.width,t=xToTime(px,state,L,W,R),i=Math.max(0,Math.min(view.length-1,view.findIndex(r=>dt(r.date)>=t))),r=view[i],rawY=(e.clientY-rect.top)*H/rect.height,clampedY=Math.max(T,Math.min(H-B,rawY)),yValue=top*(1-(clampedY-T)/(H-T-B));setCross(svg,e,{W,H,L,R,T,B},yValue,v=>fmt(v));show(e,`<b>${r.date}</b><br>Holdings: ${fmt(r.holdings)}`)};svg.onmouseleave=()=>hideCross(svg);renderOverview("holdings-chart",p.rows,r=>r.holdings,state,drawHoldings,{color:"#7c3aed",area:false})}
function drawConcentration(){const c=data.concentration,kind=$("conc-kind").value,k=+$("conc-k").value;if(!c.available)return;$("conc-k-label").textContent=k;const items=c.items[kind].slice(0,k),W=1520,rowH=30,H=Math.max(260,items.length*rowH+80),L=190,R=90,T=22,B=42,maxAbs=Math.max(...items.map(i=>Math.abs(i.pnl||0)),1),x=v=>L+Math.max(0,v)/(maxAbs*1.05)*(W-L-R);let axis="";for(let i=0;i<=5;i++){const v=maxAbs*1.05*i/5,xx=x(v);axis+=`<line x1="${xx}" y1="${T}" x2="${xx}" y2="${H-B}" stroke="#e2e8f0"/><text class="tick" x="${xx}" y="${H-15}" text-anchor="middle">${money(v)}</text>`}
const bars=items.map((item,i)=>{const y=T+i*rowH+8,pnl=item.pnl||0,xx=pnl>=0?x(0):x(Math.abs(pnl)),w=Math.max(2,Math.abs(pnl)/(maxAbs*1.05)*(W-L-R)),cx=x(maxAbs*.3);return`<text x="8" y="${y+11}" font-size="12">${item.label}</text><rect x="${xx}" y="${y}" width="${w}" height="18" rx="4" fill="${pnl>=0?"#16a34a":"#dc2626"}" opacity=".82" data-i="${i}"/><circle cx="${cx}" cy="${y+9}" r="3" fill="#facc15" stroke="#a16207"/><text x="${cx+7}" y="${y+13}" font-size="10" fill="#a16207">${(item.cumulative_share*100).toFixed(1)}%</text>`}).join("");
$("concentration-chart").setAttribute("viewBox",`0 0 ${W} ${H}`);$("concentration-chart").innerHTML=axis+bars;$("concentration-chart").querySelectorAll("rect[data-i]").forEach(n=>{n.onmousemove=e=>{const item=items[+n.dataset.i];show(e,`<b>${item.label}</b><br>PnL: ${money(item.pnl)}<br>Abs share: ${pct(item.share)}<br>Cumulative: ${pct(item.cumulative_share)}`)};n.onmouseleave=hide})}
function outcomeColor(row){return row.fully_closed?(+(row.realized_pnl||0)>=0?"#16a34a":"#dc2626"):"#2563eb"}
function intervalColor(row,interval){return interval.closed_pnl===undefined||interval.closed_pnl===null?"#2563eb":+(interval.closed_pnl)>=0?"#16a34a":"#dc2626"}
function drawTimeline(){const tl=data.timeline;if(!tl.available)return;const search=$("timeline-search").value.trim().toLowerCase(),buy=$("filter-buy").checked,sell=$("filter-sell").checked,rows=tl.symbols.filter(r=>r.symbol.toLowerCase().includes(search));const W=1520,L=140,R=16,T=42,rowH=30,H=Math.max(260,rows.length*rowH+82);D["gantt-chart"]=D["gantt-chart"]||{s:dt(tl.start_date),e:new Date(dt(tl.end_date).getTime()+864e5),fullS:dt(tl.start_date),fullE:new Date(dt(tl.end_date).getTime()+864e5)};const state=D["gantt-chart"],x=d=>L+(Math.max(state.s,Math.min(state.e,dt(d)))-state.s)/(state.e-state.s)*(W-L-R);let axis="",tickDates=tl.events.map(e=>e.date).filter((v,i,a)=>a.indexOf(v)===i).sort(),step=Math.max(1,Math.floor(tickDates.length/9));tickDates.forEach((d,i)=>{if(i%step===0)axis+=`<text class="tick" x="${x(d)}" y="22" text-anchor="middle">${d}</text><line x1="${x(d)}" y1="30" x2="${x(d)}" y2="${H-28}" stroke="#e2e8f0"/>`});
const selectedSymbol=$("timeline-symbol")?.value,body=rows.map((row,i)=>{const y=T+i*rowH,color=outcomeColor(row),intervalColor=q=>q.closed_pnl===undefined||q.closed_pnl===null?'#2563eb':+(q.closed_pnl)>=0?'#16a34a':'#dc2626',intervals=tl.intervals.filter(q=>q.symbol===row.symbol),events=tl.events.filter(e=>e.symbol===row.symbol&&(e.side==="Buy"?buy:sell)),intSvg=intervals.map(q=>`<rect data-symbol="${row.symbol}" data-start="${q.start}" data-end="${q.end}" data-qty="${q.position_after}" data-pnl="${q.closed_pnl??""}" x="${x(q.start)}" y="${y+6}" width="${Math.max(2,x(q.end)-x(q.start))}" height="17" rx="5" fill="${intervalColor(q)}22" stroke="${intervalColor(q)}" stroke-width="1"/>`).join(""),evSvg=events.map(e=>`<line class="ev" data-symbol="${row.symbol}" data-side="${e.side}" data-date="${e.date}" data-qty="${e.quantity}" data-price="${e.price}" data-notional="${e.notional}" data-fees="${e.fees}" x1="${x(e.date)}" y1="${y+1}" x2="${x(e.date)}" y2="${y+28}" stroke="${e.side==="Buy"?"#3b82f6":"#f97316"}" stroke-width="1.8"/>`).join("");return`<text x="10" y="${y+19}" font-size="12">${row.symbol}</text><rect class="row ${row.symbol===selectedSymbol?"row-selected":""}" data-symbol="${row.symbol}" x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`+intSvg+evSvg}).join("");
$("gantt-chart").setAttribute("viewBox",`0 0 ${W} ${H}`);$("gantt-chart").dataset.height=H;$("gantt-chart").innerHTML=axis+body;bindZoom("gantt-chart",state,drawTimeline,{W,L,R});$("gantt-chart").onmousemove=e=>{const rect=$("gantt-chart").getBoundingClientRect(),rawY=(e.clientY-rect.top)*H/rect.height,rowIndex=Math.max(0,Math.min(rows.length-1,Math.floor((rawY-T)/rowH))),row=rows[rowIndex];setCross($("gantt-chart"),e,{W,H,L,R,T,B},rowIndex+1,()=>row.symbol)};$("gantt-chart").onmouseleave=()=>hideCross($("gantt-chart"));$("gantt-chart").querySelectorAll("rect[data-qty]").forEach(n=>{n.onmousemove=e=>show(e,`<b>${n.dataset.symbol}</b><br>${n.dataset.start} ~ ${n.dataset.end}${n.dataset.pnl===undefined||n.dataset.pnl===""?`<br>Closed PnL: Open`:`<br>Closed PnL: ${money(n.dataset.pnl)}`}`);n.onmouseleave=hide;n.onclick=()=>{if(!$("gantt-chart")._suppressClick)select(n.dataset.symbol)}});$("gantt-chart").querySelectorAll("line.ev").forEach(n=>{n.onmousemove=e=>show(e,`<b>${n.dataset.symbol} ${n.dataset.side}</b><br>Date: ${n.dataset.date}<br>Qty: ${fmt(n.dataset.qty)}<br>Price: ${fmt(n.dataset.price)}<br>Notional: ${fmt(n.dataset.notional)}<br>Fees: ${fmt(n.dataset.fees)}`);n.onmouseleave=hide;n.onclick=()=>{if(!$("gantt-chart")._suppressClick)select(n.dataset.symbol)}});$("gantt-chart").querySelectorAll(".row").forEach(n=>{n.onclick=()=>{if(!$("gantt-chart")._suppressClick)select(n.dataset.symbol)}});const eventCounts=new Map();tl.events.forEach(e=>eventCounts.set(e.date,(eventCounts.get(e.date)||0)+1));const overviewPoints=[...eventCounts.entries()].map(([date,value])=>({date,value})).sort((a,b)=>a.date.localeCompare(b.date));renderOverview("gantt-chart",overviewPoints,p=>p.value,state,drawTimeline,{color:"#7c3aed"})}
function drawSymbolPrice(){
  const symbol=$("timeline-symbol").value,prices=data.symbol_prices,chart=prices?.symbols?.[symbol],events=data.timeline.events.filter(e=>e.symbol===symbol),svg=$("symbol-price-chart");
  if(!svg)return;
  if(!prices?.available||!chart?.points?.length){svg.innerHTML=`<text x="180" y="120" class="tick">${prices?.reason||"No price data"}</text>`;return}
  const all=chart.points,W=1520,H=360,L=64,R=18,T=22,B=44;svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  D["symbol-price-chart"]={s:dt(all[0].date),e:dt(all.at(-1).date),fullS:dt(all[0].date),fullE:dt(all.at(-1).date)};
  const state=D["symbol-price-chart"],view=timeFrame(state,all);
  if(!view.length){svg.innerHTML='<text x="180" y="120" class="tick">No data in current range</text>';return}
  const visible=events.filter(e=>{const t=dt(e.date);return t>=state.s&&t<=state.e}),values=chart.candles?view.flatMap(p=>[p.high,p.low]):view.map(p=>p.price),eventValues=visible.map(e=>e.price).filter(Number.isFinite),[y,min,max]=yScale([...values,...eventValues],T,H,B,true),x=t=>L+(t-state.s)/(state.e-state.s)*(W-L-R),bw=Math.max(2,(W-L-R)/Math.max(1,view.length)*.62);
  let body="",axis="";
  for(let i=0;i<=4;i++){const v=min+(max-min)*i/4,yy=y(v);axis+=`<line x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}" stroke="#e2e8f0"/><text class="tick" x="${L-8}" y="${yy+4}" text-anchor="end">${money(v)}</text>`}
  const step=Math.max(1,Math.floor(view.length/7));
  view.forEach((p,i)=>{if(i%step===0)axis+=`<text class="tick" x="${x(dt(p.date))}" y="${H-14}" text-anchor="middle">${p.date}</text>`;const xx=x(dt(p.date));
    if(chart.candles){const up=p.close>=p.open,yTop=y(Math.max(p.open,p.close)),h=Math.max(1,Math.abs(y(p.open)-y(p.close)));body+=`<line x1="${xx}" y1="${y(p.high)}" x2="${xx}" y2="${y(p.low)}" stroke="${up?"#16a34a":"#dc2626"}"/><rect x="${xx-bw/2}" y="${yTop}" width="${bw}" height="${h}" fill="${up?"#16a34a":"#dc2626"}" opacity=".82"/>`}
  });
  if(!chart.candles)body=`<path d="${view.map((p,i)=>`${i?"L":"M"}${x(dt(p.date))},${y(p.price)}`).join("")}L${x(dt(view.at(-1).date))},${y(0)}L${x(dt(view[0].date))},${y(0)}Z" fill="#2563eb18" stroke="none"/><path d="${view.map((p,i)=>`${i?"L":"M"}${x(dt(p.date))},${y(p.price)}`).join("")}" fill="none" stroke="#2563eb" stroke-width="2.2"/>`;
  const nearest=date=>view.reduce((best,p)=>Math.abs(dt(p.date)-dt(date))<Math.abs(dt(best.date)-dt(date))?p:best,view[0]),markers=visible.map(e=>{const xx=x(dt(e.date)),value=e.price??nearest(e.date).price,yy=y(value),buy=e.side==="Buy",shape=buy?`M${xx},${yy+7}L${xx-6},${yy+19}L${xx+6},${yy+19}Z`:`M${xx},${yy-7}L${xx-6},${yy-19}L${xx+6},${yy-19}Z`;return`<path class="trade-marker" data-symbol="${symbol}" data-side="${e.side}" data-date="${e.date}" data-qty="${e.quantity}" data-price="${value}" data-fees="${e.fees}" d="${shape}" fill="${buy?"#16a34a":"#dc2626"}" stroke="#fff" stroke-width=".8"/>`}).join("");
  svg.innerHTML=axis+body+markers;bindZoom("symbol-price-chart",state,drawSymbolPrice,{W,L,R});
  svg.onmousemove=e=>{const rect=svg.getBoundingClientRect(),px=(e.clientX-rect.left)*W/rect.width,t=xToTime(px,state,L,W,R),i=Math.max(0,Math.min(view.length-1,view.findIndex(p=>dt(p.date)>=t))),p=view[i],clampedY=Math.max(T,Math.min(H-B,(e.clientY-rect.top)*H/rect.height)),value=chart.candles?min+(max-min)*(1-(clampedY-T)/(H-T-B)):p.price;setCross(svg,e,{W,H,L,R,T,B},value,v=>money(v));show(e,`<b>${p.date}</b><br>${chart.candles?`O:${fmt(p.open)} H:${fmt(p.high)}<br>L:${fmt(p.low)} C:${fmt(p.close)}`:`Close: ${fmt(p.price)}`}`)};
  svg.onmouseleave=()=>hideCross(svg);
  svg.querySelectorAll(".trade-marker").forEach(n=>{n.onmousemove=e=>show(e,`<b>${n.dataset.symbol} ${n.dataset.side}</b><br>Date: ${n.dataset.date}<br>Qty: ${fmt(n.dataset.qty)}<br>Price: ${fmt(n.dataset.price)}<br>Fees: ${fmt(n.dataset.fees)}`);n.onmouseleave=hide});
  renderOverview("symbol-price-chart",all,p=>p.price,state,drawSymbolPrice,{color:"#2563eb"})
}
function select(symbol){const selector=$("timeline-symbol");if(selector&&selector.value!==symbol)selector.value=symbol;const tl=data.timeline,row=tl.symbols.find(r=>r.symbol===symbol),trades=tl.trades.filter(t=>t.symbol===symbol),events=tl.events.filter(e=>e.symbol===symbol);if(!row)return;renderSymbolPrice(symbol,events);$("symbol-detail").innerHTML=`<h3>${symbol}</h3><div class="subchart-title">价格K线与买卖点</div><div class="chart"><svg id="symbol-price-chart"></svg></div><div class="hint">蓝/红点为买入/卖出；bundle 含 OHLC 时显示蜡烛，当前仅含 close 时显示收盘价线。</div><div class="cards">${card("First",row.first_date)+card("Last",row.last_date)+card("Realized PnL",money(row.realized_pnl),pc(row.realized_pnl))+(row.unrealized_pnl===null?card("Unrealized","No price"):card("Unrealized",money(row.unrealized_pnl),pc(row.unrealized_pnl)))+card("End Qty",fmt(row.open_quantity))+card("Fees",money(row.fees))}</div><div class="table-scroll"><table><thead><tr><th>Type</th><th>Date</th><th>Qty</th><th>Price</th><th>Notional</th><th>Fees</th></tr></thead><tbody>${events.map(e=>`<tr><th>${e.side}</th><td>${e.date}</td><td>${fmt(e.quantity)}</td><td>${fmt(e.price)}</td><td>${fmt(e.notional)}</td><td>${fmt(e.fees)}</td></tr>`).join("")}</tbody></table></div><div class="table-scroll" style="margin-top:8px"><table><thead><tr><th>Open</th><th>Close</th><th>Qty</th><th>Buy</th><th>Sell</th><th>Fees</th><th>Net PnL</th><th>Return</th></tr></thead><tbody>${trades.map(t=>`<tr><td>${t.open_date}</td><td>${t.close_date}</td><td>${fmt(t.quantity)}</td><td>${fmt(t.buyPrice)}</td><td>${fmt(t.sellPrice)}</td><td>${fmt(t.fees)}</td><td class="${pc(t.net_pnl)}">${money(t.net_pnl)}</td><td class="${pc(t.return_on_cost)}">${pct(t.return_on_cost)}</td></tr>`).join("")||'<tr><td colspan="8" class="muted">No closed trades</td></tr>'}</tbody></table></div>`;drawTimeline()}
function render(){const meta=data.meta||{},bm=data.nav_analysis.series?.benchmark?.metrics,st=data.nav_analysis.series?.strategy?.metrics;$("subtitle").textContent=`${meta.strategyId||""} ${meta.strategyVersion||""} · Benchmark: ${meta.benchmark||"-"}`;$("overview-cards").innerHTML=st?card("Total Return",pct(st.cumulative_return),pc(st.cumulative_return))+card("CAGR",pct(st.cagr),pc(st.cagr))+card("Max Drawdown",pct(data.drawdown_analysis.series.strategy.summary.max_drawdown),pc(data.drawdown_analysis.series.strategy.summary.max_drawdown))+(bm?card("Benchmark Return",pct(bm.cumulative_return),pc(bm.cumulative_return))+card("Active CAGR",pct(st.cagr-bm.cagr),pc(st.cagr-bm.cagr)):card("Benchmark","-"))+card("Sharpe",fmt(st.sharpe_ratio))+card("Sortino",fmt(st.sortino_ratio)):"";$("bundle-info").innerHTML=data.bundle?`Bundle: ${data.bundle.bundleId||"-"} · Datasets: ${data.bundle.datasets.join(", ")}`:"";
if(data.nav_analysis.available){const s=data.nav_analysis.series.strategy.metrics,b=data.nav_analysis.series.benchmark?.metrics;$("nav-summary").innerHTML=card("Total Return",pct(s.cumulative_return),pc(s.cumulative_return))+card("CAGR",pct(s.cagr),pc(s.cagr))+card("Volatility",pct(s.annual_volatility))+card("Sharpe",fmt(s.sharpe_ratio))+(b?card("Benchmark Return",pct(b.cumulative_return),pc(b.cumulative_return))+card("Active CAGR",pct(s.cagr-b.cagr),pc(s.cagr-b.cagr)):card("Benchmark","No series"))+card("Sortino",fmt(s.sortino_ratio));const redraw=()=>{const map=new Map();["strategy","benchmark"].forEach(name=>{if(!$(name==="strategy"?"show-strategy":"show-benchmark").checked)return;data.nav_analysis.nav[name].forEach(p=>{const q=map.get(p.date)||{date:p.date};q[name]=p.value;map.set(p.date,q)})});const points=[...map.values()].sort((a,b)=>a.date.localeCompare(b.date));drawTimeSeries("nav-chart",points,[{name:"strategy",label:"Strategy",color:"#2563eb",visible:$("show-strategy").checked},{name:"benchmark",label:"Benchmark",color:"#94a3b8",visible:$("show-benchmark").checked}],{fullS:dt(points[0].date),fullE:dt(points.at(-1).date)})};$("show-strategy").onchange=redraw;$("show-benchmark").onchange=redraw;redraw()}else $("nav-summary").innerHTML=`<div class="empty">${data.nav_analysis.reason}</div>`;
if(data.drawdown_analysis.available){const ss=data.drawdown_analysis.series.strategy.summary,bb=data.drawdown_analysis.series.benchmark?.summary;$("dd-summary").innerHTML=card("Strategy Max DD",pct(ss.max_drawdown),pc(ss.max_drawdown))+card("Peak",ss.peak_date)+card("Trough",ss.trough_date)+card("Recovery",ss.recovery_date||"Not recovered")+(bb?card("Benchmark Max DD",pct(bb.max_drawdown),pc(bb.max_drawdown)):"");const redraw=()=>{const map=new Map();["strategy","benchmark"].forEach(name=>{if(!$(name==="strategy"?"dd-strategy":"dd-benchmark").checked)return;data.drawdown_analysis.series[name].points.forEach(p=>{const q=map.get(p.date)||{date:p.date};q[name]=p.value;map.set(p.date,q)})});const points=[...map.values()].sort((a,b)=>a.date.localeCompare(b.date));drawTimeSeries("dd-chart",points,[{name:"strategy",label:"Strategy DD",color:"#dc2626",visible:$("dd-strategy").checked},{name:"benchmark",label:"Benchmark DD",color:"#94a3b8",visible:$("dd-benchmark").checked}],{fullS:dt(points[0].date),fullE:dt(points.at(-1).date),area:"strategy",zero:true})};$("dd-strategy").onchange=redraw;$("dd-benchmark").onchange=redraw;redraw()}else $("dd-summary").innerHTML=`<div class="empty">${data.drawdown_analysis.reason}</div>`;
const pw=data.position_analysis.weights,ps=data.position_analysis.summary;$("position-summary").innerHTML=ps?card("Symbols",fmt(ps.symbol_count))+card("Open Symbols",fmt(ps.open_symbol_count))+card("Avg Position",pct(pw.available?pw.summary.average_weight:null))+card("Max Position",pct(pw.available?pw.summary.max_weight:null))+card("Avg Holdings",fmt(pw.available?pw.summary.average_symbol_count:null))+card("Max Holdings",fmt(pw.available?pw.summary.max_symbol_count:null))+(pw.available?card("Max Date",pw.summary.max_weight_date):"")+card("Realized PnL",money(ps.realized_pnl),pc(ps.realized_pnl)):"";if(pw.available){$("position-legend").innerHTML=pw.symbols.slice(0,18).map((s,i)=>`<span><span class="dot" style="background:${PALETTE[i%PALETTE.length]}"></span>${s}</span>`).join("")+'<span><span class="dot" style="background:#94a3b8"></span>Other holdings</span>';$("position-mode").onchange=drawStacked;drawStacked();drawHoldings()}else $("position-legend").innerHTML=`<span>${pw.reason}</span>`;
const pnl=data.daily_pnl;if(pnl.available){$("pnl-summary").innerHTML=card("Mean",money(pnl.mean))+card("Std Dev",money(pnl.std))+card(">1σ Days",fmt(pnl.sigma_counts["1"]))+card(">2σ Days",fmt(pnl.sigma_counts["2"]),pnl.sigma_counts["2"]?"neg":"")+card(">3σ Days",fmt(pnl.sigma_counts["3"]),pnl.sigma_counts["3"]?"neg":"")+card(">4σ Days",fmt(pnl.sigma_counts["4"]),pnl.sigma_counts["4"]?"neg":"")+card(">5σ Days",fmt(pnl.sigma_counts["5"]),pnl.sigma_counts["5"]?"neg":"");["show-1sigma","show-2sigma","show-3sigma","show-4sigma","show-5sigma"].forEach(id=>$(id).onchange=drawPnl);drawPnl()}else $("pnl-summary").innerHTML=`<div class="empty">${pnl.reason}</div>`;
const c=data.concentration;if(c.available){const m=c.metrics;$("concentration-summary").innerHTML=card("Top-1 Trade",pct(m.trade_top_1))+card("Top-5 Trades",pct(m.trade_top_5))+card("Top-10 Trades",pct(m.trade_top_10))+card("Profit Symbols",fmt(m.positive_symbol_count))+card("Loss Symbols",fmt(m.negative_symbol_count));$("conc-kind").onchange=drawConcentration;$("conc-k").oninput=drawConcentration;$("conc-k").max=Math.min(50,c.items.trade.length);drawConcentration()}else $("concentration-summary").innerHTML=`<div class="empty">${c.reason}</div>`;
const tl=data.timeline;if(tl.available){const closed=tl.symbols.filter(r=>r.fully_closed),win=closed.filter(r=>+(r.realized_pnl||0)>=0).length;$("timeline-summary").innerHTML=card("Symbols",fmt(tl.symbols.length))+card("Closed Symbols",fmt(closed.length))+card("Closed Winners",fmt(win),"pos")+card("Closed Losers",fmt(closed.length-win),"neg")+card("Closed Trades",fmt(tl.trades.length));const symbolSelector=$("timeline-symbol");symbolSelector.innerHTML=tl.symbols.map(row=>`<option value="${row.symbol}">${row.symbol}</option>`).join("");symbolSelector.value=tl.symbols[0]?.symbol||"";symbolSelector.onchange=()=>{const value=symbolSelector.value;if(value){select(value);$("symbol-detail").scrollIntoView({behavior:"smooth",block:"nearest"})}};$("timeline-search").oninput=()=>{const exact=tl.symbols.find(row=>row.symbol.toLowerCase()===$("timeline-search").value.trim().toLowerCase());if(exact)select(exact.symbol);else drawTimeline()};["filter-buy","filter-sell"].forEach(id=>$(id).oninput=drawTimeline);drawTimeline();const first=tl.symbols[0];if(first)select(first.symbol)}else $("symbol-detail").innerHTML=`<div class="empty">${tl.reason}</div>`}
document.querySelectorAll("[data-reset]").forEach(btn=>btn.onclick=()=>{const id=btn.dataset.reset;if(D[id]){D[id].s=D[id].fullS;D[id].e=D[id].fullE}if(id==="nav-chart")$("show-strategy").dispatchEvent(new Event("change"));else if(id==="dd-chart")$("dd-strategy").dispatchEvent(new Event("change"));else if(id==="pnl-chart")drawPnl();else if(id==="position-chart")drawStacked();else if(id==="holdings-chart")drawHoldings();else if(id==="gantt-chart")drawTimeline()});
render();
</script>
</body>
</html>
"""


def render_interactive_html(data: dict[str, Any]) -> str:
    """Render the interactive HTML page."""
    title = data.get("meta", {}).get("title") or "Interactive Analysis"
    html = _TEMPLATE.replace("__TITLE__", str(title)).replace("__GENERATED__", data.get("generated_at", "")).replace("__PAYLOAD__", _json(data))
    return html
