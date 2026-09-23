from __future__ import annotations

"""Order loading, normalization, FIFO matching and trading-style analytics.

The implementation intentionally performs a conservative, order-only analysis:
it does not infer corporate actions, missing fills, dividends or position
transfers.  Unmatched sells and open lots are reported for reconciliation
instead of being silently converted into profit and loss.
"""

from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_ORDER_COLUMNS = {"symbol", "timestamp", "size", "price", "side"}
OPTIONAL_ORDER_COLUMNS = {"order_id", "fees"}
COLUMN_ALIASES = {
    "order_id": "order_id",
    "orderid": "order_id",
    "symbol": "symbol",
    "instrument": "symbol",
    "security": "symbol",
    "column": "symbol",
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


def _normalize_side(value: Any) -> str:
    """Convert common buy/sell spellings to canonical Buy/Sell values."""
    text = str(value).strip().lower()
    if text in {"buy", "b", "long", "cover"}:
        return "Buy"
    if text in {"sell", "s", "short"}:
        return "Sell"
    raise ValueError(f"unsupported order side: {value}")


def normalize_orders(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases, coerce types, filter invalid rows and sort fills."""
    renamed: dict[str, str] = {}
    for column in frame.columns:
        normalized = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in COLUMN_ALIASES:
            renamed[column] = COLUMN_ALIASES[normalized]
    frame = frame.rename(columns=renamed)
    missing = REQUIRED_ORDER_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"orders file is missing required columns: {', '.join(sorted(missing))}")

    for column in OPTIONAL_ORDER_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0 if column != "order_id" else None

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["size"] = pd.to_numeric(frame["size"], errors="raise").abs()
    frame["price"] = pd.to_numeric(frame["price"], errors="raise")
    frame["fees"] = pd.to_numeric(frame["fees"], errors="coerce").fillna(0.0)
    frame["side"] = frame["side"].map(_normalize_side)
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame = frame.dropna(subset=["timestamp", "size", "price"])
    frame = frame[(frame["size"] > 0) & (frame["price"] >= 0)].copy()
    if "order_id" in frame.columns:
        frame = frame.sort_values(["timestamp", "order_id"], kind="stable")
    else:
        frame = frame.sort_values(["timestamp"], kind="stable")
    return frame.reset_index(drop=True)


def load_orders(orders_path: str | Path) -> pd.DataFrame:
    """Load an orders table from CSV, Excel or Parquet."""
    path = Path(orders_path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    return normalize_orders(frame)


def _match_fifo_lots(
    orders: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Match sells against oldest open buys and return lot-level results.

    Each sell can consume multiple open lots.  Fees are allocated by matched
    quantity, and residual sell quantity becomes an unmatched-sale record.
    """
    open_lots: dict[str, deque[dict[str, Any]]] = {}
    round_trips: list[dict[str, Any]] = []
    unmatched_sells: list[dict[str, Any]] = []

    for row in orders.itertuples(index=False):
        symbol = row.symbol
        quantity = float(row.size)
        if row.side == "Buy":
            open_lots.setdefault(symbol, deque()).append(
                {
                    "open_date": row.timestamp,
                    "quantity": quantity,
                    "remaining_quantity": quantity,
                    "price": float(row.price),
                    "fee": float(row.fees),
                }
            )
            continue

        remaining_sell_quantity = quantity
        sell_fee = float(row.fees)
        for lot in list(open_lots.get(symbol, deque())):
            if remaining_sell_quantity <= 1e-12:
                break
            if lot["remaining_quantity"] <= 1e-12:
                continue
            matched_quantity = min(lot["remaining_quantity"], remaining_sell_quantity)
            quantity_share = matched_quantity / lot["quantity"]
            sell_quantity_share = matched_quantity / quantity
            buy_notional = matched_quantity * lot["price"]
            sell_notional = matched_quantity * row.price
            buy_fee_allocated = lot["fee"] * quantity_share
            sell_fee_allocated = sell_fee * sell_quantity_share
            cost_basis = buy_notional + buy_fee_allocated
            gross_pnl = sell_notional - buy_notional
            net_pnl = gross_pnl - buy_fee_allocated - sell_fee_allocated
            round_trips.append(
                {
                    "symbol": symbol,
                    "open_date": lot["open_date"].isoformat(),
                    "close_date": row.timestamp.isoformat(),
                    "quantity": matched_quantity,
                    "buy_price": lot["price"],
                    "sell_price": float(row.price),
                    "cost_basis": cost_basis,
                    "sell_notional": sell_notional,
                    "gross_pnl": gross_pnl,
                    "fees": buy_fee_allocated + sell_fee_allocated,
                    "net_pnl": net_pnl,
                    "return_on_cost": net_pnl / cost_basis if cost_basis else 0.0,
                    "holding_days": float((row.timestamp - lot["open_date"]).days),
                }
            )
            lot["remaining_quantity"] -= matched_quantity
            remaining_sell_quantity -= matched_quantity

        if remaining_sell_quantity > 1e-12:
            unmatched_sells.append(
                {
                    "symbol": symbol,
                    "timestamp": row.timestamp.isoformat(),
                    "quantity": remaining_sell_quantity,
                    "price": float(row.price),
                    "notional": remaining_sell_quantity * float(row.price),
                }
            )

    open_positions: list[dict[str, Any]] = []
    for symbol, lots in open_lots.items():
        for lot in lots:
            if lot["remaining_quantity"] <= 1e-12:
                continue
            quantity_share = lot["remaining_quantity"] / lot["quantity"]
            cost = lot["remaining_quantity"] * lot["price"] + lot["fee"] * quantity_share
            open_positions.append(
                {
                    "symbol": symbol,
                    "open_date": lot["open_date"].isoformat(),
                    "quantity": lot["remaining_quantity"],
                    "price": lot["price"],
                    "cost_basis": cost,
                }
            )
    return round_trips, unmatched_sells, open_positions


def _top_records(values: dict[str, float], ascending: bool, count: int = 5) -> list[dict[str, Any]]:
    """Return the highest or lowest symbols as compact JSON-friendly records."""
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=not ascending)
    return [{"symbol": symbol, "value": float(value)} for symbol, value in ordered[:count]]


def analyze_orders(orders: pd.DataFrame) -> dict[str, Any]:
    """Build activity, fee, concentration, FIFO PnL and reconciliation views."""
    if orders.empty:
        raise ValueError("orders data is empty")

    orders = orders.copy()
    orders["notional"] = orders["size"] * orders["price"]
    buy_orders = orders[orders["side"] == "Buy"]
    sell_orders = orders[orders["side"] == "Sell"]
    round_trips, unmatched_sells, open_positions = _match_fifo_lots(orders)

    if round_trips:
        trades = pd.DataFrame(round_trips)
        trades["close_date"] = pd.to_datetime(trades["close_date"])
        total_net_pnl = float(trades["net_pnl"].sum())
        total_trade_fees = float(trades["fees"].sum())
        winning_trades = int((trades["net_pnl"] > 0).sum())
        losing_trades = int((trades["net_pnl"] < 0).sum())
        win_rate = winning_trades / len(trades)
        average_holding_days = float(trades["holding_days"].mean())
        median_holding_days = float(trades["holding_days"].median())
        max_holding_days = float(trades["holding_days"].max())
        pnl_by_symbol = trades.groupby("symbol")["net_pnl"].sum().to_dict()
        monthly_realized_pnl = trades.set_index("close_date").resample("ME")["net_pnl"].sum().dropna().to_dict()
    else:
        total_net_pnl = 0.0
        total_trade_fees = 0.0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        average_holding_days = 0.0
        median_holding_days = 0.0
        max_holding_days = 0.0
        pnl_by_symbol = {}
        monthly_realized_pnl = {}

    total_notional = float(orders["notional"].sum())
    buy_notional = float(buy_orders["notional"].sum())
    sell_notional = float(sell_orders["notional"].sum())
    total_fees = float(orders["fees"].sum())
    symbol_notional = orders.groupby("symbol")["notional"].sum().to_dict()
    symbol_share = pd.Series(symbol_notional) / total_notional
    years = max(1e-12, (orders["timestamp"].max() - orders["timestamp"].min()).days / 365.25)
    monthly_activity = {
        str(index.strftime("%Y-%m")): {
            "order_count": int(group.shape[0]),
            "buy_notional": float(group.loc[group["side"] == "Buy", "notional"].sum()),
            "sell_notional": float(group.loc[group["side"] == "Sell", "notional"].sum()),
            "gross_notional": float(group["notional"].sum()),
            "fees": float(group["fees"].sum()),
        }
        for index, group in orders.set_index("timestamp").resample("MS")
    }

    return {
        "available": True,
        "matching_method": "FIFO",
        "period_start": orders["timestamp"].min().isoformat(),
        "period_end": orders["timestamp"].max().isoformat(),
        "order_summary": {
            "order_count": int(len(orders)),
            "buy_count": int(len(buy_orders)),
            "sell_count": int(len(sell_orders)),
            "symbol_count": int(orders["symbol"].nunique()),
            "total_notional": total_notional,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "annualized_gross_notional": total_notional / years,
            "buy_sell_notional_ratio": buy_notional / sell_notional if sell_notional else None,
            "avg_order_notional": float(orders["notional"].mean()),
            "median_order_notional": float(orders["notional"].median()),
        },
        "fee_summary": {
            "total_fees": total_fees,
            "buy_fees": float(buy_orders["fees"].sum()),
            "sell_fees": float(sell_orders["fees"].sum()),
            "fee_to_notional_ratio": total_fees / total_notional if total_notional else None,
            "round_trip_fees": total_trade_fees,
        },
        "trade_activity": {
            "annualized_order_count": float(len(orders) / years),
            "monthly_activity": monthly_activity,
            "top_symbols_by_gross_notional": _top_records(symbol_notional, ascending=False),
            "concentration": {
                "top_5_share": float(symbol_share.nlargest(5).sum()),
                "top_10_share": float(symbol_share.nlargest(10).sum()),
                "gross_notional_hhi": float((symbol_share**2).sum()),
            },
        },
        "round_trip_analysis": {
            "available": bool(round_trips),
            "trade_count": int(len(round_trips)),
            "winning_trade_count": winning_trades,
            "losing_trade_count": losing_trades,
            "win_rate": win_rate,
            "total_net_pnl": total_net_pnl,
            "avg_net_pnl_per_trade": total_net_pnl / len(round_trips) if round_trips else None,
            "avg_holding_days": average_holding_days,
            "median_holding_days": median_holding_days,
            "max_holding_days": max_holding_days,
            "monthly_realized_pnl": {
                str(key.strftime("%Y-%m")): float(value) for key, value in monthly_realized_pnl.items()
            },
            "top_winners_by_symbol": _top_records(pnl_by_symbol, ascending=False),
            "top_losers_by_symbol": _top_records(pnl_by_symbol, ascending=True),
        },
        "position_reconciliation": {
            "unmatched_sell_count": int(len(unmatched_sells)),
            "unmatched_sell_quantity": float(sum(item["quantity"] for item in unmatched_sells)),
            "unmatched_sell_notional": float(sum(item["notional"] for item in unmatched_sells)),
            "unmatched_sells": unmatched_sells,
            "open_lot_count": int(len(open_positions)),
            "open_position_symbol_count": int(len({item["symbol"] for item in open_positions})),
            "open_cost_basis": float(sum(item["cost_basis"] for item in open_positions)),
            "open_positions": open_positions,
        },
        "assumptions": [
            "已实现盈亏使用 FIFO 手数匹配。",
            "费用按匹配数量分摊到对应交易。",
            "未匹配卖出只做对账提示，不会强行生成盈亏。",
            "不自动推断公司行为、拆股和分红调整。",
        ],
    }
