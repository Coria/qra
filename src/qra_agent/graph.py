from __future__ import annotations

"""LangGraph orchestration for the quantitative backtest review workflow."""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph
from loguru import logger

from .anomaly import detect_anomalies
from .attribution import run_attribution
from .integrity import check_integrity
from .llm import summarize_with_llm
from .metrics import calculate_metrics, load_returns
from .models import AgentState
from .orders import analyze_orders, load_orders
from .rag import search_similar_reports
from .report_parser import parse_report


DEFAULT_LLM_BASE_URL = os.getenv("QRA_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
DEFAULT_LLM_API_KEY = os.getenv("QRA_LLM_API_KEY", "EMPTY")
DEFAULT_LLM_MODEL = os.getenv("QRA_LLM_MODEL", "local-model")
DEFAULT_LLM_TIMEOUT = float(os.getenv("QRA_LLM_TIMEOUT", "300"))
DEFAULT_HISTORY_DIR = os.getenv("QRA_HISTORY_DIR", "workspace/history")


def parse_report_node(state: AgentState) -> AgentState:
    """Parse the QuantStats report and resolve strategy metadata."""
    report = parse_report(state["report_path"])
    metadata: dict[str, Any] = dict(state.get("metadata", {}))
    if state.get("metadata_path"):
        metadata_path = Path(state["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    elif not state.get("bundle_path"):
        sidecar = Path(state["report_path"]).with_suffix(".metadata.json")
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    return {"report": report, "metadata": metadata, "status": "ok"}


def parse_orders_node(state: AgentState) -> AgentState:
    """Load an optional orders table; a missing table is not fatal."""
    orders_path = state.get("orders_path")
    if not orders_path and state.get("orders_frame") is not None:
        return {"orders_frame": state["orders_frame"]}
    if not orders_path:
        return {"orders_frame": None}
    orders = load_orders(orders_path)
    logger.info("Orders parsed | rows={} | columns={}", len(orders), list(orders.columns))
    return {"orders_frame": orders}


def orders_analysis_node(state: AgentState) -> AgentState:
    """Summarize trading activity and FIFO round trips when orders exist."""
    orders = state.get("orders_frame")
    if orders is None:
        return {
            "orders_analysis": {
                "available": False,
                "reason": "orders file was not provided",
            }
        }
    try:
        analysis = analyze_orders(orders)
    except Exception as exc:
        logger.exception("Order analysis failed")
        return {
            "orders_analysis": {
                "available": False,
                "reason": str(exc),
            }
        }
    return {"orders_analysis": analysis}


def integrity_check_node(state: AgentState) -> AgentState:
    """Check metadata and raw data needed to reproduce reported metrics."""
    has_raw_returns = bool(state.get("returns_path") or state.get("returns_frame") is not None)
    integrity = check_integrity(state["report"], state.get("metadata", {}), has_raw_returns=has_raw_returns)
    return {"integrity": integrity, "status": "ok" if integrity["passed"] else "rejected"}


def recompute_metrics_node(state: AgentState) -> AgentState:
    """Recompute metrics after clipping returns to the reported review period."""
    report = state["report"]
    returns = load_returns(state["returns_path"]) if state.get("returns_path") else state["returns_frame"]
    report_start = pd.to_datetime(report.get("period_start"), errors="coerce")
    report_end = pd.to_datetime(report.get("period_end"), errors="coerce")
    if returns is not None and not returns.empty and pd.notna(report_start):
        returns = returns.loc[returns.index >= report_start.normalize()]
    if returns is not None and not returns.empty and pd.notna(report_end):
        returns = returns.loc[returns.index <= report_end.normalize()]
    if returns is not None and returns.empty:
        raise ValueError("returns data is empty after aligning to the report period")
    metrics = calculate_metrics(
        returns,
        periods_per_year=report["periods_per_year"],
        risk_free_rate=report["risk_free_rate"],
    )
    parsed = report["parsed_metrics"]
    diffs: dict[str, float] = {}
    for key, recalculated in metrics.items():
        if key in parsed:
            try:
                parsed_value = float(parsed[key])
                if abs(recalculated - parsed_value) > 1e-6:
                    diffs[key] = recalculated - parsed_value
            except (TypeError, ValueError):
                continue
    return {
        "recalculated_metrics": metrics,
        "metric_diffs": diffs,
        "returns_frame": returns,
        "status": "ok",
    }


def anomaly_detection_node(state: AgentState) -> AgentState:
    """Detect statistical and reporting anomalies in the recomputed series."""
    returns = state.get("returns_frame")
    anomalies = detect_anomalies(state["report"], state.get("recalculated_metrics"), returns)
    return {"anomalies": anomalies, "status": "ok"}


def attribution_node(state: AgentState) -> AgentState:
    """Compute simple period and factor attribution from available data."""
    attribution = run_attribution(state.get("returns_frame"))
    return {"attribution": attribution, "status": "ok"}


def draft_summary_node(state: AgentState) -> AgentState:
    """Create an LLM review draft, with a deterministic fallback on failure."""
    prompt = _build_prompt(state)
    facts = {"prompt": prompt}
    base_url = state.get("llm_base_url", DEFAULT_LLM_BASE_URL)
    api_key = state.get("llm_api_key", DEFAULT_LLM_API_KEY)
    model = state.get("llm_model", DEFAULT_LLM_MODEL)
    timeout = float(state.get("llm_timeout", DEFAULT_LLM_TIMEOUT))
    try:
        draft = summarize_with_llm(facts, base_url, api_key, model, timeout=timeout)
        logger.info("LLM summary generated | model={} | timeout={}s", model, timeout)
    except Exception as exc:
        logger.warning("LLM summary failed; using fallback draft | model={} | timeout={}s | error={}", model, timeout, exc)
        draft = _fallback_draft(state)
    return {"draft": draft, "status": "needs_human_review"}


def human_review_node(state: AgentState) -> AgentState:
    """Keep reviews pending unless an automated pipeline explicitly approves."""
    auto_approved = bool(state.get("auto_approve", False))
    return {
        "human_review": {
            "required": True,
            "auto_approved": auto_approved,
            "verdict": "approved" if auto_approved else "pending",
        },
        "status": "ok" if auto_approved else "needs_human_review",
    }


def reproducibility_check_node(state: AgentState) -> AgentState:
    """Convert integrity blockers into an actionable reproducibility verdict."""
    integrity = state.get("integrity", {})
    return {
        "reproducibility": {
            "passed": bool(integrity.get("passed")),
            "blockers": integrity.get("blockers", []),
            "action": "return_to_strategy_generator_or_author" if not integrity.get("passed") else "pass",
        },
        "status": "rejected" if not integrity.get("passed") else "ok",
    }


def route_after_integrity(state: AgentState) -> str:
    """Route to full analysis only when reproduction inputs are complete."""
    return "recompute_metrics" if state["integrity"]["passed"] else "reproducibility_check"


def _compact_metadata_for_llm(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep LLM input small by truncating large metadata such as universes."""
    compact = dict(metadata)
    universe = compact.pop("universe", None)
    if isinstance(universe, list):
        compact["universe_count"] = len(universe)
        compact["universe_sample"] = universe[:20]
        compact["universe_truncated"] = len(universe) > 20
    elif universe:
        text = str(universe)
        compact["universe"] = text[:200] + ("..." if len(text) > 200 else "")
    return compact


def _style_features(orders_analysis: dict[str, Any]) -> dict[str, Any]:
    """Extract compact trading-style facts from the detailed orders analysis."""
    order_summary = orders_analysis.get("order_summary") or {}
    fee_summary = orders_analysis.get("fee_summary") or {}
    trade_activity = orders_analysis.get("trade_activity") or {}
    concentration = trade_activity.get("concentration") or {}
    round_trip = orders_analysis.get("round_trip_analysis") or {}
    return {
        "annualized_order_count": trade_activity.get("annualized_order_count"),
        "avg_order_notional": order_summary.get("avg_order_notional"),
        "median_order_notional": order_summary.get("median_order_notional"),
        "buy_sell_notional_ratio": order_summary.get("buy_sell_notional_ratio"),
        "fee_to_notional_ratio": fee_summary.get("fee_to_notional_ratio"),
        "top_5_concentration": concentration.get("top_5_share"),
        "top_10_concentration": concentration.get("top_10_share"),
        "gross_notional_hhi": concentration.get("gross_notional_hhi"),
        "trade_count": round_trip.get("trade_count"),
        "trade_win_rate": round_trip.get("win_rate"),
        "avg_holding_days": round_trip.get("avg_holding_days"),
        "median_holding_days": round_trip.get("median_holding_days"),
        "max_holding_days": round_trip.get("max_holding_days"),
    }


def _build_prompt(state: AgentState) -> str:
    """Build a compact, auditable structured prompt for the LLM.

    Data coverage is included so the model does not mistake a filtered series
    for the full backtest period.  The instruction deliberately asks for risk,
    style and improvement analysis instead of enumerating every metric.
    """
    report = state.get("report", {})
    attribution = state.get("attribution", {})
    orders_analysis = state.get("orders_analysis", {})
    trade_activity = orders_analysis.get("trade_activity") or {}
    round_trip = orders_analysis.get("round_trip_analysis") or {}
    monthly_activity = trade_activity.get("monthly_activity") or {}
    top_activity = sorted(
        monthly_activity.items(),
        key=lambda item: item[1].get("gross_notional", 0),
        reverse=True,
    )[:5]
    position_reconciliation = orders_analysis.get("position_reconciliation") or {}
    compact_orders = {
        "available": orders_analysis.get("available"),
        "order_summary": orders_analysis.get("order_summary"),
        "fee_summary": orders_analysis.get("fee_summary"),
        "top_activity_by_gross_notional": [
            {"month": month, "gross_notional": values.get("gross_notional")}
            for month, values in top_activity
        ],
        "concentration": trade_activity.get("concentration"),
        "round_trip_summary": {
            "trade_count": round_trip.get("trade_count"),
            "win_rate": round_trip.get("win_rate"),
            "total_net_pnl": round_trip.get("total_net_pnl"),
            "avg_holding_days": round_trip.get("avg_holding_days"),
            "top_winners_by_symbol": round_trip.get("top_winners_by_symbol"),
            "top_losers_by_symbol": round_trip.get("top_losers_by_symbol"),
        },
        "position_reconciliation": {
            "unmatched_sell_count": position_reconciliation.get("unmatched_sell_count"),
            "open_lot_count": position_reconciliation.get("open_lot_count"),
            "open_cost_basis": position_reconciliation.get("open_cost_basis"),
        },
    }
    style_features = _style_features(orders_analysis)
    compact_facts = {
        "bundle_info": state.get("bundle_info"),
        "metadata": _compact_metadata_for_llm(state.get("metadata") or {}),
        "report_meta": {
            "benchmark": report.get("benchmark"),
            "period_start": report.get("period_start"),
            "period_end": report.get("period_end"),
            "report_version": report.get("report_version"),
            "periods_per_year": report.get("periods_per_year"),
            "risk_free_rate": report.get("risk_free_rate"),
        },
        "data_coverage": {
            "returns_rows": int(state.get("returns_frame").shape[0]) if state.get("returns_frame") is not None else 0,
            "orders_rows": int(state.get("orders_frame").shape[0]) if state.get("orders_frame") is not None else 0,
            "recalculated_observation_count": (state.get("recalculated_metrics") or {}).get("observation_count"),
            "report_period": [report.get("period_start"), report.get("period_end")],
            "returns_period": [
                (state.get("recalculated_metrics") or {}).get("period_start"),
                (state.get("recalculated_metrics") or {}).get("period_end"),
            ],
        },
        "parsed_metrics": report.get("parsed_metrics"),
        "recalculated_metrics": state.get("recalculated_metrics"),
        "metric_diffs": state.get("metric_diffs"),
        "integrity": state.get("integrity"),
        "anomalies": state.get("anomalies"),
        "attribution": {
            "available": attribution.get("available"),
            "monthly_contribution": attribution.get("monthly_contribution"),
            "factor_exposures": attribution.get("factor_exposures"),
            "benchmark_beta": attribution.get("benchmark_beta"),
            "worst_drawdown_date": attribution.get("worst_drawdown_date"),
        },
        "orders_analysis": compact_orders,
        "style_features": style_features,
    }
    query = json.dumps(
        {
            "benchmark": state["report"].get("benchmark"),
            "metrics": state["report"].get("parsed_metrics"),
            "anomalies": state.get("anomalies"),
        },
        ensure_ascii=False,
    )
    similar = search_similar_reports(query, state.get("history_dir", DEFAULT_HISTORY_DIR))
    return json.dumps(
        {
            "facts": compact_facts,
            "historical_context": similar,
            "instruction": "对照相似历史报告，输出可执行改进建议；不要编造未提供的数据。请结合 data_coverage 判断重算样本量，不要把示例数据或前置空仓期误判为真实回测只有少量检查点。"
            "请重点解读交易风格特征、异常警告、风险模式和总体评价，并给出可执行的改进方向。不要逐项罗列指标，最多引用两三个关键数字来支撑判断。",
        },
        ensure_ascii=False,
        default=str,
    )


def build_graph() -> StateGraph:
    """Compile the LangGraph state machine used by CLI and pipeline callers."""
    graph = StateGraph(AgentState)
    graph.add_node("parse_report", parse_report_node)
    graph.add_node("parse_orders", parse_orders_node)
    graph.add_node("orders_analysis", orders_analysis_node)
    graph.add_node("integrity_check", integrity_check_node)
    graph.add_node("reproducibility_check", reproducibility_check_node)
    graph.add_node("recompute_metrics", recompute_metrics_node)
    graph.add_node("anomaly_detection", anomaly_detection_node)
    graph.add_node("attribution", attribution_node)
    graph.add_node("draft_summary", draft_summary_node)
    graph.add_node("human_review", human_review_node)
    graph.set_entry_point("parse_report")
    graph.add_edge("parse_report", "parse_orders")
    graph.add_edge("parse_orders", "orders_analysis")
    graph.add_edge("orders_analysis", "integrity_check")
    graph.add_conditional_edges(
        "integrity_check",
        route_after_integrity,
        {"recompute_metrics": "recompute_metrics", "reproducibility_check": "reproducibility_check"},
    )
    graph.add_edge("recompute_metrics", "anomaly_detection")
    graph.add_edge("anomaly_detection", "attribution")
    graph.add_edge("attribution", "draft_summary")
    graph.add_edge("draft_summary", "human_review")
    graph.add_edge("human_review", END)
    graph.add_edge("reproducibility_check", "draft_summary")
    return graph.compile()


def _fallback_draft(state: AgentState) -> str:
    """Generate a short, conservative draft when the LLM call fails."""
    report = state.get("report") or {}
    integrity = state.get("integrity") or {}
    orders = state.get("orders_analysis") or {}
    order_summary = orders.get("order_summary") or {}
    fee_summary = orders.get("fee_summary") or {}
    round_trip = orders.get("round_trip_analysis") or {}
    parsed_metrics = report.get("parsed_metrics") or {}

    lines = ["## 审查结论（受限模式）", "- LLM 调用失败，以下基于结构化数据生成。"]
    if not integrity.get("passed"):
        missing = "、".join(integrity.get("missing") or [])
        lines.append(f"- 可复现校验未通过，缺少：{missing or '关键输入'}。")
    else:
        lines.append("- 可复现校验通过，可进入指标重算与异常检测。")

    if parsed_metrics:
        lines.append(
            "- 报告指标：年化 {cagr:.2%}，夏普 {sharpe:.2f}，最大回撤 {mdd:.2%}。".format(
                cagr=float(parsed_metrics.get("cagr", 0)),
                sharpe=float(parsed_metrics.get("sharpe", 0)),
                mdd=float(parsed_metrics.get("max_drawdown", 0)),
            )
        )
    if orders.get("available"):
        lines.append(
            "- 订单面：{orders:,} 笔订单、{symbols:,} 个标的；费用/名义额 {fee:.2%}；FIFO 胜率 {win:.2%}。".format(
                orders=int(order_summary.get("order_count", 0)),
                symbols=int(order_summary.get("symbol_count", 0)),
                fee=float(fee_summary.get("fee_to_notional_ratio") or 0),
                win=float(round_trip.get("win_rate") or 0),
            )
        )
    lines.append("- 主要限制：审查依赖回测引擎提供的数据口径，且指标重算使用简化实现。")
    if not integrity.get("passed"):
        lines.append("- 下一步：补充 returns.csv、数据版本、策略版本、参数、品种池和调仓规则后重跑完整审查。")
    else:
        lines.append("- 下一步：请人工核对重算指标差异、异常检测和归因结果后出具结论。")
    return "\n".join(lines)

