"""Render the structured review state as a human-readable HTML report.

The renderer deliberately keeps all review evidence in one page: status cards,
reproducibility blockers, original/recalculated metrics with threshold badges,
anomalies, attribution, order behavior, zoomable SVG charts, and the LLM draft.
SVG tooltips carry additional values so the page does not have to list every
underlying data point.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from html import escape
import json
import math
import re
from typing import Any


PERCENT_KEYS = {
    "cumulative_return",
    "cagr",
    "annual_volatility",
    "max_drawdown",
    "daily_value_at_risk",
    "expected_shortfall",
    "win_days",
    "win_month",
    "correlation",
    "alpha",
    "benchmark_cagr",
    "prob_sharpe_ratio",
    "risk_of_ruin",
    "fee_to_notional_ratio",
    "return_on_cost",
    "top_5_share",
    "top_10_share",
    "gross_notional_hhi",
    "win_rate",
}

INTEGER_KEYS = {
    "observation_count",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "order_count",
    "buy_count",
    "sell_count",
    "symbol_count",
    "trade_count",
    "winning_trade_count",
    "losing_trade_count",
    "open_lot_count",
    "open_position_symbol_count",
    "unmatched_sell_count",
}

METRIC_LABELS = {
    "cumulative_return": "累计收益",
    "cagr": "年化收益",
    "annual_volatility": "年化波动率",
    "sharpe": "夏普比率",
    "sortino": "索提诺比率",
    "calmar": "卡玛比率",
    "max_drawdown": "最大回撤",
    "daily_value_at_risk": "日 VaR",
    "expected_shortfall": "期望损失",
    "profit_factor": "盈利因子",
    "win_days": "日胜率",
    "beta": "Beta",
    "alpha": "Alpha",
    "correlation": "相关性",
    "observation_count": "观测数",
    "benchmark_cagr": "基准年化收益",
    "max_consecutive_wins": "最大连赢",
    "max_consecutive_losses": "最大连亏",
}

METRIC_RULES = {
    "cagr": (0.15, 0.08, "higher"),
    "annual_volatility": (0.15, 0.25, "lower"),
    "sharpe": (1.00, 0.50, "higher"),
    "sortino": (1.50, 0.80, "higher"),
    "calmar": (0.80, 0.40, "higher"),
    "max_drawdown": (-0.15, -0.30, "higher"),
    "daily_value_at_risk": (-0.02, -0.03, "higher"),
    "expected_shortfall": (-0.03, -0.05, "higher"),
    "profit_factor": (1.50, 1.10, "higher"),
    "win_days": (0.55, 0.50, "higher"),
    "win_month": (0.60, 0.50, "higher"),
    "prob_sharpe_ratio": (0.95, 0.80, "higher"),
    "risk_of_ruin": (0.01, 0.05, "lower"),
    "alpha": (0.05, 0.00, "higher"),
    "fee_to_notional_ratio": (0.001, 0.003, "lower"),
    "top_5_share": (0.20, 0.40, "lower"),
    "top_10_share": (0.40, 0.60, "lower"),
}


def _metric_label(key: str) -> str:
    """Convert internal metric keys into Chinese display labels."""
    return METRIC_LABELS.get(key, key.replace("_", " ").title())


def _metric_assessment(key: str, value: Any) -> tuple[str, str, str]:
    """Assess a metric against generic good/watch thresholds.

    Thresholds are presentation-level heuristics, not investment advice or a
    replacement for strategy-specific risk budgets.
    """
    if key not in METRIC_RULES:
        return ("未评估", "neutral", "暂无通用阈值")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ("未评估", "neutral", "数值缺失或无法评估")
    if math.isnan(number):
        return ("未评估", "neutral", "数值缺失或无法评估")

    good, watch, direction = METRIC_RULES[key]
    if direction == "higher":
        is_good = number >= good
        is_watch = number >= watch
    else:
        is_good = number <= good
        is_watch = number <= watch

    if is_good:
        label, css_class = "好", "good"
    elif is_watch:
        label, css_class = "关注", "watch"
    else:
        label, css_class = "弱", "poor"

    if direction == "higher":
        hint = f"参考阈值：≥{_format_metric(key, good)} 好；{_format_metric(key, watch)}~{_format_metric(key, good)} 关注；&lt;{_format_metric(key, watch)} 弱"
    else:
        hint = f"参考阈值：≤{_format_metric(key, good)} 好；{_format_metric(key, good)}~{_format_metric(key, watch)} 关注；&gt;{_format_metric(key, watch)} 弱"
    return label, css_class, hint


def _metric_badge(key: str, value: Any) -> str:
    """Render a colored badge and threshold hint for one metric."""
    label, css_class, hint = _metric_assessment(key, value)
    return (
        f'<span class="badge metric-{css_class}" title="{hint}">{label}</span>'
        f'<small class="metric-rule">{hint}</small>'
    )


def _plain_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1_000_000:
            return f"{value:,.0f}"
        return f"{value:,.4f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)


def _format_value(value: Any) -> str:
    return escape(_plain_value(value))


def _format_metric(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    if key in INTEGER_KEYS and isinstance(value, (int, float)):
        return f"{int(value):,}"
    if key in PERCENT_KEYS and isinstance(value, (int, float)):
        return f"{float(value) * 100:,.2f}%"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1_000_000:
            return f"{value:,.0f}"
        return f"{value:,.4f}"
    return escape(str(value))


def _format_percent(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{float(value) * 100:,.2f}%"


def _render_table(headers: list[str], rows: list[list[Any]], *, raw: bool = False) -> str:
    if not rows:
        return '<p class="muted">无数据</p>'
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{cell if raw else _format_value(cell)}</td>" for cell in row)
        + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>'


def _render_metric_table(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return '<p class="muted">无数据</p>'
    header_html = "".join(f"<th>{escape(header)}</th>" for header in ["字段", "指标", "数值", "快速评价", "参考阈值"])
    body_html = ""
    for key, value in metrics.items():
        label, css_class, hint = _metric_assessment(key, value)
        badge = f'<span class="badge metric-{css_class}" title="{hint}">{label}</span>'
        body_html += (
            "<tr>"
            f"<td>{escape(str(key))}</td>"
            f"<td>{escape(_metric_label(key))}</td>"
            f"<td class='metric-value'>{_format_metric(key, value)}</td>"
            f"<td>{badge}</td>"
            f"<td><small class='metric-rule'>{hint}</small></td>"
            "</tr>"
        )
    return f'<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>'


def _render_metric_comparison(
    parsed_metrics: Mapping[str, Any],
    recalculated_metrics: Mapping[str, Any],
    metric_diffs: Mapping[str, Any],
) -> str:
    all_keys = list(dict.fromkeys([*parsed_metrics.keys(), *recalculated_metrics.keys()]))
    header_html = "".join(
        f"<th>{escape(header)}</th>"
        for header in ["字段", "指标", "报告值", "重算值", "差异", "快速评价", "参考阈值"]
    )
    body_html = ""
    rows = []
    for key in all_keys:
        diff = metric_diffs.get(key)
        assessment_value = recalculated_metrics.get(key, parsed_metrics.get(key))
        label, css_class, hint = _metric_assessment(key, assessment_value)
        badge = f'<span class="badge metric-{css_class}" title="{hint}">{label}</span>'
        body_html += (
            "<tr>"
            f"<td>{escape(str(key))}</td>"
            f"<td>{escape(_metric_label(key))}</td>"
            f"<td class='metric-value'>{_format_metric(key, parsed_metrics.get(key))}</td>"
            f"<td class='metric-value'>{_format_metric(key, recalculated_metrics.get(key))}</td>"
            f"<td class='metric-value'>{_format_metric(key, diff) if diff is not None else '-'}</td>"
            f"<td>{badge}</td>"
            f"<td><small class='metric-rule'>{hint}</small></td>"
            "</tr>"
        )
    return f'<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>'


def _render_key_value_pairs(items: Mapping[str, Any]) -> str:
    rows = [[key, _format_value(value)] for key, value in items.items()]
    return _render_table(["项目", "数值"], rows)


def _render_status_badge(status: str | None) -> str:
    status_text = status or "unknown"
    css_class = {
        "ok": "status ok",
        "rejected": "status rejected",
        "needs_human_review": "status pending",
    }.get(status_text, "status")
    label = {
        "ok": "通过",
        "rejected": "已打回",
        "needs_human_review": "待人审",
    }.get(status_text, status_text)
    return f'<span class="{css_class}">{escape(label)}</span>'


def _render_markdown(text: str | None) -> str:
    if not text:
        return '<p class="muted">暂无草稿</p>'
    rendered_lines: list[str] = []
    in_list = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            if in_list:
                rendered_lines.append("</ul>")
                in_list = False
            level = min(4, len(line) - len(line.lstrip("#")) + 2)
            content = _inline_markdown(line.lstrip("#").strip())
            rendered_lines.append(f"<h{level}>{content}</h{level}>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                rendered_lines.append("<ul>")
                in_list = True
            content = _inline_markdown(line[2:].strip())
            rendered_lines.append(f"<li>{content}</li>")
        else:
            if in_list:
                rendered_lines.append("</ul>")
                in_list = False
            if line:
                rendered_lines.append(f"<p>{_inline_markdown(line)}</p>")
    if in_list:
        rendered_lines.append("</ul>")
    return f'<div class="markdown">{"".join(rendered_lines)}</div>'


def _inline_markdown(text: str) -> str:
    escaped = escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped


def _render_summary_cards(review: Mapping[str, Any]) -> str:
    report = review.get("report") or {}
    integrity = review.get("integrity") or {}
    orders = review.get("orders_analysis") or {}
    human_review = review.get("human_review") or {}
    cards = [
        ("状态", _render_status_badge(review.get("status"))),
        ("基准", _format_value(report.get("benchmark"))),
        ("报告期间", f"{_format_value(report.get('period_start'))} 至 {_format_value(report.get('period_end'))}"),
        ("可复现校验", "通过" if integrity.get("passed") else "未通过"),
        ("订单分析", "可用" if orders.get("available") else "不可用"),
        ("人审状态", _format_value(human_review.get("verdict"))),
    ]
    card_html = "".join(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div></div>' for label, value in cards)
    return f'<section class="cards">{card_html}</section>'


def _render_reproducibility(review: Mapping[str, Any]) -> str:
    integrity = review.get("integrity") or {}
    reproducibility = review.get("reproducibility") or {}
    blockers = integrity.get("blockers") or []
    rows = [[item.get("field"), item.get("reason")] for item in blockers]
    blockers_html = _render_table(["缺失字段", "原因"], rows)
    action = reproducibility.get("action")
    action_html = f'<p class="highlight">处理建议：{escape(str(action))}</p>' if action else ""
    metadata = review.get("metadata") or {}
    metadata_html = ""
    if metadata:
        metadata_rows = [[key, _metadata_value_cell(key, value)] for key, value in metadata.items()]
        metadata_html = '<h3>审查输入元数据</h3>' + _render_table(["字段", "值"], metadata_rows, raw=True)
    return f'<section><h2>可复现校验</h2>{action_html}{blockers_html}{metadata_html}</section>'


def _metadata_value_cell(key: str, value: Any) -> str:
    if key == "universe" and isinstance(value, (list, tuple)):
        symbols = [str(item) for item in value]
        if len(symbols) > 12:
            preview = "、".join(symbols[:10])
            chips = "".join(
                f'<span class="chip">{escape(symbol)}</span>'
                for symbol in symbols
            )
            return (
                f'<details><summary>{escape(preview)} 等 {len(symbols)} 个品种</summary>'
                f'<div class="chip-grid">{chips}</div></details>'
            )
    return _format_value(value)


def _render_metrics(review: Mapping[str, Any]) -> str:
    report = review.get("report") or {}
    parsed_metrics = report.get("parsed_metrics") or {}
    recalculated_metrics = review.get("recalculated_metrics") or {}
    metric_diffs = review.get("metric_diffs") or {}
    assessments = [_metric_assessment(key, value) for key, value in parsed_metrics.items()]
    counts = {
        "好": sum(1 for item in assessments if item[1] == "good"),
        "关注": sum(1 for item in assessments if item[1] == "watch"),
        "弱": sum(1 for item in assessments if item[1] == "poor"),
    }
    count_html = (
        '<div class="metric-summary">'
        f'<span class="badge metric-good">好 {counts["好"]}</span>'
        f'<span class="badge metric-watch">关注 {counts["关注"]}</span>'
        f'<span class="badge metric-poor">弱 {counts["弱"]}</span>'
        '<small class="muted">阈值是常见日线策略的粗略参考，需结合基准、风格和市场阶段解读。</small>'
        '</div>'
    )
    parsed_html = f"<h3>报告解析指标</h3>{count_html}{_render_metric_table(parsed_metrics)}"
    if recalculated_metrics:
        comparison_html = f"<h3>重算值与差异</h3>{_render_metric_comparison(parsed_metrics, recalculated_metrics, metric_diffs)}"
    else:
        comparison_html = '<p class="muted">缺少原始收益数据，未执行指标重算。</p>'
    bundle_info = review.get("bundle_info") or {}
    dataset_names = ", ".join(bundle_info.get("datasets") or [])
    coverage_rows = [
        ["Bundle ID", bundle_info.get("bundle_id")],
        ["引擎", bundle_info.get("engine")],
        ["数据集", dataset_names],
        ["重算观测数", recalculated_metrics.get("observation_count")],
        ["报告期", [report.get("period_start"), report.get("period_end")]],
        ["重算期", [recalculated_metrics.get("period_start"), recalculated_metrics.get("period_end")]],
    ]
    coverage_html = (
        '<details><summary>数据覆盖</summary>'
        + _render_table(["项目", "值"], coverage_rows)
        + '<p class="muted">重算已按报告期裁剪前置空仓期；如果重算期与报告期不一致，请优先检查 bundle 数据版本。</p>'
        + "</details>"
    )
    return f'<section><h2>指标</h2>{parsed_html}{comparison_html}{coverage_html}</section>'


def _render_anomalies(review: Mapping[str, Any]) -> str:
    anomalies = review.get("anomalies") or {}
    issues = anomalies.get("issues") or []
    rows = [[item.get("severity"), item.get("message")] for item in issues]
    return f'<section><h2>异常检测</h2>{_render_table(["级别", "说明"], rows)}</section>'


def _render_attribution(review: Mapping[str, Any]) -> str:
    attribution = review.get("attribution") or {}
    if not attribution.get("available"):
        reason = attribution.get("reason") or "原始收益数据不可用"
        return f'<section><h2>归因分析</h2><p class="muted">{escape(str(reason))}</p></section>'
    monthly = attribution.get("monthly_contribution") or {}
    rows = [
        ["正收益月份数", monthly.get("positive_months")],
        ["负收益月份数", monthly.get("negative_months")],
        ["正收益月均值", monthly.get("positive_month_mean")],
        ["负收益月均值", monthly.get("negative_month_mean")],
        ["最大回撤日期", attribution.get("worst_drawdown_date")],
        ["基准 Beta", attribution.get("benchmark_beta")],
    ]
    return f'<section><h2>归因分析</h2>{_render_table(["项目", "数值"], rows)}</section>'


def _render_orders(review: Mapping[str, Any]) -> str:
    orders = review.get("orders_analysis") or {}
    if not orders.get("available"):
        reason = orders.get("reason") or "未提供订单数据"
        return f'<section><h2>订单分析</h2><p class="muted">{escape(str(reason))}</p></section>'

    order_summary = orders.get("order_summary") or {}
    fee_summary = orders.get("fee_summary") or {}
    trade_activity = orders.get("trade_activity") or {}
    round_trip = orders.get("round_trip_analysis") or {}
    reconciliation = orders.get("position_reconciliation") or {}

    summary_rows = [
        ["订单数量", order_summary.get("order_count")],
        ["买入订单", order_summary.get("buy_count")],
        ["卖出订单", order_summary.get("sell_count")],
        ["标的数量", order_summary.get("symbol_count")],
        ["总名义成交额", order_summary.get("total_notional")],
        ["买入名义额", order_summary.get("buy_notional")],
        ["卖出名义额", order_summary.get("sell_notional")],
        ["年化名义成交额", order_summary.get("annualized_gross_notional")],
        ["买卖名义额比", order_summary.get("buy_sell_notional_ratio")],
        ["平均订单名义额", order_summary.get("avg_order_notional")],
        ["中位订单名义额", order_summary.get("median_order_notional")],
    ]
    health_rows = [
        ("费用/名义成交额", "fee_to_notional_ratio", fee_summary.get("fee_to_notional_ratio")),
        ("FIFO 胜率", "win_rate", round_trip.get("win_rate")),
        ("Top 5 集中度", "top_5_share", (trade_activity.get("concentration") or {}).get("top_5_share")),
        ("Top 10 集中度", "top_10_share", (trade_activity.get("concentration") or {}).get("top_10_share")),
    ]
    health_html = "".join(
        f'<div class="metric-tile"><div class="metric-tile-label">{escape(label)}</div>'
        f'<div class="metric-tile-value">{_format_metric(key, value)}</div>'
        f'{_metric_badge(key, value)}</div>'
        for label, key, value in health_rows
    )
    health_section = f'<h3>订单指标体检</h3><div class="metric-tiles">{health_html}</div>'
    fee_rows = [
        ["总费用", fee_summary.get("total_fees")],
        ["买入费用", fee_summary.get("buy_fees")],
        ["卖出费用", fee_summary.get("sell_fees")],
        ["费用/名义成交额", _format_percent(fee_summary.get("fee_to_notional_ratio"))],
        ["已匹配交易费用", fee_summary.get("round_trip_fees")],
    ]
    concentration = trade_activity.get("concentration") or {}
    concentration_rows = [
        ["Top 5 集中度", _format_percent(concentration.get("top_5_share"))],
        ["Top 10 集中度", _format_percent(concentration.get("top_10_share"))],
        ["名义额 HHI", _format_value(concentration.get("gross_notional_hhi"))],
    ]
    round_trip_rows = [
        ["已实现交易数", round_trip.get("trade_count")],
        ["盈利交易数", round_trip.get("winning_trade_count")],
        ["亏损交易数", round_trip.get("losing_trade_count")],
        ["胜率", _format_percent(round_trip.get("win_rate"))],
        ["总净盈亏", round_trip.get("total_net_pnl")],
        ["单笔平均净盈亏", round_trip.get("avg_net_pnl_per_trade")],
        ["平均持仓天数", round_trip.get("avg_holding_days")],
        ["中位持仓天数", round_trip.get("median_holding_days")],
        ["最长持仓天数", round_trip.get("max_holding_days")],
    ]
    reconciliation_rows = [
        ["未匹配卖出数", reconciliation.get("unmatched_sell_count")],
        ["未匹配卖出数量", reconciliation.get("unmatched_sell_quantity")],
        ["未匹配卖出名义额", reconciliation.get("unmatched_sell_notional")],
        ["未平仓 lot 数", reconciliation.get("open_lot_count")],
        ["未平仓标的数", reconciliation.get("open_position_symbol_count")],
        ["未平仓成本", reconciliation.get("open_cost_basis")],
    ]

    winners = round_trip.get("top_winners_by_symbol") or []
    losers = round_trip.get("top_losers_by_symbol") or []
    winners_rows = [[item.get("symbol"), item.get("value")] for item in winners]
    losers_rows = [[item.get("symbol"), item.get("value")] for item in losers]

    monthly_activity = trade_activity.get("monthly_activity") or []
    monthly_rows = [
        [month, values.get("order_count"), values.get("buy_notional"), values.get("sell_notional"), values.get("gross_notional"), values.get("fees")]
        for month, values in monthly_activity.items()
    ]
    monthly_details = (
        '<details><summary>展开全部月度交易活动</summary>'
        + _render_table(["月份", "订单数", "买入名义额", "卖出名义额", "总名义额", "费用"], monthly_rows)
        + "</details>"
    )
    monthly_realized_pnl = round_trip.get("monthly_realized_pnl") or {}
    monthly_realized_rows = [[month, value] for month, value in monthly_realized_pnl.items()]
    monthly_realized_details = (
        '<details><summary>展开全部月度已实现盈亏</summary>'
        + _render_table(["月份", "净盈亏"], monthly_realized_rows)
        + "</details>"
    )

    assumptions = orders.get("assumptions") or []
    assumption_items = "".join(f"<li>{escape(str(item))}</li>" for item in assumptions)
    assumptions_html = f'<div class="note"><strong>口径说明</strong><ul>{assumption_items}</ul></div>'

    return (
        '<section><h2>订单分析</h2>'
        + health_section
        + '<div class="grid-2"><div><h3>交易摘要</h3>' + _render_table(["项目", "数值"], summary_rows) + "</div>"
        + '<div><h3>费用</h3>' + _render_table(["项目", "数值"], fee_rows) + "</div></div>"
        '<div class="grid-2"><div><h3>已实现交易</h3>' + _render_table(["项目", "数值"], round_trip_rows) + "</div>"
        '<div><h3>集中度</h3>' + _render_table(["项目", "数值"], concentration_rows)
        + '<h4>Top 盈利品种</h4>' + _render_table(["品种", "净盈亏"], winners_rows)
        + '<h4>Top 亏损品种</h4>' + _render_table(["品种", "净盈亏"], losers_rows) + "</div></div>"
        '<h3>持仓与对账</h3>' + _render_table(["项目", "数值"], reconciliation_rows)
        + monthly_details
        + monthly_realized_details
        + assumptions_html
        + "</section>"
    )


def _render_human_review(review: Mapping[str, Any]) -> str:
    human_review = review.get("human_review") or {}
    rows = [
        ["是否需要人审", human_review.get("required")],
        ["是否自动通过", human_review.get("auto_approved")],
        ["当前结论", human_review.get("verdict")],
    ]
    return f'<section><h2>人审</h2>{_render_table(["项目", "状态"], rows)}</section>'


def _zoomable_chart(title: str, svg: str, legend: str = "") -> str:
    controls = (
        '<div class="zoom-controls">'
        '<button type="button" data-zoom-action="out" aria-label="缩小">−</button>'
        '<button type="button" data-zoom-action="in" aria-label="放大">＋</button>'
        '<button type="button" data-zoom-action="reset" aria-label="重置">1:1</button>'
        "</div>"
    )
    return (
        '<div class="chart zoomable">'
        f'<div class="chart-header"><div class="chart-title">{escape(title)}</div>{controls}</div>'
        '<div class="zoom-stage">'
        f"{svg}"
        "</div>"
        f"{legend}"
        "</div>"
    )


def _render_monthly_activity_chart(activity: Mapping[str, Mapping[str, Any]]) -> str:
    if not activity:
        return '<p class="muted">暂无月度交易活动</p>'
    width, height = 1000, 300
    left, right, top, bottom = 70, 20, 20, 50
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(
        max(item.get("buy_notional", 0), item.get("sell_notional", 0))
        for item in activity.values()
    )
    max_value = max(1.0, float(max_value))
    group_width = plot_width / len(activity)
    bar_width = max(2.0, min(14.0, group_width * 0.32))
    bars: list[str] = []
    labels: list[str] = []
    label_step = max(1, math.ceil(len(activity) / 18))
    for index, (month, values) in enumerate(activity.items()):
        buy_notional = float(values.get("buy_notional", 0) or 0)
        sell_notional = float(values.get("sell_notional", 0) or 0)
        gross_notional = float(values.get("gross_notional", buy_notional + sell_notional) or 0)
        buy_sell_ratio = buy_notional / sell_notional if sell_notional else None
        tooltip = " | ".join(
            [
                escape(str(month)),
                f"订单数 {values.get('order_count', 0)}",
                f"买入 {_format_value(buy_notional)}",
                f"卖出 {_format_value(sell_notional)}",
                f"总名义额 {_format_value(gross_notional)}",
                f"费用 {_format_value(values.get('fees', 0))}",
                f"买卖比 {_format_value(buy_sell_ratio)}",
            ]
        )
        group_x = left + index * group_width
        buy_height = plot_height * buy_notional / max_value
        sell_height = plot_height * sell_notional / max_value
        buy_x = group_x + group_width / 2 - bar_width - 1
        sell_x = group_x + group_width / 2 + 1
        bars.append(
            f'<rect x="{buy_x:.1f}" y="{top + plot_height - buy_height:.1f}" width="{bar_width:.1f}" height="{buy_height:.1f}" fill="#2563eb" rx="1"><title>{tooltip}</title></rect>'
        )
        bars.append(
            f'<rect x="{sell_x:.1f}" y="{top + plot_height - sell_height:.1f}" width="{bar_width:.1f}" height="{sell_height:.1f}" fill="#dc2626" rx="1"><title>{tooltip}</title></rect>'
        )
        if index % label_step == 0:
            labels.append(
                f'<text x="{group_x + group_width / 2:.1f}" y="{height - bottom + 18}" text-anchor="middle" font-size="10" fill="#6b7280">{escape(month)}</text>'
            )
    grid_lines = []
    for ratio in (0, 0.25, 0.5, 0.75, 1):
        y = top + plot_height * (1 - ratio)
        value = max_value * ratio
        grid_lines.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#6b7280">{_format_value(value)}</text>'
        )
    svg = f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(grid_lines)}{"".join(bars)}{"".join(labels)}</svg>'
    legend = '<div class="legend"><span><i style="background:#2563eb"></i>买入名义额</span><span><i style="background:#dc2626"></i>卖出名义额</span></div>'
    return _zoomable_chart("月度交易活跃度", svg, legend)


def _render_monthly_pnl_chart(monthly_pnl: Mapping[str, float]) -> str:
    if not monthly_pnl:
        return '<p class="muted">暂无月度已实现盈亏</p>'
    width, height = 1000, 300
    left, right, top, bottom = 70, 20, 20, 50
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(value) for value in monthly_pnl.values()]
    min_value = min(0.0, min(values))
    max_value = max(0.0, max(values))
    value_range = max(1.0, max_value - min_value)

    def y_position(value: float) -> float:
        return top + (max_value - value) / value_range * plot_height

    group_width = plot_width / len(monthly_pnl)
    bar_width = max(2.0, min(20.0, group_width * 0.62))
    bars: list[str] = []
    labels: list[str] = []
    label_step = max(1, math.ceil(len(monthly_pnl) / 18))
    zero_y = y_position(0)
    cumulative = 0.0
    total = float(sum(values))
    for index, (month, value) in enumerate(monthly_pnl.items()):
        cumulative += float(value)
        share = float(value) / total if total else None
        tooltip = " | ".join(
            [
                escape(str(month)),
                f"净盈亏 {_format_value(value)}",
                f"累计 {_format_value(cumulative)}",
                f"占总盈亏 {_format_value(share)}",
            ]
        )
        group_x = left + index * group_width
        y = y_position(float(value))
        bar_height = abs(zero_y - y)
        fill = "#16a34a" if float(value) >= 0 else "#dc2626"
        x = group_x + group_width / 2 - bar_width / 2
        bars.append(
            f'<rect x="{x:.1f}" y="{min(y, zero_y):.1f}" width="{bar_width:.1f}" height="{max(1.0, bar_height):.1f}" fill="{fill}" rx="1"><title>{tooltip}</title></rect>'
        )
        if index % label_step == 0:
            labels.append(
                f'<text x="{group_x + group_width / 2:.1f}" y="{height - bottom + 18}" text-anchor="middle" font-size="10" fill="#6b7280">{escape(month)}</text>'
            )
    grid_lines = []
    for ratio in (0, 0.25, 0.5, 0.75, 1):
        value = min_value + value_range * ratio
        y = y_position(value)
        grid_lines.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#6b7280">{_format_value(value)}</text>'
        )
    svg = f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(grid_lines)}{"".join(bars)}{"".join(labels)}</svg>'
    legend = '<div class="legend"><span><i style="background:#16a34a"></i>盈利</span><span><i style="background:#dc2626"></i>亏损</span></div>'
    return _zoomable_chart("月度已实现盈亏", svg, legend)


def _render_top_symbols_chart(records: list[Mapping[str, Any]], title: str, color: str) -> str:
    if not records:
        return f'<div class="chart"><div class="chart-title">{escape(title)}</div><p class="muted">暂无数据</p></div>'
    width, height = 480, max(220, 50 + len(records) * 34)
    left, right, top = 120, 60, 20
    plot_width = width - left - right
    max_abs = max(1.0, max(abs(float(item.get("value", 0))) for item in records))
    bars: list[str] = []
    top_total = sum(abs(float(item.get("value", 0))) for item in records)
    for index, item in enumerate(records):
        y = top + index * 34
        value = float(item.get("value", 0))
        share = abs(value) / top_total if top_total else None
        tooltip = " | ".join(
            [
                escape(str(item.get("symbol", "-"))),
                f"金额 {_format_value(value)}",
                f"Top 排名 {index + 1}",
                f"占Top合计 {_format_value(share)}",
            ]
        )
        bar_width = plot_width * abs(value) / max_abs
        bars.append(
            f'<text x="{left - 8}" y="{y + 16}" text-anchor="end" font-size="11" fill="#374151">{escape(str(item.get("symbol", "-")))}</text>'
            f'<rect x="{left}" y="{y}" width="{max(2.0, bar_width):.1f}" height="20" fill="{color}" rx="4"><title>{tooltip}</title></rect>'
            f'<text x="{left + max(2.0, bar_width) + 6:.1f}" y="{y + 15}" font-size="11" fill="#374151">{_format_value(value)}</text>'
        )
    return (
        f'<div class="chart"><div class="chart-title">{escape(title)}</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(bars)}</svg></div>'
    )


def _render_visualizations(review: Mapping[str, Any]) -> str:
    """Compose zoomable monthly activity, PnL and top-symbol charts."""
    orders = review.get("orders_analysis") or {}
    if not orders.get("available"):
        reason = orders.get("reason") or "未提供订单数据"
        return f'<section><h2>可视化分析</h2><p class="muted">{escape(str(reason))}</p></section>'
    trade_activity = orders.get("trade_activity") or {}
    round_trip = orders.get("round_trip_analysis") or {}
    activity_chart = _render_monthly_activity_chart(trade_activity.get("monthly_activity") or {})
    pnl_chart = _render_monthly_pnl_chart(round_trip.get("monthly_realized_pnl") or {})
    winners_chart = _render_top_symbols_chart(round_trip.get("top_winners_by_symbol") or [], "Top 盈利品种", "#16a34a")
    losers_chart = _render_top_symbols_chart(round_trip.get("top_losers_by_symbol") or [], "Top 亏损品种", "#dc2626")
    return (
        '<section><h2>可视化分析</h2>'
        f'{activity_chart}{pnl_chart}'
        '<div class="chart-grid">'
        f'{winners_chart}{losers_chart}'
        '</div></section>'
    )


def _render_llm_section(review: Mapping[str, Any]) -> str:
    draft = review.get("draft")
    if not draft:
        return '<section><h2>LLM 审查意见</h2><p class="muted">LLM 未返回内容，或当前运行未进入草稿节点。</p></section>'
    return f'<section><h2>LLM 审查意见</h2>{_render_markdown(draft)}</section>'


_CHART_SCRIPT = """
<script>
(() => {
  document.querySelectorAll('.zoomable').forEach((container) => {
    const stage = container.querySelector('.zoom-stage');
    const svg = container.querySelector('svg');
    if (!stage || !svg) return;

    let scale = 1;
    let tx = 0;
    let ty = 0;
    let dragging = false;
    let startX = 0;
    let startY = 0;
    let startTx = 0;
    let startTy = 0;

    const render = () => {
      svg.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    };

    const zoomTo = (nextScale) => {
      scale = Math.min(5, Math.max(0.5, nextScale));
      if (scale === 1) { tx = 0; ty = 0; }
      render();
    };

    const zoomBy = (factor) => zoomTo(scale * factor);

    container.querySelectorAll('[data-zoom-action]').forEach((button) => {
      button.addEventListener('click', () => {
        const action = button.dataset.zoomAction;
        if (action === 'in') zoomBy(1.2);
        if (action === 'out') zoomBy(0.8);
        if (action === 'reset') zoomTo(1);
      });
    });

    stage.addEventListener('wheel', (event) => {
      event.preventDefault();
      zoomBy(event.deltaY < 0 ? 1.12 : 0.89);
    }, { passive: false });

    stage.addEventListener('pointerdown', (event) => {
      dragging = true;
      startX = event.clientX;
      startY = event.clientY;
      startTx = tx;
      startTy = ty;
      stage.setPointerCapture(event.pointerId);
    });

    stage.addEventListener('pointermove', (event) => {
      if (!dragging) return;
      tx = startTx + event.clientX - startX;
      ty = startTy + event.clientY - startY;
      render();
    });

    stage.addEventListener('pointerup', () => { dragging = false; });
    stage.addEventListener('pointercancel', () => { dragging = false; });
  });
})();
</script>
"""


def render_review_html(review: Mapping[str, Any]) -> str:
    """Render the full review report as a self-contained HTML page."""
    report = review.get("report") or {}
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "Quantitative Backtest Review"
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f7fb; --card:#fff; --text:#1f2937; --muted:#6b7280; --line:#e5e7eb; --primary:#2563eb; --ok:#16a34a; --warn:#d97706; --bad:#dc2626; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; line-height:1.55; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 20px 60px; }}
    h1 {{ font-size:28px; margin:0 0 6px; }}
    h2 {{ font-size:21px; margin:34px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line); }}
    h3 {{ font-size:17px; margin:20px 0 10px; }}
    h4 {{ font-size:15px; margin:16px 0 8px; }}
    section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; margin-bottom:18px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .meta {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:24px 0 0; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .card-label {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
    .card-value {{ font-size:16px; font-weight:600; word-break:break-word; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th, td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ color:var(--muted); font-weight:600; white-space:nowrap; }}
    tr:last-child td {{ border-bottom:none; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:13px; }}
    .muted {{ color:var(--muted); }}
    .highlight {{ background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:12px 14px; }}
    .note {{ background:#f8fafc; border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-top:16px; }}
    .status {{ display:inline-block; border-radius:999px; padding:3px 10px; font-size:13px; font-weight:700; background:#e5e7eb; color:#374151; }}
    .status.ok {{ background:#dcfce7; color:var(--ok); }}
    .status.rejected {{ background:#fee2e2; color:var(--bad); }}
    .status.pending {{ background:#fef3c7; color:var(--warn); }}
    .badge {{ display:inline-block; min-width:44px; text-align:center; border-radius:999px; padding:3px 10px; font-size:12px; font-weight:700; }}
    .metric-good {{ background:#dcfce7; color:var(--ok); }}
    .metric-watch {{ background:#fef3c7; color:var(--warn); }}
    .metric-poor {{ background:#fee2e2; color:var(--bad); }}
    .metric-neutral {{ background:#e5e7eb; color:var(--muted); }}
    .metric-rule {{ display:block; color:var(--muted); font-size:11px; margin-top:3px; white-space:normal; }}
    .metric-summary {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:14px; }}
    .metric-value {{ font-variant-numeric: tabular-nums; font-weight:600; }}
    .metric-tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:18px; }}
    .metric-tile {{ border:1px solid var(--line); border-radius:12px; padding:14px; background:#fff; }}
    .metric-tile-label {{ color:var(--muted); font-size:12px; margin-bottom:6px; }}
    .metric-tile-value {{ font-size:18px; font-weight:700; font-variant-numeric:tabular-nums; margin-bottom:8px; }}
    td .metric-rule {{ margin-top:4px; }}
    .grid-2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:20px; }}
    .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:20px; margin-top:18px; }}
    .chart {{ border:1px solid var(--line); border-radius:12px; padding:16px; background:#fff; }}
    .chart-header {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }}
    .chart-title {{ font-weight:600; }}
    .zoom-controls {{ display:flex; gap:6px; }}
    .zoom-controls button {{ border:1px solid var(--line); background:#fff; border-radius:8px; padding:4px 9px; font-size:12px; cursor:pointer; }}
    .zoom-controls button:hover {{ background:#f3f4f6; }}
    .zoom-stage {{ overflow:hidden; cursor:grab; border-radius:10px; touch-action:none; }}
    .zoom-stage:active {{ cursor:grabbing; }}
    .zoom-stage svg {{ transition:transform .12s ease-out; transform-origin:center center; }}
    .chip-grid {{ display:flex; flex-wrap:wrap; gap:6px; max-height:280px; overflow:auto; margin-top:10px; }}
    .chip {{ background:#eef2ff; border:1px solid #dbeafe; border-radius:999px; padding:3px 8px; font-size:12px; white-space:nowrap; }}
    .chart svg {{ width:100%; height:auto; display:block; }}
    .legend {{ display:flex; gap:16px; color:var(--muted); font-size:12px; margin-top:8px; }}
    .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
    .legend i {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}
    .markdown h2 {{ border:none; margin:18px 0 8px; }}
    .markdown h3 {{ margin:16px 0 8px; }}
    details {{ border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-top:16px; background:#fff; }}
    summary {{ cursor:pointer; font-weight:600; }}
    footer {{ color:var(--muted); font-size:12px; text-align:center; margin-top:30px; }}
  </style>
</head>
<body>
<main>
  <h1>{escape(title)}</h1>
  <div class="meta">生成时间：{escape(generated_at)}　|　报告文件：{escape(str(review.get('report_path', '-')))}　|　订单文件：{escape(str(review.get('orders_path') or '-'))}</div>
  {_render_summary_cards(review)}
  {_render_llm_section(review)}
  {_render_reproducibility(review)}
  {_render_visualizations(review)}
  {_render_metrics(review)}
  {_render_anomalies(review)}
  {_render_attribution(review)}
  {_render_orders(review)}
  {_render_human_review(review)}
  <footer>Generated by qra-agent</footer>
</main>
  {_CHART_SCRIPT}
</body>
</html>'''
