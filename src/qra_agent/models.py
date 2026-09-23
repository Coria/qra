"""Typed state shared by every LangGraph node."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    """Carry report, data frames, review facts, settings and final verdict."""
    report_path: str
    bundle_path: str | None
    returns_path: str | None
    orders_path: str | None
    metadata_path: str | None
    auto_approve: bool
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: float
    history_dir: str
    metadata: dict[str, Any]
    bundle: dict[str, Any]
    bundle_info: dict[str, Any]
    report: dict[str, Any]
    parsed_metrics: dict[str, float]
    recalculated_metrics: dict[str, Any]
    metric_diffs: dict[str, float]
    integrity: dict[str, Any]
    reproducibility: dict[str, Any]
    anomalies: dict[str, Any]
    attribution: dict[str, Any]
    orders_analysis: dict[str, Any]
    orders_frame: object
    historical_context: list[dict[str, Any]]
    draft: str
    human_review: dict[str, Any]
    returns_frame: object
    status: Literal["ok", "rejected", "needs_human_review", "error"]
    errors: list[str]
