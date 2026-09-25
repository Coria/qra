"""Command-line entry point for local analysis and MCP tool serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from .html_report import render_review_html
from .logging_setup import configure_logging
from .interactive_report import build_interactive_data, render_interactive_html
from .data_bundle import (
    bundle_info,
    bundle_orders_frame,
    bundle_report_path,
    bundle_returns_frame,
    load_data_bundle,
)
from .graph import (
    DEFAULT_HISTORY_DIR,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
    build_graph,
)


def _parser() -> argparse.ArgumentParser:
    """Define the analyze and serve-mcp subcommands."""
    parser = argparse.ArgumentParser(description="Quantitative strategy backtest review agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Analyze a QuantStats HTML report")
    analyze.add_argument("--report", help="QuantStats HTML report path")
    analyze.add_argument("--bundle", help="Portable backtest bundle JSON path")
    analyze.add_argument("--returns", help="CSV/Excel/Parquet with date,strategy[,benchmark,factor_*]")
    analyze.add_argument("--orders", help="CSV/Excel/Parquet with timestamp,symbol,size,price,side[,fees]")
    analyze.add_argument("--metadata", help="JSON file with data version, parameters and costs")
    analyze.add_argument("--output", default="artifacts/review.json", help="JSON or HTML output path")
    analyze.add_argument("--auto-approve", action="store_true")
    analyze.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    analyze.add_argument("--llm-api-key", default=DEFAULT_LLM_API_KEY)
    analyze.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    analyze.add_argument("--llm-timeout", type=float, default=DEFAULT_LLM_TIMEOUT, help="LLM request timeout in seconds")
    analyze.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR)
    subparsers.add_parser("serve-mcp", help="Run the MCP tool server")
    return parser


def _state(args: argparse.Namespace) -> dict:
    """Convert CLI arguments and bundle data into LangGraph input state."""
    bundle = None
    if args.bundle:
        bundle = load_data_bundle(args.bundle)

    report_path = args.report or (bundle_report_path(bundle) if bundle else None)
    if not report_path:
        raise ValueError("report path is required; pass --report or provide it in --bundle")

    returns_frame = bundle_returns_frame(bundle) if bundle else None
    orders_frame = bundle_orders_frame(bundle) if bundle else None
    metadata = bundle.get("metadata", {}) if bundle else {}
    bundle_path = bundle_info(bundle)

    return {
        "report_path": report_path,
        "bundle_path": args.bundle,
        "returns_path": args.returns,
        "orders_path": args.orders,
        "metadata_path": args.metadata,
        "metadata": metadata,
        "bundle": bundle,
        "bundle_info": bundle_path,
        "returns_frame": returns_frame,
        "orders_frame": orders_frame,
        "auto_approve": args.auto_approve,
        "llm_base_url": args.llm_base_url,
        "llm_api_key": args.llm_api_key,
        "llm_model": args.llm_model,
        "llm_timeout": args.llm_timeout,
        "history_dir": args.history_dir,
    }


def main() -> None:
    """Run the review graph and write JSON or HTML output."""
    parser = _parser()
    args = parser.parse_args()
    configure_logging()
    logger.info("QRA CLI started | command={}", args.command)
    if args.command == "serve-mcp":
        from .mcp_server import run as run_mcp

        run_mcp()
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Analysis started | bundle_path={} | report_path={} | output_path={}", args.bundle, args.report, output_path)
    graph = build_graph()
    final_state = graph.invoke(_state(args))
    logger.info("Analysis finished | status={}", final_state.get("status"))
    serializable = {
        key: value
        for key, value in final_state.items()
        if key not in {"returns_frame", "orders_frame", "bundle"}
    }
    if output_path.suffix.lower() in {".html", ".htm"}:
        interactive_path = output_path.with_name(f"{output_path.stem}_interactive{output_path.suffix.lower()}")
        interactive_data = build_interactive_data(final_state)
        interactive_path.write_text(render_interactive_html(interactive_data), encoding="utf-8")
        print(f"Interactive analysis written to {interactive_path}")
        logger.info("Interactive output written | output_path={} | format=html", interactive_path)
        output_path.write_text(render_review_html(serializable), encoding="utf-8")
        print(f"HTML review written to {output_path}")
        logger.info("Review output written | output_path={} | format=html", output_path)
    else:
        output_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"JSON review written to {output_path} ({serializable.get('status', 'unknown')})")
        logger.info("Review output written | output_path={} | format=json", output_path)


if __name__ == "__main__":
    main()
