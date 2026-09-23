"""Centralized Loguru configuration for CLI and MCP execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


def configure_logging(
    log_file: str | Path | None = None,
    *,
    level: str | None = None,
) -> None:
    """Configure console and rotating-file logs from env vars or arguments."""
    level = level or os.getenv("QRA_LOG_LEVEL", "INFO")
    log_file = Path(log_file or os.getenv("QRA_LOG_FILE", "artifacts/logs/qra.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        log_file,
        level=level,
        rotation="20 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        encoding="utf-8",
    )
    logger.bind(component="qra").info("Logging initialized | path={} | level={}", log_file, level)
