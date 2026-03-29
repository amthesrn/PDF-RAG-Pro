"""Logging utilities (Loguru) used across the project."""

from __future__ import annotations

from pathlib import Path
from loguru import logger

from config.settings import settings


def setup_logger() -> None:
    """Configure Loguru logging for console + rotating file output."""

    # Remove default handlers to avoid duplicate logs
    logger.remove()

    # Console output (color)
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
    )

    # File output (rotating)
    log_dir = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    logger.add(
        log_path,
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )


def get_logger(name: str):
    """Return a logger with a module-specific name."""
    return logger.bind(module=name)
