"""
conftest.py — Shared fixtures and logging setup for all bot2 tests.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest

# ── Ensure bot2 is importable ────────────────────────────────────────────────
BOT2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BOT2_ROOT))

# ── Structured file logging for every test run ──────────────────────────────
LOG_DIR = BOT2_ROOT / "tests" / "logs"
RESULTS_DIR = BOT2_ROOT / "tests" / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_log_file = LOG_DIR / f"test_run_{_timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("bot2_tests")
logger.info("Test session started. Log file: %s", _log_file)


# ── Helper: persist JSON results ────────────────────────────────────────────
def save_results(filename: str, data: dict | list) -> Path:
    out = RESULTS_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", out)
    return out


@pytest.fixture(scope="session")
def results_dir():
    return RESULTS_DIR


@pytest.fixture(scope="session")
def bot2_root():
    return BOT2_ROOT
