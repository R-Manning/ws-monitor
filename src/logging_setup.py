"""Logging setup shared by command-line services."""

import logging
from typing import List
import os
from pathlib import Path


def configure_logging(level: str = "INFO") -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    log_dir = os.getenv("WSM_LOG_DIR")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "ws-monitor.log"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("ws-monitor")
