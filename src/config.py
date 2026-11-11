"""Configuration for the environment collector.

This module deliberately has no Raspberry Pi dependencies so it can be used by
development and diagnostic tooling on any machine.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Union

try:
    from dotenv import load_dotenv
except ImportError:  # The collector remains usable with shell environment only.
    load_dotenv = None


@dataclass(frozen=True)
class Config:
    db_path: Path
    sample_interval: float = 5.0
    history_days: int = 7
    sensor_mode: str = "auto"
    log_level: str = "INFO"
    retries: int = 2


def default_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "var" / "ws-monitor.db"


def load_config(db_path: Optional[Union[str, Path]] = None, sensor_mode: Optional[str] = None) -> Config:
    """Load safe defaults and optional environment overrides."""
    if load_dotenv is not None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    path = Path(db_path or os.getenv("WSM_DB_PATH", default_db_path()))
    return Config(
        db_path=path,
        sample_interval=max(0.1, float(os.getenv("WSM_SAMPLE_INTERVAL", "5"))),
        history_days=max(1, int(os.getenv("WSM_HISTORY_DAYS", "7"))),
        sensor_mode=(sensor_mode or os.getenv("WSM_SENSOR_MODE", "auto")).lower(),
        log_level=os.getenv("WSM_LOG_LEVEL", "INFO").upper(),
        retries=max(0, int(os.getenv("WSM_SENSOR_RETRIES", "2"))),
    )
