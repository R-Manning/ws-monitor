"""Database-backed current and rate metrics shared by the app and watchdog."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
import sqlite3

from database import connect, initialize

METRIC_COLUMNS = (
    ("flueF", "Flue (F)"),
    ("sttF", "STT (F)"),
    ("tempF", "Room (F)"),
    ("humid", "Humidity (%)"),
)


def _number(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 1)


def get_metrics(
    db_path: Union[str, Path], ensure_schema: bool = True
) -> list[dict]:
    """Return dashboard metrics without importing Dash or pandas."""
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if ensure_schema:
            initialize(conn)
        settings = conn.execute(
            "SELECT sampleFreq, rateDenominator FROM settings LIMIT 1"
        ).fetchone()
        sample_period = max(0.1, float(settings[0])) if settings and settings[0] else 5.0
        rate_window = max(1, int(settings[1])) if settings and settings[1] else 60
        rows_needed = max(2, int(rate_window / sample_period))
        if settings:
            rows_needed += 1
        rows = conn.execute(
            "SELECT datetime, flueF, sttF, tempF, humid "
            "FROM stove_room ORDER BY datetime DESC LIMIT ?",
            (rows_needed,),
        ).fetchall()

    rows.reverse()
    current: dict[str, Optional[float]] = {label: None for _, label in METRIC_COLUMNS}
    rates: dict[str, Optional[float]] = {label: None for _, label in METRIC_COLUMNS}
    if rows:
        for column, label in METRIC_COLUMNS:
            current[label] = _number(rows[-1][column])
        if len(rows) > 1:
            try:
                elapsed = (
                    datetime.fromisoformat(rows[-1]["datetime"])
                    - datetime.fromisoformat(rows[0]["datetime"])
                ).total_seconds() / 60.0
            except (TypeError, ValueError):
                elapsed = 0.0
            if elapsed > 0:
                for column, label in METRIC_COLUMNS:
                    newest = _number(rows[-1][column])
                    oldest = _number(rows[0][column])
                    if newest is not None and oldest is not None:
                        rates[label] = (newest - oldest) / elapsed

    return [
        {"Data-Type": "Current Values", **{k: _rounded(v) for k, v in current.items()}},
        {"Data-Type": "Rate/minute", **{k: _rounded(v) for k, v in rates.items()}},
    ]
