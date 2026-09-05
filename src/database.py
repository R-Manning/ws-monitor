"""SQLite persistence for collector samples."""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Dict, Optional, Union


SAMPLE_COLUMNS = ("tempF", "humid", "flueF", "sttF")


class ManagedConnection(sqlite3.Connection):
    """SQLite connection whose context manager also closes the handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect(path: Union[str, Path]) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=10, factory=ManagedConnection)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_ok_columns(conn: sqlite3.Connection) -> None:
    """Add missing *_ok flag columns to an existing stove_room and backfill them."""
    row = conn.execute("PRAGMA table_info(stove_room)").fetchall()
    existing = {entry[1] for entry in row}
    added = [f"{column}_ok" for column in SAMPLE_COLUMNS if f"{column}_ok" not in existing]
    for name in added:
        conn.execute(f"ALTER TABLE stove_room ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0")
    if added:
        for column in SAMPLE_COLUMNS:
            conn.execute(
                f"UPDATE stove_room SET {column}_ok = 1 "
                f"WHERE {column} IS NOT NULL AND {column}_ok = 0"
            )
        conn.commit()


def initialize(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stove_room (
            id INTEGER PRIMARY KEY,
            datetime TEXT NOT NULL,
            tempF REAL, humid REAL, flueF REAL, sttF REAL,
            tempF_ok INTEGER NOT NULL DEFAULT 0,
            humid_ok INTEGER NOT NULL DEFAULT 0,
            flueF_ok INTEGER NOT NULL DEFAULT 0,
            sttF_ok INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_stove_room_datetime ON stove_room(datetime);
        CREATE TABLE IF NOT EXISTS settings (
            sampleFreq INTEGER NOT NULL, dataHist INTEGER NOT NULL,
            graphRange INTEGER NOT NULL, rateDenominator INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS watchDog (
            id INTEGER PRIMARY KEY CHECK (id = 1), emailtimelastsent TIMESTAMP
        );
        INSERT INTO settings(sampleFreq, dataHist, graphRange, rateDenominator)
        SELECT 5, 7, 12, 60 WHERE NOT EXISTS (SELECT 1 FROM settings);
        INSERT INTO watchDog (emailtimelastsent)
        SELECT NULL WHERE NOT EXISTS (SELECT 1 FROM watchDog);
    """)
    conn.commit()
    _ensure_ok_columns(conn)


def diagnose(conn: sqlite3.Connection) -> dict[str, object]:
    """Return non-secret health information for an operator diagnostic."""
    row = conn.execute(
        "SELECT COUNT(*), MAX(datetime) FROM stove_room"
    ).fetchone()
    journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    return {
        "rows": row[0],
        "latest": row[1],
        "journal_mode": journal,
        "integrity": integrity,
    }


def insert_sample(conn: sqlite3.Connection, values: Dict[str, Optional[float]], timestamp: Optional[str] = None) -> None:
    timestamp = timestamp or datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    valid = {column: values.get(column) is not None for column in SAMPLE_COLUMNS}
    conn.execute(
        "INSERT INTO stove_room "
        "(datetime,tempF,humid,flueF,sttF,tempF_ok,humid_ok,flueF_ok,sttF_ok) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (timestamp, *(values.get(column) for column in SAMPLE_COLUMNS),
         *(int(valid[column]) for column in SAMPLE_COLUMNS)),
    )
    conn.commit()


def delete_older_than(conn: sqlite3.Connection, history_days: int) -> int:
    cursor = conn.execute("DELETE FROM stove_room WHERE datetime < datetime('now', ?)", (f"-{int(history_days)} days",))
    conn.commit()
    return cursor.rowcount


def settings(conn: sqlite3.Connection, default_interval: float = 5.0, default_history: int = 7) -> tuple[float, int]:
    row = conn.execute("SELECT sampleFreq, dataHist FROM settings LIMIT 1").fetchone()
    if not row:
        return default_interval, default_history
    try:
        return max(0.1, float(row[0])), max(1, int(row[1] or default_history))
    except (TypeError, ValueError):
        return default_interval, default_history


def sync_runtime_settings(conn: sqlite3.Connection) -> tuple[float, int]:
    """Align the settings table with the declared Config defaults."""
    conn.execute(
        "UPDATE settings SET sampleFreq=?, dataHist=? WHERE NOT "
        "(sampleFreq=? AND dataHist=?)",
        (5, 7, 5, 7),
    )
    conn.commit()
    return settings(conn)
