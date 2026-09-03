import datetime as dt
import time
from typing import Dict, List, Tuple, Set, Optional

"""including this here for testing purposes----

from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
"""
try:
    from telegramalert import send_message
except ImportError:  # Alerting is optional and must not disable acquisition.
    def send_message(message: str) -> bool:
        return False

from metrics import get_metrics
from database import connect, initialize
from paths import get_db_path
import logging

logger = logging.getLogger("ws-monitor.watchdog")

# === Config ===
FAIL_THRESHOLD = 12                 # consecutive bad reads before alert
ALERT_COOLDOWN_SEC = 2.5 * 60        # per-sensor cooldown in sensor_health_check
REPORT_RECOVERY = True              # send "[RECOVERY]" when sensor becomes OK

ALERT_DELAY_SEC = int(2.5 * 60)     # main watchdog message min spacing

DB_NAME = str(get_db_path())
TABLE = "watchDog"                  # <- constant/whitelist identifiers
COL_LAST_SENT = "emailtimelastsent"

# Metrics keys used by update_metrics()
# metrics[0] = current; metrics[1] = rate/min
K_FLUE = "Flue (F)"
K_STT = "STT (F)"
K_ROOM = "Room (F)"
K_HUM  = "Humidity (%)"

# === State for sensor_health_check ===
failure_counts: Dict[str, int] = {'tempF': 0, 'humid': 0, 'flueF': 0, 'sttF': 0}
last_alert_time: Dict[str, float] = {k: 0.0 for k in failure_counts}
in_failure_state: Dict[str, bool] = {k: False for k in failure_counts}

_monotonic = time.monotonic  # localize for tiny perf win


# ---------- DB helpers ----------
def _get_last_sent() -> dt.datetime:
    """Returns last send time (localtime) stored in DB."""
    # Identifiers are constants defined above (safe); values are parameterized when present.
    with connect(DB_NAME) as conn:
        initialize(conn)
        cur = conn.execute(f"SELECT {COL_LAST_SENT} FROM {TABLE} LIMIT 1")
        row = cur.fetchone()
    # If the table is guaranteed to have one row, row[0] is a datetime string; normalize to aware-free datetime
    # SQLite returns str for datetime('now','localtime'); parse with fromisoformat if needed.
    if row is None or row[0] is None:
        return dt.datetime.min
    val = row[0]
    if isinstance(val, dt.datetime):
        return val
    # Attempt ISO 8601 parse; fallback to "a while ago" if needed.
    try:
        return dt.datetime.fromisoformat(val)
    except Exception:
        # default far in past to allow sending if DB malformed
        return dt.datetime.min


def _update_last_sent() -> None:
    """Update the last-sent timestamp in the watchdog table to the current local time."""
    with connect(DB_NAME) as conn:
        initialize(conn)
        cur = conn.cursor()
        cur.execute(
            f"UPDATE {TABLE} SET {COL_LAST_SENT} = datetime('now','localtime') WHERE id = 1"
        )
        conn.commit()


def _update_last_sent_for(db_path: str) -> None:
    """Persist cooldown state for an explicitly selected database."""
    with connect(db_path) as conn:
        initialize(conn)
        conn.execute(
            f"UPDATE {TABLE} SET {COL_LAST_SENT} = datetime('now','localtime') WHERE id = 1"
        )
        conn.commit()


def _get_last_sent_for(db_path: str) -> dt.datetime:
    with connect(db_path) as conn:
        initialize(conn)
        row = conn.execute(f"SELECT {COL_LAST_SENT} FROM {TABLE} WHERE id = 1").fetchone()
    if row is None or row[0] is None:
        return dt.datetime.min
    try:
        return row[0] if isinstance(row[0], dt.datetime) else dt.datetime.fromisoformat(row[0])
    except (TypeError, ValueError):
        return dt.datetime.min


def _cooldown_window_open(
    now_local: dt.datetime,
    delay_sec: int = ALERT_DELAY_SEC,
    db_path: Optional[str] = None,
) -> bool:
    last = _get_last_sent() if db_path is None else _get_last_sent_for(db_path)
    return (now_local - last) >= dt.timedelta(seconds=delay_sec)


# ---------- Message generation ----------
def _fmt(v) -> str:
    """Format numeric values as ints if they look integral, else 1 decimal."""
    try:
        if abs(v - int(v)) < 1e-9:
            return str(int(v))
        return f"{v:.1f}"
    except Exception:
        return str(v)


def _generate_message(metrics: Tuple[dict, dict]) -> str:
    """
    metrics[0]: current values dict
    metrics[1]: rate/min dict
    Keys include: 'Flue (F)', 'STT (F)', 'Room (F)', 'Humidity (%)'
    """
    current, rate = metrics

    # Declarative rules: (dict, key, comparator, threshold, label)
    rules = [
        (current, K_FLUE, ">=", 425, "Flue High Temperature"),
        (rate,    K_FLUE, ">=", 25,  "Flue High Rate of Temperature Increase"),
        (current, K_STT,  ">=", 675, "Stove High Temperature"),
        (rate,    K_STT,  ">=", 25,  "Stove High Rate of Temperature Increase"),
        (current, K_ROOM, "<=", 65,  "Room Temperature Below Limit"),
        (current, K_ROOM, ">=", 85,  "Room Temperature Above Limit"),
        (current, K_HUM,  ">=", 65,  "Room High Humidity"),
        (current, K_HUM,  "<=", 20,  "Room Low Humidity"),
    ]

    lines: List[str] = []
    for src, key, op, thresh, label in rules:
        val = src.get(key, None)
        if val is None:
            continue
        if (
            (op == ">=" and val >= thresh) or
            (op == "<=" and val <= thresh) or
            (op == ">"  and val >  thresh) or
            (op == "<"  and val <  thresh)
        ):
            lines.append(f"{label}: {_fmt(val)}")

    return "\n".join(lines)


# ---------- Public API ----------
def watchdog(db_path: Optional[str] = None) -> None:
    """
    Decide whether to send a consolidated alert based on thresholds and cooldown.
    Optimization: only compute metrics if cooldown window is open.
    """
    now_local = dt.datetime.now()  # matches SQLite localtime()

    try:
        if not _cooldown_window_open(now_local, ALERT_DELAY_SEC, db_path):
            return  # too soon; skip work entirely
        path = db_path or DB_NAME
        metrics = get_metrics(path)  # only called if we *might* send
    except Exception:
        logger.exception("unable to evaluate threshold alert")
        return
    message = _generate_message((metrics[0], metrics[1]))

    if message:
        if send_message(message):
            try:
                if db_path:
                    _update_last_sent_for(db_path)
                else:
                    _update_last_sent()
            except Exception:
                logger.exception("unable to persist alert cooldown")


def sensor_health_check(bad_sensors: Set[str]) -> None:
    """
    Update per-sensor failure counters and route alerts via Telegram.
    bad_sensors: a set of keys among {'tempF','humid','flueF','sttF'}
    """
    global failure_counts, last_alert_time, in_failure_state

    now = _monotonic()
    messages: List[str] = []

    for sensor in failure_counts:
        if sensor in bad_sensors:
            failure_counts[sensor] += 1

            crossed = (failure_counts[sensor] == FAIL_THRESHOLD)
            cooled  = (now - last_alert_time[sensor] >= ALERT_COOLDOWN_SEC)

            if crossed and cooled:
                messages.append(f"[ALERT] {sensor} failed {FAIL_THRESHOLD} consecutive reads.")
                last_alert_time[sensor] = now
                in_failure_state[sensor] = True
        else:
            # Good read this tick
            if REPORT_RECOVERY and in_failure_state[sensor] and failure_counts[sensor] >= FAIL_THRESHOLD:
                messages.append(f"[RECOVERY] {sensor} is reading again after {failure_counts[sensor]} consecutive failures.")
            failure_counts[sensor] = 0
            in_failure_state[sensor] = False

    if messages:
        try:
            send_message("\n".join(messages))
        except Exception:
            logger.exception("unable to send sensor health alert")
