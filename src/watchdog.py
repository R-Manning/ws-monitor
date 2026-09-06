import datetime as dt
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Deque

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

logger = logging.getLogger("ws-monitor.watchdog")

# === Config ===
FAIL_THRESHOLD = 12                 # consecutive bad reads before alert
ALERT_COOLDOWN_SEC = 2.5 * 60        # per-sensor cooldown in sensor_health_check
REPORT_RECOVERY = True              # send "[RECOVERY]" when sensor becomes OK

ALERT_DELAY_SEC = int(2.5 * 60)     # main watchdog message min spacing

DB_NAME = str(get_db_path())
TABLE = "watchDog"                  # <- constant/whitelist identifiers
COL_LAST_SENT = "emailtimelastsent"

# Stuck-sensor detection: a channel reporting the exact same value for this
# long while at least one other channel is moving is treated as frozen.
STUCK_WINDOW_S = max(30.0, float(os.getenv("WSM_STUCK_WINDOW_S", "300")))
STUCK_PEER_DELTA_F = max(0.1, float(os.getenv("WSM_STUCK_PEER_DELTA_F", "1.0")))

STUCK_SENSORS = ("tempF", "humid", "flueF", "sttF")

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

# === State for stuck_sensor_check ===
_stuck_windows: Dict[str, Deque[Tuple[float, Optional[float]]]] = {k: deque() for k in STUCK_SENSORS}
_stuck_in_alert: Dict[str, bool] = {k: False for k in STUCK_SENSORS}
_stuck_last_alert_time: Dict[str, float] = {k: 0.0 for k in STUCK_SENSORS}
_stuck_persisted: Dict[str, bool] = {k: False for k in STUCK_SENSORS}
_stuck_value: Dict[str, Optional[float]] = {k: None for k in STUCK_SENSORS}

_monotonic = time.monotonic  # localize for tiny perf win


# ---------- DB helpers ----------
_db_initialized: Set[str] = set()
_ensure_lock = threading.Lock()


def _ensure_database(path: str) -> None:
    """Initialize a database's schema once per path (thread-safe)."""
    key = str(Path(path).resolve())
    if key in _db_initialized:
        return
    with _ensure_lock:
        if key in _db_initialized:
            return
        with connect(key) as conn:
            initialize(conn)
        _db_initialized.add(key)


def _get_last_sent() -> dt.datetime:
    """Returns last send time (localtime) stored in DB."""
    _ensure_database(DB_NAME)
    with connect(DB_NAME) as conn:
        row = conn.execute(f"SELECT {COL_LAST_SENT} FROM {TABLE} LIMIT 1").fetchone()
    if row is None or row[0] is None:
        return dt.datetime.min
    val = row[0]
    if isinstance(val, dt.datetime):
        return val
    try:
        return dt.datetime.fromisoformat(val)
    except Exception:
        # default far in past to allow sending if DB malformed
        return dt.datetime.min


def _update_last_sent() -> None:
    """Update the last-sent timestamp in the watchdog table to the current local time."""
    _ensure_database(DB_NAME)
    with connect(DB_NAME) as conn:
        conn.execute(
            f"UPDATE {TABLE} SET {COL_LAST_SENT} = datetime('now','localtime')"
        )
        conn.commit()


def _update_last_sent_for(db_path: str) -> None:
    """Persist cooldown state for an explicitly selected database."""
    _ensure_database(db_path)
    with connect(db_path) as conn:
        conn.execute(
            f"UPDATE {TABLE} SET {COL_LAST_SENT} = datetime('now','localtime')"
        )
        conn.commit()


def _get_last_sent_for(db_path: str) -> dt.datetime:
    _ensure_database(db_path)
    with connect(db_path) as conn:
        row = conn.execute(f"SELECT {COL_LAST_SENT} FROM {TABLE} LIMIT 1").fetchone()
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
        metrics = get_metrics(path, ensure_schema=False)  # only called if we *might* send
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
    crossed_ids: Set[str] = set()

    for sensor in failure_counts:
        if sensor in bad_sensors:
            failure_counts[sensor] += 1

            crossed = (failure_counts[sensor] >= FAIL_THRESHOLD)
            cooled  = (now - last_alert_time[sensor] >= ALERT_COOLDOWN_SEC)

            if crossed and cooled:
                messages.append(f"[ALERT] {sensor} failed {FAIL_THRESHOLD} consecutive reads.")
                crossed_ids.add(sensor)
        else:
            # Good read this tick
            if REPORT_RECOVERY and in_failure_state[sensor] and failure_counts[sensor] >= FAIL_THRESHOLD:
                messages.append(f"[RECOVERY] {sensor} is reading again after {failure_counts[sensor]} consecutive failures.")
            failure_counts[sensor] = 0
            in_failure_state[sensor] = False

    if messages:
        logger.info("sensor health alert: %s", messages)
        try:
            if send_message("\n".join(messages)):
                # Delivery-gated state: only enter failure state (and start the
                # per-sensor cooldown) after Telegram actually accepted the alert;
                # otherwise retry next tick while the threshold is still crossed.
                for sensor in crossed_ids:
                    last_alert_time[sensor] = now
                    in_failure_state[sensor] = True
        except Exception:
            logger.exception("unable to send sensor health alert")


def _sensor_frozen(sensor: str, now: float) -> bool:
    """True if the channel has held the exact same value for STUCK_WINDOW_S."""
    window = _stuck_windows[sensor]
    if not window or now - window[0][0] < STUCK_WINDOW_S:
        return False
    first = window[0][1]
    return all(value == first for _, value in window)


def _peer_active(sensor: str, now: float) -> bool:
    """True if any other channel moved by >= STUCK_PEER_DELTA_F recently."""
    for peer in STUCK_SENSORS:
        if peer == sensor:
            continue
        window = _stuck_windows[peer]
        values = [value for _, value in window if value is not None]
        if len(values) >= 2 and (max(values) - min(values)) >= STUCK_PEER_DELTA_F:
            return True
    return False


def _set_stuck_status(db_path: Optional[str], sensor: str, stuck: bool) -> None:
    """Persist the freeze flag for the dashboard tile indicator."""
    try:
        path = db_path or DB_NAME
        _ensure_database(path)
        with connect(path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sensor_status(sensor, stuck, changed_at) "
                "VALUES (?, ?, datetime('now','localtime'))",
                (sensor, 1 if stuck else 0),
            )
            conn.commit()
    except Exception:
        logger.exception("unable to persist stuck status for %s", sensor)


def stuck_sensor_check(values: Dict[str, Optional[float]], db_path: Optional[str] = None) -> None:
    """Update per-channel freeze detection and route alerts via Telegram.

    A channel is 'stuck' when it reports the exact same value across the full
    STUCK_WINDOW_S window while at least one other channel moved by >=
    STUCK_PEER_DELTA_F in that window. A flat but healthy cold stove never
    triggers (no peer movement); a channel frozen mid-burn does.
    """
    now = _monotonic()
    for sensor in STUCK_SENSORS:
        window = _stuck_windows[sensor]
        while window and now - window[0][0] > STUCK_WINDOW_S:
            window.popleft()
        value = values.get(sensor)
        if value is not None:
            window.append((now, value))

    messages: List[str] = []
    newly_stuck: Set[str] = set()

    for sensor in STUCK_SENSORS:
        flagged = _stuck_in_alert[sensor]
        frozen_now = _sensor_frozen(sensor, now)
        ref = _stuck_value[sensor]

        if flagged:
            # Exit on a real value change only; sampling gaps (None) never
            # count as movement, so a freeze ends only when the channel moves.
            if any(value != ref for _, value in _stuck_windows[sensor]):
                messages.append(f"[RECOVERY] {sensor} is no longer stuck (value changed)")
                _stuck_in_alert[sensor] = False
                _stuck_value[sensor] = None
        elif frozen_now and _peer_active(sensor, now):
            if now - _stuck_last_alert_time[sensor] >= ALERT_COOLDOWN_SEC:
                messages.append(
                    f"[ALERT] {sensor} appears stuck: identical for {int(STUCK_WINDOW_S)}s "
                    "while another channel is moving"
                )
                # Throttle retries to once per cooldown even when delivery fails.
                _stuck_last_alert_time[sensor] = now
                newly_stuck.add(sensor)

        stuck = _stuck_in_alert[sensor] or (frozen_now and _peer_active(sensor, now))
        if stuck != _stuck_persisted[sensor]:
            _set_stuck_status(db_path, sensor, stuck)
            _stuck_persisted[sensor] = stuck

    if messages:
        logger.info("stuck sensor alert: %s", messages)
        try:
            # Delivery-gated: only latch the alert state after Telegram accepts
            # the message; otherwise retry once the cooldown elapses while the
            # freeze persists.
            if send_message("\n".join(messages)):
                for sensor in newly_stuck:
                    _stuck_value[sensor] = _stuck_windows[sensor][0][1]
                    _stuck_in_alert[sensor] = True
        except Exception:
            logger.exception("unable to send stuck sensor alert")
