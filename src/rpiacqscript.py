# RPi4 Stove Room Environment (consolidated & optimized)

import math
import time
import sqlite3
from collections import deque
from contextlib import closing
from typing import Tuple
from bisect import bisect_left, insort
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
if not env_path.exists():
    raise RuntimeError(f".env file not found at {env_path}")
load_dotenv(env_path)

import board
import digitalio
import adafruit_dht
import adafruit_max31856

from watchdog import watchdog, sensor_health_check
from paths import get_db_path


# ──────────────────────────────────────────────────────────────────────────────
# Config / Globals
# ──────────────────────────────────────────────────────────────────────────────
dbname = str(get_db_path())
dataTable = "stove_room"
settingsTable = "settings"

# median window size & sensor keys (raw °C inputs for filtering)
window_size = 5
keys = ["tempC", "humid", "flueC", "sttC"]

# Populated from settings at startup (seconds / days)
sampleFreq: float = 5.0
dbHistory: int = 7


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────
def _connect() -> sqlite3.Connection:
    """Open SQLite with WAL + NORMAL sync for lower contention & fsync cost."""
    conn = sqlite3.connect(dbname, detect_types=sqlite3.PARSE_DECLTYPES, timeout=5)
    with closing(conn.cursor()) as cur:
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _load_settings() -> Tuple[int, int]:
    """Read (sampleFreq seconds, dbHistory days) from settings; provide safe defaults."""
    with closing(_connect()) as conn, closing(conn.cursor()) as cur:
        cur.execute(f'SELECT * FROM "{settingsTable}" LIMIT 1;')
        row = cur.fetchone()
    if not row:
        return 5, 7
    # row[0] = sample period (s), row[1] = db history (days)
    s = int(row[0])
    d = int(row[1]) if len(row) >= 2 else 7
    return s, d


def _startup_retention_cleanup(history_days: int) -> None:
    """Delete old rows once at startup using parameter binding (no string concat)."""
    with closing(_connect()) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            f'''DELETE FROM "{dataTable}"
                WHERE datetime <= date('now', printf('-%d day', ?));''',
            (int(history_days),),
        )
        conn.commit()


def _ensure_time_index() -> None:
    """Create an index on datetime if it doesn't exist (idempotent)."""
    with closing(_connect()) as conn, closing(conn.cursor()) as cur:
        cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{dataTable}_datetime ON "{dataTable}"(datetime);')
        conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def _is_bad(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


class RunningMedian:
    """Fixed-size running median without resorting the entire window each sample."""
    def __init__(self, size: int):
        self.size = int(size)
        self.buffers = {k: deque(maxlen=size) for k in keys}
        self.sorted = {k: [] for k in keys}

    def add(self, sensor: str, value):
        # Ignore invalids: don't change the window; return current median (or None)
        if _is_bad(value):
            arr = self.sorted[sensor]
            n = len(arr)
            if n == 0:
                return None
            mid = n // 2
            return (arr[mid - 1] + arr[mid]) / 2 if n % 2 == 0 else arr[mid]

        dq = self.buffers[sensor]
        arr = self.sorted[sensor]

        # If about to evict, remove matching value from the sorted list as well
        if len(dq) == dq.maxlen:
            old = dq[0]
            if not _is_bad(old):
                i = bisect_left(arr, old)
                if i < len(arr) and arr[i] == old:
                    arr.pop(i)

        dq.append(value)
        insort(arr, value)

        n = len(arr)
        mid = n // 2
        return (arr[mid - 1] + arr[mid]) / 2 if n % 2 == 0 else arr[mid]


median = RunningMedian(window_size)


# ──────────────────────────────────────────────────────────────────────────────
# Hardware init
# ──────────────────────────────────────────────────────────────────────────────
# SPI bus (thermocouples)
spi = board.SPI()

# DHT11 on GPIO18; respect DHT11 minimum read interval (~2s) at the callsite
DHT_SENSOR = adafruit_dht.DHT11(board.D18)

# Chip selects for MAX31856 thermocouples
flue = digitalio.DigitalInOut(board.D5)
flue.direction = digitalio.Direction.OUTPUT
stt = digitalio.DigitalInOut(board.D6)
stt.direction = digitalio.Direction.OUTPUT

# Thermocouple objects (type K)
flueT = adafruit_max31856.MAX31856(spi, flue, adafruit_max31856.ThermocoupleType.K)
sttT = adafruit_max31856.MAX31856(spi, stt, adafruit_max31856.ThermocoupleType.K)
thermocouples = (flueT, sttT)


# ──────────────────────────────────────────────────────────────────────────────
# Data I/O
# ──────────────────────────────────────────────────────────────────────────────
def logData(data: dict) -> None:
    """Insert a single sample (now, localtime) with parameter binding."""
    with closing(_connect()) as conn:
        conn.execute(
            f'''INSERT OR IGNORE INTO "{dataTable}"
                VALUES (datetime('now','localtime'), ?, ?, ?, ?)''',
            [data["tempF"], data["humid"], data["flueF"], data["sttF"]],
        )
        conn.commit()
    watchdog()


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────
def getData():
    """
    Read sensors, run median on raw °C values, convert to F, clamp to sane ranges.
    Returns (data_dict, bad_set) — if any field is None, returns (None, bad_set).
    """
    bad = set()

    # --- DHT11 (room temp/humidity) ---
    try:
        humidity = DHT_SENSOR.humidity
        temperature = DHT_SENSOR.temperature  # °C
    except RuntimeError:
        humidity, temperature = None, None

    tempC_med = median.add("tempC", temperature)
    humid_med = median.add("humid", humidity)

    tempF = None if tempC_med is None else round(tempC_med * 1.8 + 32, 1)
    humid = None if humid_med is None else round(humid_med)

    if tempF is None:
        bad.add("tempF")
    if humid is None:
        bad.add("humid")

    # --- Thermocouples (flue/stove) ---
    try:
        flueC = thermocouples[0].temperature  # °C
    except Exception:
        flueC = None
    try:
        sttC = thermocouples[1].temperature  # °C
    except Exception:
        sttC = None

    flueC_med = median.add("flueC", flueC)
    sttC_med = median.add("sttC", sttC)

    flueF = None if flueC_med is None else round(flueC_med * 1.8 + 32, 1)
    sttF = None if sttC_med is None else round(sttC_med * 1.8 + 32, 1)

    if flueF is None:
        bad.add("flueF")
    if sttF is None:
        bad.add("sttF")

    # If any invalid -> skip this tick (but report failing fields)
    if bad:
        return None, bad

    # Clamp to sane ranges
    tempF = _clamp(tempF, 10, 125)
    humid = _clamp(humid, 5, 100)
    flueF = _clamp(flueF, 0, 1000)
    sttF = _clamp(sttF, 0, 1000)

    return {"tempF": tempF, "humid": humid, "flueF": flueF, "sttF": sttF}, bad


def dataAcquisition():
    """Monotonic scheduler with daily retention cleanup."""
    deleteElapsed = 0.0

    # Enforce DHT11 minimum interval (~2s); honor setting otherwise
    period = max(float(sampleFreq), 2.0)

    next_tick = time.monotonic()  # schedule first tick immediately
    while True:
        now = time.monotonic()
        if now >= next_tick:
            data, bad = getData()
            if data is not None:
                logData(data)
                sensor_health_check(set())  # mark all OK
                deleteElapsed += period
            else:
                sensor_health_check(bad)

            # schedule next sample; catch up if fell behind
            next_tick += period
            if next_tick < now:
                next_tick = now + period

            # once a day, delete data older than dbHistory days
            if deleteElapsed >= 86400:
                with closing(_connect()) as conn, closing(conn.cursor()) as cur:
                    cur.execute(
                        f'''DELETE FROM "{dataTable}"
                            WHERE datetime <= date('now', printf('-%d day', ?));''',
                        (int(dbHistory),),
                    )
                    conn.commit()
                deleteElapsed = 0.0

        # sleep until next tick without busy-waiting
        time.sleep(max(0.01, next_tick - time.monotonic()))


# ──────────────────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load settings and set globals
    s, d = _load_settings()
    sampleFreq, dbHistory = float(s), int(d)

    # One-time DB maintenance
    _ensure_time_index()
    _startup_retention_cleanup(dbHistory)

    # Run acquisition loop forever
    dataAcquisition()
