"""Standalone, fault-tolerant stove-room collector."""

import argparse
import signal
import time
from typing import Any, Dict, List, Optional

from config import load_config
from database import connect, delete_older_than, diagnose as diagnose_database, initialize, insert_sample, sync_runtime_settings
from logging_setup import configure_logging
from sensors import create_reader

try:
    from watchdog import sensor_health_check, stuck_sensor_check, watchdog
except ImportError:  # Alerting is optional and must not disable collection.
    sensor_health_check = None
    stuck_sensor_check = None
    watchdog = None


def _c_to_f(value: Any, low: float, high: float) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return round(min(high, max(low, value * 1.8 + 32)), 1)


def convert_reading(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "tempF": _c_to_f(raw.get("tempC"), 10, 125),
        "humid": round(min(100, max(5, float(raw["humid"]))), 1) if isinstance(raw.get("humid"), (int, float)) else None,
        "flueF": _c_to_f(raw.get("flueC"), 0, 1000),
        "sttF": _c_to_f(raw.get("sttC"), 0, 1000),
    }


def read_with_retries(reader: Any, retries: int, logger: Any) -> Dict[str, Optional[float]]:
    last: Dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            last = reader.read()
            return convert_reading(last)
        except Exception as exc:
            logger.warning("sensor read failed (attempt %d/%d): %s", attempt + 1, retries + 1, exc)
            time.sleep(min(0.25 * (2 ** attempt), 2.0))
    return convert_reading(last)


def run(config: Any, samples: Optional[int] = None, once: bool = False, diagnose: bool = False) -> int:
    logger = configure_logging(config.log_level)
    reader = create_reader(config.sensor_mode)
    conn = connect(config.db_path)
    try:
        try:
            initialize(conn)
        except Exception:
            logger.exception("schema initialization failed")
        if diagnose:
            print({"db_path": str(config.db_path), **diagnose_database(conn),
                   "sensor_mode": type(reader).__name__, "sensors": reader.diagnose()})
            return 0
        interval, history = sync_runtime_settings(conn)
        delete_older_than(conn, history)
        count = 0
        stopped = False

        def stop(_signum: int, _frame: Any) -> None:
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        last_retention_cleanup = time.monotonic()
        next_tick = time.monotonic()
        while not stopped and (samples is None or count < samples) and not (once and count):
            values = read_with_retries(reader, config.retries, logger)
            insert_sample(conn, values)
            if sensor_health_check is not None:
                bad = {name for name, value in values.items() if value is None}
                try:
                    sensor_health_check(bad)
                except Exception:
                    logger.exception("sensor health alert failed")
            if stuck_sensor_check is not None:
                try:
                    stuck_sensor_check(values, str(config.db_path))
                except Exception:
                    logger.exception("stuck-sensor check failed")
            if watchdog is not None:
                try:
                    watchdog(str(config.db_path))
                except Exception:
                    logger.exception("threshold alert failed")
            logger.info("sample %d: %s", count + 1, values)
            count += 1
            if time.monotonic() - last_retention_cleanup >= 86400.0:
                delete_older_than(conn, history)
                last_retention_cleanup = time.monotonic()
            next_tick += max(2.0, interval)
            time.sleep(max(0.0, next_tick - time.monotonic()))
        return 0
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, help="number of samples, then exit")
    parser.add_argument("--once", action="store_true", help="collect one sample, then exit")
    parser.add_argument("--diagnose", action="store_true", help="read sensors and print their status")
    parser.add_argument("--sensor-mode", choices=("auto", "hardware", "simulated"), default=None)
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args(argv)
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be positive")
    return run(load_config(args.db_path, args.sensor_mode), args.samples, args.once, args.diagnose)


if __name__ == "__main__":
    raise SystemExit(main())
