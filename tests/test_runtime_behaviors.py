import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import watchdog
from config import Config
from database import connect, initialize
from metrics import get_metrics
from rpiacqscript import run


class RuntimeBehaviorTests(unittest.TestCase):
    def test_simulated_collector_can_run_a_finite_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.db"
            config = Config(path, sample_interval=0.1, sensor_mode="simulated")

            self.assertEqual(run(config, samples=1), 0)
            with connect(path) as conn:
                row = conn.execute("SELECT tempF, humid, flueF, sttF FROM stove_room").fetchone()
            self.assertEqual(row, (68.0, 45.0, 212.0, 176.0))

    def test_watchdog_does_not_require_telegram_delivery(self):
        with patch.object(watchdog, "send_message", return_value=False) as send:
            watchdog.failure_counts = {key: watchdog.FAIL_THRESHOLD - 1 for key in watchdog.failure_counts}
            now = watchdog._monotonic()
            watchdog.last_alert_time = {
                key: now - watchdog.ALERT_COOLDOWN_SEC - 1
                for key in watchdog.last_alert_time
            }
            watchdog.in_failure_state = {key: False for key in watchdog.in_failure_state}
            watchdog.sensor_health_check(set(watchdog.failure_counts))
            send.assert_called_once()

    def test_metrics_preserve_null_values_for_dashboard_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.db"
            with connect(path) as conn:
                initialize(conn)
            rows = get_metrics(path)

        self.assertEqual(rows[0]["Room (F)"], None)
        self.assertEqual(rows[0]["Humidity (%)"], None)
        self.assertEqual(rows[1]["Flue (F)"], None)


if __name__ == "__main__":
    unittest.main()
