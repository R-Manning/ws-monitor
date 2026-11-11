import tempfile
import unittest
from collections import deque
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

    def test_sensor_health_alert_retries_until_delivery_succeeds(self):
        with patch.object(watchdog, "send_message") as send:
            send.return_value = False
            watchdog.failure_counts = {key: watchdog.FAIL_THRESHOLD - 1 for key in watchdog.failure_counts}
            now = watchdog._monotonic()
            watchdog.last_alert_time = {
                key: now - watchdog.ALERT_COOLDOWN_SEC - 1
                for key in watchdog.last_alert_time
            }
            watchdog.in_failure_state = {key: False for key in watchdog.in_failure_state}

            watchdog.sensor_health_check({"tempF"})

            # Delivery rejected: do NOT enter failure state or start cooldown.
            self.assertFalse(watchdog.in_failure_state["tempF"])
            self.assertEqual(watchdog.failure_counts["tempF"], watchdog.FAIL_THRESHOLD)
            self.assertEqual(send.call_count, 1)

            # Next tick retries (threshold still crossed, cooldown not started).
            send.return_value = True
            watchdog.sensor_health_check({"tempF"})
            self.assertEqual(send.call_count, 2)
            self.assertTrue(watchdog.in_failure_state["tempF"])

    def test_metrics_preserve_null_values_for_dashboard_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.db"
            with connect(path) as conn:
                initialize(conn)
            rows = get_metrics(path)

        self.assertEqual(rows[0]["Room (F)"], None)
        self.assertEqual(rows[0]["Humidity (%)"], None)
        self.assertEqual(rows[1]["Flue (F)"], None)


class StuckSensorTests(unittest.TestCase):
    """State-machine tests for watchdog.stuck_sensor_check (driven clock)."""

    def setUp(self):
        self.clock = [0.0]
        self.clock_patcher = patch.object(watchdog, "_monotonic", side_effect=lambda: self.clock[0])
        self.clock_patcher.start()
        self.addCleanup(self.clock_patcher.stop)
        self.window_patcher = patch.object(watchdog, "STUCK_WINDOW_S", 10.0)
        self.delta_patcher = patch.object(watchdog, "STUCK_PEER_DELTA_F", 1.0)
        self.cooldown_patcher = patch.object(watchdog, "ALERT_COOLDOWN_SEC", 60.0)
        for patcher in (self.window_patcher, self.delta_patcher, self.cooldown_patcher):
            patcher.start()
            self.addCleanup(patcher.stop)

        self._saved = (
            watchdog._stuck_windows,
            watchdog._stuck_in_alert,
            watchdog._stuck_last_alert_time,
            watchdog._stuck_persisted,
            watchdog._stuck_value,
        )
        watchdog._stuck_windows = {k: deque() for k in watchdog.STUCK_SENSORS}
        watchdog._stuck_in_alert = {k: False for k in watchdog.STUCK_SENSORS}
        watchdog._stuck_last_alert_time = {
            k: self.clock[0] - watchdog.ALERT_COOLDOWN_SEC - 1 for k in watchdog.STUCK_SENSORS
        }
        watchdog._stuck_persisted = {k: False for k in watchdog.STUCK_SENSORS}
        watchdog._stuck_value = {k: None for k in watchdog.STUCK_SENSORS}
        self.addCleanup(self._restore_state)

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._db = str(Path(self._tmpdir.name) / "stuck.db")

    def _restore_state(self):
        (
            watchdog._stuck_windows,
            watchdog._stuck_in_alert,
            watchdog._stuck_last_alert_time,
            watchdog._stuck_persisted,
            watchdog._stuck_value,
        ) = self._saved

    def _tick(self, values, seconds=1.0):
        self.clock[0] += seconds
        watchdog.stuck_sensor_check(values, self._db)

    def _stuck_scenario(self, steps=12, stt=75.0):
        for i in range(steps):
            self._tick({
                "tempF": 75.0 + 0.2 * i,
                "humid": 45.0 + 0.1 * i,
                "flueF": 74.0 + 0.1 * i,
                "sttF": stt,
            })

    def test_flat_cold_stove_never_triggers(self):
        with patch.object(watchdog, "send_message", return_value=True) as send:
            base = {"tempF": 75.4, "humid": 45.0, "flueF": 73.3, "sttF": 75.4}
            for _ in range(15):
                self._tick(base)
        send.assert_not_called()
        for sensor in watchdog.STUCK_SENSORS:
            self.assertFalse(watchdog._stuck_in_alert[sensor])

    def test_stuck_with_peer_movement_alerts_persists_and_recovers(self):
        with patch.object(watchdog, "send_message", return_value=True) as send:
            self._stuck_scenario()
            send.assert_called_once()
            self.assertIn("[ALERT]", send.call_args.args[0])
            self.assertIn("sttF", send.call_args.args[0])
            self.assertTrue(watchdog._stuck_in_alert["sttF"])
        with connect(self._db) as conn:
            self.assertEqual(
                conn.execute("SELECT stuck FROM sensor_status WHERE sensor='sttF'").fetchone()[0], 1
            )

        with patch.object(watchdog, "send_message", return_value=True) as send:
            for _ in range(3):
                self._tick({"tempF": 77.4, "humid": 45.0, "flueF": 74.0, "sttF": 75.5})
            self.assertFalse(watchdog._stuck_in_alert["sttF"])
            self.assertIn("[RECOVERY]", send.call_args.args[0])
        with connect(self._db) as conn:
            self.assertEqual(
                conn.execute("SELECT stuck FROM sensor_status WHERE sensor='sttF'").fetchone()[0], 0
            )

    def test_stuck_alert_is_delivery_gated(self):
        with patch.object(watchdog, "send_message") as send:
            send.return_value = False
            self._stuck_scenario()
            self.assertEqual(send.call_count, 1)
            self.assertFalse(watchdog._stuck_in_alert["sttF"])

            # Gap past the cooldown, then delivery starts succeeding.
            self._tick({"tempF": 78.0, "humid": 45.0, "flueF": 74.0, "sttF": 75.0}, seconds=61.0)
            send.return_value = True
            for i in range(1, 13):
                self._tick({
                    "tempF": 78.0 + 0.2 * i,
                    "humid": 45.0 + 0.1 * i,
                    "flueF": 74.0 + 0.1 * i,
                    "sttF": 75.0,
                })
            self.assertTrue(watchdog._stuck_in_alert["sttF"])
            self.assertGreaterEqual(send.call_count, 2)

    def test_none_reads_never_count_as_value_change(self):
        with patch.object(watchdog, "send_message", return_value=True) as send:
            for i in range(13):
                stt = 75.0 if i % 2 == 0 else None
                self._tick({"tempF": 75.0 + 0.2 * i, "humid": 45.0, "flueF": 74.0, "sttF": stt})
        send.assert_called_once()
        self.assertIn("[ALERT]", send.call_args.args[0])
        self.assertTrue(watchdog._stuck_in_alert["sttF"])


if __name__ == "__main__":
    unittest.main()
