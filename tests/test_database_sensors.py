import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import connect, initialize, insert_sample, settings, sync_runtime_settings
from metrics import get_metrics
from rpiacqscript import convert_reading
from sensors import HardwareReader, SimulatedReader, _plausible, create_reader


class _FakeClock:
    """Mutable monotonic clock for driving cooldown timers in tests."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t


class DatabaseTests(unittest.TestCase):
    def test_sample_keeps_missing_values_as_null(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = connect(Path(directory) / "samples.db")
            try:
                initialize(conn)
                insert_sample(conn, {"tempF": None, "humid": 50, "flueF": 212, "sttF": None}, "2026-01-01 00:00:00")
                row = conn.execute("SELECT tempF, humid, flueF, sttF FROM stove_room").fetchone()
                self.assertEqual(row, (None, 50.0, 212.0, None))
            finally:
                conn.close()

    def test_metrics_keep_nulls_and_calculate_rates_without_pandas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.db"
            with connect(path) as conn:
                initialize(conn)
                insert_sample(conn, {"tempF": 70, "humid": None, "flueF": 200, "sttF": None}, "2026-01-01 00:00:00")
                insert_sample(conn, {"tempF": 71, "humid": None, "flueF": 210, "sttF": None}, "2026-01-01 00:01:00")

            metrics = get_metrics(path)
            self.assertEqual(metrics[0]["Room (F)"], 71.0)
            self.assertEqual(metrics[0]["Humidity (%)"], None)
            self.assertEqual(metrics[1]["Flue (F)"], 10.0)
            self.assertEqual(metrics[1]["Humidity (%)"], None)

    def test_initialize_migrates_legacy_table_without_ok_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            with sqlite3.connect(path) as conn:
                conn.executescript("""
                    CREATE TABLE stove_room (
                        datetime TEXT PRIMARY KEY,
                        tempF FLOAT, humid FLOAT, flueF FLOAT, sttF FLOAT
                    );
                    INSERT INTO stove_room VALUES ('2026-01-01 00:00:00', 70, NULL, 212, 80);
                """)
            with connect(path) as conn:
                initialize(conn)
                columns = [row[1] for row in conn.execute("PRAGMA table_info(stove_room)").fetchall()]
                for col in ("tempF_ok", "humid_ok", "flueF_ok", "sttF_ok"):
                    self.assertIn(col, columns)
                row = conn.execute(
                    "SELECT tempF_ok, humid_ok, flueF_ok, sttF_ok FROM stove_room"
                ).fetchone()
            self.assertEqual(row, (1, 0, 1, 1))

    def test_initialize_creates_sensor_status_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.db"
            with connect(path) as conn:
                initialize(conn)
                tables = [row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
        self.assertIn("sensor_status", tables)

    def test_context_manager_closes_connection_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cm.db"
            with connect(path) as conn:
                conn.execute("SELECT 1")
            with self.assertRaises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")

    def test_sync_runtime_settings_aligns_table_to_declared_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sync.db"
            with connect(path) as conn:
                initialize(conn)
                conn.execute("UPDATE settings SET sampleFreq=99, dataHist=42")
                conn.commit()
            with connect(path) as conn:
                self.assertEqual(sync_runtime_settings(conn), (5.0, 7))
                self.assertEqual(settings(conn), (5.0, 7))

    def test_get_metrics_without_schema_init_requires_existing_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(sqlite3.OperationalError):
                get_metrics(Path(directory) / "no-schema.db", ensure_schema=False)

    def test_get_metrics_without_schema_init_works_on_initialized_db(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.db"
            with connect(path) as conn:
                initialize(conn)
                insert_sample(conn, {"tempF": 70, "humid": 45, "flueF": 200, "sttF": 80}, "2026-01-01 00:00:00")
            metrics = get_metrics(path, ensure_schema=False)
        self.assertEqual(metrics[0]["Room (F)"], 70.0)


class SensorTests(unittest.TestCase):
    def test_simulated_reader_requires_no_hardware_packages(self):
        reader = create_reader("simulated")
        self.assertIsInstance(reader, SimulatedReader)
        self.assertEqual(reader.read(), {"tempC": 20.0, "humid": 45.0, "flueC": 100.0, "sttC": 80.0})

    def test_plausible_rejects_implausible_reads(self):
        self.assertTrue(_plausible(20.0, 1100.0))          # 68 F, fine
        self.assertTrue(_plausible(300.0, 1100.0))         # 572 F, fine below clamp
        self.assertFalse(_plausible(560.0, 1100.0))        # ~1040 F, would clamp to 1000
        self.assertFalse(_plausible(610.0, 1100.0))        # ~1130 F, above stt ceiling
        self.assertFalse(_plausible(500.0, 850.0))         # 932 F, above flue ceiling
        self.assertTrue(_plausible(450.0, 850.0))          # 842 F, fine for flue
        self.assertFalse(_plausible(None, 1100.0))
        self.assertFalse(_plausible(float("nan"), 1100.0))
        self.assertFalse(_plausible(float("inf"), 1100.0))

    def test_stt_saturation_is_rejected_and_fault_rearmed(self):
        modules, max31856 = self._hardware_modules()
        with patch.dict(sys.modules, modules):
            reader = HardwareReader()
            stt = reader.stt
            flue = reader.flue
            stt.temperature = 560.0  # ~1040 F: would clamp to exactly 1000 in the collector
            out = reader.read()

        self.assertIsNone(out["sttC"], "saturated stt read must be rejected")
        self.assertEqual(out["flueC"], 300.0)
        self.assertEqual(stt.writes[-1], (0x00, 0x12), "fault must be re-armed after reject")
        self.assertEqual(flue.writes, [])

    def test_stt_read_above_ceiling_is_rejected(self):
        modules, _ = self._hardware_modules()
        with patch.dict(sys.modules, modules):
            reader = HardwareReader()
            reader.stt.temperature = 610.0  # ~1130 F > 1100 ceiling
            out = reader.read()
        self.assertIsNone(out["sttC"])
        self.assertEqual(out["flueC"], 300.0)

    def test_flue_read_above_ceiling_is_rejected_without_stt_rearm(self):
        modules, max31856 = self._hardware_modules()
        with patch.dict(sys.modules, modules):
            reader = HardwareReader()
            flue, stt = max31856.MAX31856.instances
            flue.temperature = 500.0  # 932 F > 850 ceiling (still < 1000 clamp)
            out = reader.read()
        self.assertIsNone(out["flueC"], "over-ceiling flue read must be rejected")
        self.assertEqual(out["sttC"], 300.0)
        self.assertEqual(stt.writes[-1], (0x00, 0x12), "only the rejected channel re-arms")

    def test_conversion_handles_partial_reading(self):
        self.assertEqual(
            convert_reading({"tempC": None, "humid": 50, "flueC": 100, "sttC": None}),
            {"tempF": None, "humid": 50.0, "flueF": 212.0, "sttF": None},
        )

    def test_hardware_reader_repairs_only_stt_thresholds_and_clears_faults(self):
        modules, max31856 = self._hardware_modules()
        with patch.dict(sys.modules, modules):
            HardwareReader()

        flue, stt = max31856.MAX31856.instances
        self.assertEqual(flue.writes, [])
        self.assertEqual(
            stt.writes,
            [(0x03, 0x7F), (0x04, 0xC0), (0x05, 0x7F), (0x06, 0xFF),
             (0x07, 0x80), (0x08, 0x00), (0x00, 0x12)],
        )
        self.assertEqual(stt.registers[0x0F], 0)

    def test_hardware_reader_fails_when_stt_threshold_repair_cannot_be_verified(self):
        modules, _ = self._hardware_modules(ignored_address=0x03)
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "startup repair failed"):
                HardwareReader()

    def test_hardware_reader_fails_when_stt_fault_latch_does_not_clear(self):
        modules, _ = self._hardware_modules(clear_faults=False)
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "status=0x28"):
                HardwareReader()

    def test_auto_mode_propagates_hardware_initialization_errors(self):
        with patch("sensors.HardwareReader", side_effect=RuntimeError("SPI unavailable")):
            with self.assertRaisesRegex(RuntimeError, "SPI unavailable"):
                create_reader("auto")

    def test_dht_cooldown_skips_sensor_until_window_elapses(self):
        fake = _FakeClock()
        modules, _ = self._hardware_modules(dht_mode="raise")
        with patch.dict(sys.modules, modules):
            with patch("sensors._moment", fake):
                reader = HardwareReader()

                out = reader.read()
                self.assertIsNone(out["humid"])
                self.assertIsNone(out["tempC"])

                dht = modules["adafruit_dht"].DHT11.instances[-1]
                acc = list(dht.accesses)

                out2 = reader.read()
                self.assertIsNone(out2["humid"])
                self.assertEqual(
                    list(dht.accesses), acc, "DHT must not be probed during cooldown"
                )

                fake.t += 40  # past the 30s window
                out3 = reader.read()
                self.assertIsNone(out3["humid"])
                self.assertGreater(
                    list(dht.accesses), acc, "DHT is probed again after cooldown"
                )

    def test_dht_success_resets_cooldown(self):
        fake = _FakeClock()
        modules, _ = self._hardware_modules()
        with patch.dict(sys.modules, modules):
            with patch("sensors._moment", fake):
                reader = HardwareReader()
                dht = modules["adafruit_dht"].DHT11.instances[-1]

                reader.dht.mode = "raise"
                reader.read()
                self.assertIsNone(reader.read()["humid"])  # inside cooldown

                fake.t += 40
                dht.mode = "valid"
                out = reader.read()  # probes again, succeeds, clears cooldown
                self.assertEqual(out["humid"], 45.0)

                acc = list(dht.accesses)
                out2 = reader.read()  # immediate re-read must probe again
                self.assertEqual(out2["humid"], 45.0)
                self.assertGreater(
                    list(dht.accesses), acc, "successful read must clear cooldown"
                )

    @staticmethod
    def _hardware_modules(ignored_address=None, clear_faults=True, dht_mode="valid"):
        class Pin:
            def __init__(self, name):
                self.name = name

        class DigitalInOut:
            def __init__(self, pin):
                self.pin = pin
                self.direction = None

        class DHT11:
            instances = []

            def __init__(self, pin):
                self.pin = pin
                self.mode = dht_mode  # "valid" | "raise"
                self.accesses = []
                type(self).instances.append(self)

            @property
            def humidity(self):
                self.accesses.append("humidity")
                if self.mode == "raise":
                    raise RuntimeError("DHT read failed")
                return 45.0

            @property
            def temperature(self):
                self.accesses.append("temperature")
                if self.mode == "raise":
                    raise RuntimeError("DHT read failed")
                return 20.0

        class MAX31856:
            instances = []

            def __init__(self, spi, cs, thermocouple_type):
                self.cs = cs
                self.writes = []
                self.registers = {0x00: 0x10, 0x0F: 0x28}
                self.ignored_address = ignored_address if cs.pin.name == "D6" else None
                self.clear_faults = clear_faults
                self.temperature = 300.0  # defaults: ~572 F, plausible for both channels
                type(self).instances.append(self)

            def _write_u8(self, address, value):
                self.writes.append((address, value))
                if address != self.ignored_address:
                    self.registers[address] = value
                if address == 0x00 and value & 0x02 and self.clear_faults:
                    self.registers[0x0F] = 0

            def _read_register(self, address, length):
                return bytearray([self.registers.get(address, 0)])

        board = types.SimpleNamespace(SPI=lambda: object(), D5=Pin("D5"), D6=Pin("D6"), D18=Pin("D18"))
        digitalio = types.SimpleNamespace(DigitalInOut=DigitalInOut, Direction=types.SimpleNamespace(OUTPUT="output"))
        max31856 = types.SimpleNamespace(MAX31856=MAX31856, ThermocoupleType=types.SimpleNamespace(K="K"))
        return {
            "board": board,
            "digitalio": digitalio,
            "adafruit_dht": types.SimpleNamespace(DHT11=DHT11),
            "adafruit_max31856": max31856,
        }, max31856


if __name__ == "__main__":
    unittest.main()
