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
from sensors import HardwareReader, SimulatedReader, create_reader


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

    @staticmethod
    def _hardware_modules(ignored_address=None, clear_faults=True):
        class Pin:
            def __init__(self, name):
                self.name = name

        class DigitalInOut:
            def __init__(self, pin):
                self.pin = pin
                self.direction = None

        class DHT11:
            def __init__(self, pin):
                self.pin = pin

        class MAX31856:
            instances = []

            def __init__(self, spi, cs, thermocouple_type):
                self.cs = cs
                self.writes = []
                self.registers = {0x00: 0x10, 0x0F: 0x28}
                self.ignored_address = ignored_address if cs.pin.name == "D6" else None
                self.clear_faults = clear_faults
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
