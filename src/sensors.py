"""Sensor readers with lazy Raspberry Pi imports and a deterministic simulator."""

from dataclasses import dataclass
import math
import os
import time
from typing import Dict, Optional, Union


FIELDS = ("tempC", "humid", "flueC", "sttC")

_moment = time.monotonic  # localize for tiny perf win + test injection

_MAX31856_CR0 = 0x00
_MAX31856_CJHF = 0x03
_MAX31856_STT_DEFAULTS = {
    _MAX31856_CJHF: 0x7F,
    0x04: 0xC0,
    0x05: 0x7F,
    0x06: 0xFF,
    0x07: 0x80,
    0x08: 0x00,
}
_MAX31856_SR = 0x0F
_MAX31856_FAULTCLR = 0x02


def _valid(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


@dataclass
class SimulatedReader:
    """Repeatable values, useful for CI and commissioning without Pi packages."""
    sample: int = 0

    def read(self) -> Dict[str, float]:
        n = self.sample
        self.sample += 1
        return {"tempC": 20.0 + (n % 10) * 0.1, "humid": 45.0 + n % 5,
                "flueC": 100.0 + n * 0.5, "sttC": 80.0 + n * 0.25}

    def diagnose(self) -> dict[str, str]:
        return {field: "simulated" for field in FIELDS}


class HardwareReader:
    def __init__(self) -> None:
        # Keep these imports here: simulated mode must not require Pi packages.
        import board
        import digitalio
        import adafruit_dht
        import adafruit_max31856

        spi = board.SPI()
        self.dht = adafruit_dht.DHT11(board.D18)
        flue_cs = digitalio.DigitalInOut(board.D5)
        stt_cs = digitalio.DigitalInOut(board.D6)
        flue_cs.direction = digitalio.Direction.OUTPUT
        stt_cs.direction = digitalio.Direction.OUTPUT
        self.flue = adafruit_max31856.MAX31856(spi, flue_cs, adafruit_max31856.ThermocoupleType.K)
        self.stt = adafruit_max31856.MAX31856(spi, stt_cs, adafruit_max31856.ThermocoupleType.K)
        self._repair_stt_thresholds()

        # The DHT11 bit-bangs its GPIO; after a failed read it can wedge and
        # block for seconds. Back off re-probing it for a while so a flaky
        # sensor does not stall the tick — flue/stt readings keep flowing.
        self._dht_available_after = 0.0
        try:
            self._dht_cooldown_s = max(0.0, float(os.getenv("WSM_DHT_COOLDOWN_S", "30")))
        except (TypeError, ValueError):
            self._dht_cooldown_s = 30.0

    def _repair_stt_thresholds(self) -> None:
        """Restore the defective STT module's power-on threshold defaults."""
        for address, value in _MAX31856_STT_DEFAULTS.items():
            self.stt._write_u8(address, value)

        cr0 = self.stt._read_register(_MAX31856_CR0, 1)[0]
        self.stt._write_u8(_MAX31856_CR0, cr0 | _MAX31856_FAULTCLR)

        actual = {
            address: self.stt._read_register(address, 1)[0]
            for address in _MAX31856_STT_DEFAULTS
        }
        status = self.stt._read_register(_MAX31856_SR, 1)[0]
        if actual != _MAX31856_STT_DEFAULTS or status:
            raise RuntimeError(
                "STT MAX31856 startup repair failed: "
                f"thresholds={actual!r}, status=0x{status:02X}"
            )

    def read(self) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {}
        now = _moment()
        if now < self._dht_available_after:
            result.update(humid=None, tempC=None)
        else:
            try:
                result["humid"], result["tempC"] = self.dht.humidity, self.dht.temperature
            except Exception:
                result.update(humid=None, tempC=None)
            if result.get("tempC") is None or result.get("humid") is None:
                self._dht_available_after = now + self._dht_cooldown_s
            else:
                self._dht_available_after = 0.0
        for field, sensor in (("flueC", self.flue), ("sttC", self.stt)):
            try:
                result[field] = sensor.temperature
            except Exception:
                result[field] = None
        return result

    def diagnose(self) -> dict[str, str]:
        values = self.read()
        return {field: "ok" if _valid(values.get(field)) else "failed" for field in FIELDS}


def create_reader(mode: str = "auto") -> Union[SimulatedReader, HardwareReader]:
    mode = mode.lower()
    if mode == "simulated":
        return SimulatedReader()
    if mode in ("auto", "hardware"):
        return HardwareReader()
    raise ValueError(f"unsupported sensor mode: {mode}")
