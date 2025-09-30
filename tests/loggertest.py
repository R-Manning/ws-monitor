import board
import adafruit_max31856
import digitalio
import time
import adafruit_dht

spi = board.SPI()

# DHT11 recommends >= 2s between reads; you're doing 5s below.
DHT_SENSOR = adafruit_dht.DHT11(board.D18)

# Chip selects (active low): start high (inactive)
flue_cs = digitalio.DigitalInOut(board.D5)
flue_cs.direction = digitalio.Direction.OUTPUT
flue_cs.value = True

stt_cs = digitalio.DigitalInOut(board.D6)
stt_cs.direction = digitalio.Direction.OUTPUT
stt_cs.value = True

flueT = adafruit_max31856.MAX31856(spi, flue_cs, adafruit_max31856.ThermocoupleType.K)
sttT  = adafruit_max31856.MAX31856(spi, stt_cs, adafruit_max31856.ThermocoupleType.K)

def f2(c):  # C→F helper with one-liners below
    return (c * 9/5) + 32 if c is not None else None

t0 = time.monotonic()

while True:
    now = time.monotonic()
    if now - t0 >= 5:
        # DHT11 can throw RuntimeError on transient CRC/bus glitches
        try:
            humidity = DHT_SENSOR.humidity
            temperature = DHT_SENSOR.temperature
        except RuntimeError:
            humidity, temperature = None, None  # keep going

        print("TEMP:", f2(temperature))
        print("HUMIDITY:", humidity)

        try:
            print("FlueT:")
            print(f"{f2(flueT.reference_temperature)} cold junction")
            print(flueT.fault)  # 0 if OK; else bitfield
            print("Temp:", f2(flueT.temperature))

            print("sttT:")
            print(f"{f2(sttT.reference_temperature)} cold junction")
            print(sttT.fault)
            print("Temp:", f2(sttT.temperature))
        except Exception as e:
            print("Thermocouple read error:", e)

        print("-------------------------------------------------------")
        t0 = now

    time.sleep(0.05)  # prevent busy-waiting
