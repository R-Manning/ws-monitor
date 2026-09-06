# Wood Stove Monitor

A Python-based project for monitoring a wood stove’s performance and surrounding 
environment.  
The system tracks:

- **Flue / Stove Temperature (°F)**
- **Room Temperature (°F)**
- **Room Humidity (%)**

All readings are stored in a **SQLite3 database**, and an optional **Dash web app**
provides real-time visualization and historical analysis.

---

## Features

### Data Collection
- Captures stove and room environment data.
- Stores measurements in a lightweight SQLite3 database for persistence.

### Web App
- **Table View**: Displays the most recent values and 60-minute deltas.
  - Cells are highlighted based on acceptable value ranges.
- **Graph View**: Shows historical trends with selectable timeframes.
- **Video Stream**: Embeds an RTSP feed of the stove (via WebRTC low-latency H.264 relay, no server-side transcoding).

### Alerts
- Utilize a telegram bot to send notifications to a phone based on acceptable
    variable ranges

---

## Installation

The dependency files are split by role:

```sh
python3 -m venv .venv
python3 -m pip install -r requirements/dev.txt
```

Python 3.10 or newer is required for `requirements/dashboard.txt` (aiortc).
On macOS, use `brew install python@3.12` (or `uv venv --python 3.12` if
using `uv`) before creating the virtual environment.

`requirements/base.txt` is sufficient for the collector, diagnostics, and
watchdog. Install `requirements/pi.txt` on a Raspberry Pi for the sensor
drivers, and `requirements/dashboard.txt` where the Dash UI and camera support
are needed. The old combined `requirements.txt` is intentionally not used.

## macOS Simulated Setup

The simulated reader does not import Raspberry Pi packages, so it is suitable
for development and commissioning on macOS. It must be selected explicitly;
`auto` and `hardware` both fail if Pi hardware initialization fails, preventing
synthetic readings from entering a production database:

```sh
cp .env_template .env
mkdir -p runtime
WSM_DB_PATH="$PWD/runtime/house_environment.db" \
  .venv/bin/python src/rpiacqscript.py --sensor-mode simulated --samples 3
WSM_DB_PATH="$PWD/runtime/house_environment.db" \
  .venv/bin/python src/rpiacqscript.py --sensor-mode simulated --diagnose
```

Use `--once` for a single sample. Set `WSM_CAMERA_ENABLED=false` in `.env` on
macOS unless an RTSP camera is available. The dashboard is optional; the
collector, database, metrics, and tests do not require Dash, pandas, aiortc,
or Raspberry Pi packages.

## Raspberry Pi Services

For a Pi deployment, create `/opt/ws-monitor`, a virtual environment there,
and install `requirements/pi.txt` and `requirements/dashboard.txt` as needed.
Copy `.env_template` to `/etc/ws-monitor/ws-monitor.env`, set the Telegram and
camera values, and use a persistent runtime database path such as:

```text
WSM_DB_PATH=/var/lib/ws-monitor/house_environment.db
WSM_SENSOR_MODE=hardware
```

Create `/var/lib/ws-monitor` with ownership for the service user. Replace the
`/opt/ws-monitor` and `User=pi` values in the templates under
`deploy/systemd/` if the checkout or service account differs. Install the
collector and dashboard `.service` files, then enable both independent units:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now ws-monitor-collector.service
sudo systemctl enable --now ws-monitor-dashboard.service
```

Unit naming: the canonical service names are `ws-monitor-collector.service`,
`ws-monitor-dashboard.service`, and `ws-monitor-watchdog.service` (below).
Deployments that predate the rename may still be running as `rpiacqscript.service`,
`dashweb.service`, and `dashweb-watchdog.service`; after the rename migration
those old names are just aliases of the canonical units and keep working.

### Resilience and automatic recovery

The dashboard service is hardened against three failure modes:

1. **Process crash** — `Restart=always` + `RestartSec=3` in the unit file makes
   systemd relaunch it immediately, and `TimeoutStopSec=10` forces a fast
   `SIGKILL` if a shutdown ever hangs.

2. **Process hang (alive but unresponsive)** — the app exposes a lightweight
   liveness endpoint `GET /_healthz` (returns 200 whenever Flask can serve).
   The optional `ws-monitor-watchdog.service` polls it every 15 s and runs
   `systemctl restart ws-monitor-dashboard.service` after 3 consecutive failures
   (min 90 s between restarts to avoid thrash):

   ```sh
   sudo install -m 0755 deploy/scripts/ws-monitor-watchdog.sh /usr/local/sbin/ws-monitor-watchdog.sh
   sudo install -m 0644 deploy/systemd/ws-monitor-watchdog.service.template /etc/systemd/system/ws-monitor-watchdog.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now ws-monitor-watchdog.service
   ```

   Watchdog activity is logged under the `ws-monitor-watchdog` journal tag; the
   app logs `WEBRTC:` lines for in-process restarts.

3. **Camera/RTSP outage or abandoned signaling** — inside the WebRTC loop the
   app runs two background coroutines:
   - `_monitor_player_health()` subscribes a probe track to the relay every
     10 s; if no frame arrives within 5 s across 2 consecutive checks it
     recreates the player/relay in-place (browsers reconnect within ~3 s).
   - `_sweep_stale_peers()` expires unanswered offers (30 s) and peer
     connections stuck in `new`/`connecting` (60 s), preventing orphaned
     peer objects from accumulating (the "no video after a stray offer"
     failure mode).

Alerting runs inside the collector after each committed sample. Missing Telegram
credentials or delivery failures will not stop collection.

### STT MAX31856 startup repair

The STT MAX31856 on GPIO D6 has a hardware defect that restores corrupted
threshold registers after power loss. Collector startup restores its known-good
full-range defaults and clears the latched fault status before sampling. The
flue MAX31856 is not changed. Repair write/readback failures stop the collector
so systemd can report and restart it; replace the defective STT module when
possible.

## Runtime Checks and Diagnostics

Check service state and recent logs with:

```sh
systemctl status ws-monitor-collector.service
journalctl -u ws-monitor-collector.service -n 100 --no-pager
```

Run sensor diagnostics without writing a sample:

```sh
WSM_DB_PATH=/var/lib/ws-monitor/house_environment.db \
  .venv/bin/python src/rpiacqscript.py --diagnose --sensor-mode hardware
```

The database path is controlled by `WSM_DB_PATH` and parent directories are
created by the database connection layer. Keep the database outside the code
checkout on the Pi so upgrades do not overwrite runtime state. SQLite WAL
sidecars and local databases are ignored by git.

---

## Project Structure

ws-monitor/
├─ src/
│  ├─ assets/
│  │  ├─ clientside.js
│  │  └─ favicon.ico
│  ├─ resources/             # hardware/static resources only
│  ├─ dashwebapp.py
│  ├─ paths.py
│  ├─ rpiacqscript.py
│  ├─ telegramAlert.py
│  └─ watchDog.py
├─ tests/
│  ├─ test_database_sensors.py
│  └─ test_runtime_behaviors.py
├─ deploy/systemd/
├─ requirements/
├─ .env
├─ .gitignore
├─ README.md
└─ requirements/dev.txt
