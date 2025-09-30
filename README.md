# Wood Stove Monitor

A Python-based project for monitoring a wood stove’s performance and surrounding 
environment.  
The system tracks:

- **Flue / Stove Temperature (°F)**
- **Room Temperature (°F)**
- **Room Humidity (%)**

All readings are stored in a **SQLite3 database**, and a **Dash web app** provides 
real-time visualization and historical analysis.

---

## Features

### Data Collection
- Captures stove and room environment data.
- Stores measurements in a lightweight SQLite3 database for persistence.

### Web App
- **Table View**: Displays the most recent values and 60-minute deltas.
  - Cells are highlighted based on acceptable value ranges.
- **Graph View**: Shows historical trends with selectable timeframes.
- **Video Stream**: Embeds an RTSP feed of the stove (via OpenCV / MJPEG).

### Alerts
- Utilize a telegram bot to send notifications to a phone based on acceptable
    variable ranges

---

## Installation

Copy project folder onto machine
Install dependencies

---

## Project Structure

WS_Monitor/
├─ src/
│  ├─ assets/
│  │  ├─ clientside.js
│  │  └─ favicon.ico
│  ├─ resources/
│  │  └─ house_environment.db
│  ├─ dashWebApp.py
│  ├─ paths.py
│  ├─ rpiacqscript.py
│  ├─ telegramAlert.py
│  └─ watchDog.py
├─ tests/
│  └─ loggertest.py
├─ .env
├─ .gitignore
├─ README.md
└─ requirements.txt

