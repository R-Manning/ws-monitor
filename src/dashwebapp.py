# dash_app.py
import os
import time
import threading
import sqlite3
import math
from collections import deque
from contextlib import closing
from typing import Tuple, List, Dict, Any, Optional

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)

import cv2
import plotly.graph_objects as go
import signal, sys

import dash_mantine_components as dmc
from dash import Dash, dash_table, dcc, html, Input, Output
from dash.dependencies import ClientsideFunction
from flask import Response
from paths import get_db_path
from database import connect, initialize
from metrics import get_metrics
from threading import Lock, Timer


cv2.setNumThreads(1)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Track viewers and manage the capture thread lazily
active_viewers = 0
_viewers_lock = Lock()
_idle_stop_timer = None
IDLE_GRACE_S = 10  # wait this long after last viewer disconnects before stopping capture


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DB_PATH = str(get_db_path())
with connect(DB_PATH) as _conn:
    initialize(_conn)

DATA_TABLE = "stove_room"
SETTINGS_TABLE = "settings"
MAX_POINTS = 5000

# Column order we expect from stove_room for plotting/metrics
DATA_COLS = ["datetime", "flueF", "sttF", "tempF", "humid"]

# Example Wyze RTSP URL pattern (adjust for your camera):
# rtsp://username:password@CAMERA_IP/live
RTSP_URL = os.getenv("WYZE_RTSP_URL")
CAMERA_ENABLED = os.getenv("WSM_CAMERA_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
CAMERA_ENABLED = CAMERA_ENABLED and bool(RTSP_URL)

# Throttle the capture frame rate (processing + bandwidth saver)
TARGET_FPS = float(os.getenv("TARGET_FPS", "5"))  # frames per second
RECONNECT_DELAY_S = float(os.getenv("RECONNECT_DELAY_S", "3"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "50"))  # 0-100
FRAME_BUFFER_SIZE = int(os.getenv("FRAME_BUFFER_SIZE", "1"))  # keep latest N frames

# =========================
# Shared state
# =========================
latest_frames = deque(maxlen=FRAME_BUFFER_SIZE)  # store the most recent frames (BGR np.array)
capture_thread = None
stop_event = threading.Event()

# ──────────────────────────────────────────────────────────────────────────────
# Settings cache (re-reads DB at most once per minute)
# ──────────────────────────────────────────────────────────────────────────────
_settings_cache: Optional[Tuple[float, Tuple[int, int]]] = None
_SETTINGS_CACHE_TTL = 60.0


def get_settings_cached() -> Tuple[int, int]:
    """Return (sample_period_s, rate_window_s) from settings table, cached for 60s."""
    global _settings_cache
    now = time.monotonic()
    if _settings_cache is not None:
        cached_at, cached_val = _settings_cache
        if now - cached_at < _SETTINGS_CACHE_TTL:
            return cached_val
    _settings_cache = (now, _get_settings_raw())
    return _settings_cache[1]


def _get_settings_raw() -> Tuple[int, int]:
    with connect(DB_PATH) as conn:
        initialize(conn)
        row = conn.execute(f"SELECT * FROM {SETTINGS_TABLE} LIMIT 1;").fetchone()
    if row is None:
        return 5, 60
    sample_period_s = int(row[0])
    rate_window_s = int(row[3]) if len(row) >= 4 else 60
    return sample_period_s, rate_window_s


# ──────────────────────────────────────────────────────────────────────────────
# Graph cache (avoids rebuilding Plotly figure every tick)
# ──────────────────────────────────────────────────────────────────────────────
_graph_cache: Optional[Tuple[Tuple, float, go.Figure]] = None


def _sql_rows(sql: str, params: tuple) -> List[Dict[str, Any]]:
    """Run a read query and return a list of dicts (replaces pandas.read_sql_query)."""
    with connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        initialize(conn)
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_downsampled(timeframe_minutes: int) -> List[Dict[str, Any]]:
    """AVG-downsampled data covering the full window (<= MAX_POINTS rows)."""
    window_s = max(60, int(timeframe_minutes) * 60)
    sample_period_s, _ = get_settings_cached()
    sample_period_s = max(1, int(sample_period_s))
    bucket_s = max(sample_period_s, math.ceil(window_s / MAX_POINTS))

    sql = f"""
    WITH windowed AS (
        SELECT datetime, flueF, sttF, tempF, humid
        FROM "{DATA_TABLE}"
        WHERE datetime >= datetime('now','localtime', ?)
    ),
    bucketed AS (
        SELECT
            (CAST(strftime('%s', datetime) AS INTEGER) / ?) * ? AS bucket_epoch,
            flueF, sttF, tempF, humid
        FROM windowed
    )
    SELECT
        bucket_epoch AS t_epoch,
        AVG(flueF) AS flueF, AVG(sttF) AS sttF,
        AVG(tempF) AS tempF, AVG(humid) AS humid
    FROM bucketed
    GROUP BY bucket_epoch
    ORDER BY bucket_epoch ASC;
    """
    return _sql_rows(sql, (f"-{int(timeframe_minutes)} minutes", bucket_s, bucket_s))


# ──────────────────────────────────────────────────────────────────────────────
# App logic
# ──────────────────────────────────────────────────────────────────────────────
import plotly.io as pio

# Precompute colorway once (avoids template lookup on every tick)
_COLORWAY = pio.templates["plotly_dark"].layout.colorway or [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA"
]

_SERIES = ["flueF", "sttF", "tempF", "humid"]
_DISPLAY_NAMES = {"flueF": "Flue", "sttF": "STT", "tempF": "Room", "humid": "Humidity"}
_COLOR_MAP = {var: _COLORWAY[i % len(_COLORWAY)] for i, var in enumerate(_SERIES)}


def _rgba_from_hex(h: str, a: float) -> str:
    h = h.strip()
    if h.startswith("#") and len(h) == 7:
        r = int(h[1:3], 16); g = int(h[3:5], 16); b = int(h[5:7], 16)
        return f"rgba({r},{g},{b},{a})"
    if h.startswith("rgba("):
        base = h[:h.rfind(",")]
        return f"{base}, {a})"
    if h.startswith("rgb("):
        return h.replace("rgb(", "rgba(").replace(")", f", {a})")
    return f"rgba(255,255,255,{a})"


def update_graph(timeframe_minutes: int = 360, compact: bool = True) -> go.Figure:
    """Build chart figure, served from cache when inputs haven't changed."""
    global _graph_cache
    now = time.monotonic()
    cache_key = (timeframe_minutes, compact)

    if _graph_cache is not None:
        prev_key, prev_time, prev_fig = _graph_cache
        if prev_key == cache_key and (now - prev_time) < 30.0:
            return prev_fig

    fig = _build_figure(timeframe_minutes, compact)
    _graph_cache = (cache_key, now, fig)
    return fig


def _build_figure(timeframe_minutes: int, compact: bool) -> go.Figure:
    rows = get_downsampled(timeframe_minutes)

    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, text="Waiting for data...", showarrow=False,
            font=dict(size=16), xref="paper", yref="paper"
        )
        fig.update_layout(template="plotly_dark",
                          margin=dict(l=8, r=8, t=8, b=8))
        return fig

    datetimes = [r["datetime"] for r in rows]
    fig = go.Figure()

    # Main lines only (4 traces instead of 12 — no min/max bands)
    for var in _SERIES:
        fig.add_trace(go.Scatter(
            x=datetimes, y=[r[var] for r in rows],
            mode="lines",
            name=_DISPLAY_NAMES.get(var, var),
            line=dict(color=_COLOR_MAP[var], width=2),
        ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        hoverlabel=dict(font_size=10),
        uirevision="fixed",
        margin=dict(l=8, r=8, t=8, b=8) if compact else dict(l=40, r=20, t=30, b=40),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.1 if compact else -0.25,
        )
    )

    if compact:
        fig.update_xaxes(title=None, tickfont=dict(size=10), nticks=4,
                         tickformat="%H:%M", automargin=True)
        fig.update_yaxes(title=None, tickfont=dict(size=10), automargin=True)
    else:
        fig.update_layout(
            title={"text": "Stove Room", "x": 0.5},
            xaxis_title="Date-Time",
            yaxis_title="Temperature (F) / Humidity (%)",
            legend_title_text=""
        )

    return fig


def update_metrics() -> list[dict]:
    """Return current values and rate/min rows without pandas."""
    rows = get_metrics(DB_PATH)
    return [
        {key: ("Unavailable" if value is None else value) for key, value in row.items()}
        for row in rows
    ]


#CAMERA

def _open_capture():
    # Tell FFmpeg/OpenCV to be conservative
    # You can also export OPENCV_FFMPEG_CAPTURE_OPTIONS env var if desired.
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # tiny buffer
    # Request smaller decode size if your cam supports stream params; often ignored:
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    return cap

def capture_loop():
    """Read RTSP, but only *decode* the frames we intend to show."""
    global latest_frames
    target_fps = max(0.5, TARGET_FPS)
    frame_interval = 1.0 / target_fps
    reconnect = max(0.5, RECONNECT_DELAY_S)

    while not stop_event.is_set():
        cap = _open_capture()
        if not cap.isOpened():
            time.sleep(reconnect)
            continue

        next_keep = time.time()
        # Use grab/retrieve to avoid decoding every frame
        while not stop_event.is_set():
            grabbed = cap.grab()
            if not grabbed:
                break  # lost stream; reconnect

            now = time.time()
            # Only decode + process at the cadence we really want
            if now >= next_keep:
                ok, frame = cap.retrieve()  # decode only here
                if not ok or frame is None:
                    break
                # Downscale aggressively before JPEG
                frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                latest_frames.append(frame)
                next_keep += frame_interval

            # Tiny sleep to avoid hot looping when source FPS is high
            time.sleep(0.002)

        cap.release()
        time.sleep(reconnect)
        

def _ensure_capture_running():
    """Start the capture thread if it's not already running."""
    global capture_thread, stop_event
    if not CAMERA_ENABLED:
        return
    if capture_thread is None or not capture_thread.is_alive():
        stop_event.clear()
        capture_thread = threading.Thread(target=capture_loop, name="RTSPCapture", daemon=True)
        capture_thread.start()


def _schedule_stop_if_idle():
    """Schedule a stop in IDLE_GRACE_S seconds, but cancel if a viewer arrives."""
    global _idle_stop_timer
    if _idle_stop_timer is not None:
        try:
            _idle_stop_timer.cancel()
        except Exception:
            pass
        _idle_stop_timer = None

    def _maybe_stop():
        global capture_thread
        with _viewers_lock:
            if active_viewers == 0:
                # No viewers—stop capture
                stop_event.set()
                # Join quickly to release the camera/decoder
                if capture_thread and capture_thread.is_alive():
                    capture_thread.join(timeout=3)
                capture_thread = None    
                
    _idle_stop_timer = Timer(IDLE_GRACE_S, _maybe_stop)
    _idle_stop_timer.daemon = True
    _idle_stop_timer.start()


def mjpeg_generator():
    """
    Throttled MJPEG generator that:
      - Starts capture on first viewer (handled by caller before yielding)
      - Encodes at most TARGET_FPS frames/sec
      - Reads the latest frame with race-safe guards
      - Schedules capture stop after last viewer disconnects
    """
    global active_viewers

    # Mark viewer connected
    with _viewers_lock:
        active_viewers += 1
        _ensure_capture_running()   # cancel pending stop & ensure capture is running

    try:
        # Wait for first frame to arrive
        while not latest_frames and not stop_event.is_set():
            time.sleep(0.05)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
        min_interval = 1.0 / max(0.5, float(TARGET_FPS))
        last_sent = 0.0

        # Stream until the client disconnects
        while not stop_event.is_set():
            # --- race-safe read of newest frame ---
            frame = None
            if latest_frames:
                try:
                    frame = latest_frames[-1]  # could raise if deque shrinks between check and read
                except IndexError:
                    frame = None

            if frame is None:
                time.sleep(0.02)
                continue

            # --- throttle send rate ---
            now = time.time()
            remaining = min_interval - (now - last_sent)
            if remaining > 0:
                time.sleep(min(remaining, 0.02))
                continue

            ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                time.sleep(0.01)
                continue

            last_sent = time.time()
            bytes_ = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(bytes_)).encode() + b"\r\n\r\n" +
                bytes_ + b"\r\n"
            )

    finally:
        # Client disconnected; decrement and maybe stop capture
        with _viewers_lock:
            active_viewers = max(0, active_viewers - 1)
            if active_viewers == 0:
                _schedule_stop_if_idle()


# ──────────────────────────────────────────────────────────────────────────────
# Dash App
# ──────────────────────────────────────────────────────────────────────────────

app = Dash(__name__, title="Wood Stove Monitor", update_title=None,
           suppress_callback_exceptions=True)
server = app.server  # <-- get the Flask app from Dash

@server.route("/video.mjpg")
def video_feed():
    # Chrome happily plays MJPEG via multipart/x-mixed-replace
    if not CAMERA_ENABLED:
        return Response("Camera unavailable", status=503, mimetype="text/plain")
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


app.layout = dmc.MantineProvider(
    theme={"colorScheme": "dark"},
    withGlobalStyles=True,
    withNormalizeCSS=True,
    children=[
        dmc.Container(
            fluid=True,
            style={"minHeight": "100vh"},
            children=[
                dmc.Title("Wood Stove Monitor", color="white", size="h2", align="center"),
                dmc.Grid(
                    [
                        dmc.Col(
                            [
                                dash_table.DataTable(
                                    id="metrics",
                                    data=update_metrics(),
                                    page_size=6,

                                    # NEW: make the table fill its container and not overflow horizontally
                                    fill_width=True,
                                    style_table={"width": "100%", "minWidth": "0", "overflowX": "hidden"},

                                    # NEW: fixed layout with ellipsis so long text/numbers don't push width
                                    css=[
                                        {"selector": "table", "rule": "table-layout: fixed; width: 100%;"},
                                        {"selector": ".dash-cell div.dash-cell-value", "rule": "white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"},
                                    ],

                                    # Shrink padding/fonts + allow wrapping in headers
                                    style_cell={
                                        "textAlign": "center",   # keep your alignment style key consistent with your app
                                        "padding": "4px 6px",
                                        "fontSize": 12,
                                        "minWidth": "0px",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(15, 15, 15)",
                                        "color": "white",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                        "width": "auto",
                                        "fontWeight": "bold",
                                        "fontSize": 12,          # smaller header on mobile too
                                        "padding": "4px 6px",
                                    },

                                    # NEW: tighten specific columns so five columns fit on small screens
                                    style_cell_conditional=[
                                        # Data-Type gets a bit more room but can wrap/ellipsis
                                        {"if": {"column_id": "Data-Type"}, "minWidth": "80px", "width": "90px", "maxWidth": "140px", "whiteSpace": "normal"},
                                        # Numeric columns: narrow
                                        {"if": {"column_id": "Flue (F)"},     "minWidth": "56px", "width": "1%", "maxWidth": "80px"},
                                        {"if": {"column_id": "STT (F)"},      "minWidth": "56px", "width": "1%", "maxWidth": "80px"},
                                        {"if": {"column_id": "Room (F)"},     "minWidth": "56px", "width": "1%", "maxWidth": "80px"},
                                        {"if": {"column_id": "Humidity (%)"}, "minWidth": "56px", "width": "1%", "maxWidth": "80px"},
                                    ],

                                    # keep your existing style_data/style_data_conditional below …
                                    style_data={"backgroundColor": "rgb(30, 30, 30)", "color": "white"},
                                    style_data_conditional=[
                                        # Data-Type column header styling
                                        {
                                            "if": {"column_id": "Data-Type"},
                                            "backgroundColor": "rgb(15, 15, 15)",
                                            "color": "white",
                                            "fontWeight": "bold",
                                        },
                                        # Humidity thresholds
                                        {
                                            "if": {
                                                "filter_query": "{Humidity (%)} >= 50",
                                                "column_id": "Humidity (%)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "dodgerblue",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Humidity (%)} <= 20",
                                                "column_id": "Humidity (%)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "tomato",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Humidity (%)} > 20 && {Humidity (%)} < 50",
                                                "column_id": "Humidity (%)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "rgb(30, 30, 30)",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Humidity (%)} >= 1 || {Humidity (%)} <= -1",
                                                "column_id": "Humidity (%)",
                                                "row_index": 1,
                                            },
                                            "backgroundColor": "tomato",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Humidity (%)} > -1 && {Humidity (%)} < 1",
                                                "column_id": "Humidity (%)",
                                                "row_index": 1,
                                            },
                                            "backgroundColor": "rgb(30, 30, 30)",
                                            "color": "white",
                                        },
                                        # Flue thresholds
                                        {
                                            "if": {
                                                "filter_query": "{Flue (F)} >= 450",
                                                "column_id": "Flue (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "tomato",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Flue (F)} <= 250",
                                                "column_id": "Flue (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "dodgerblue",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Flue (F)} > 250 && {Flue (F)} < 450",
                                                "column_id": "Flue (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "rgb(30, 30, 30)",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Flue (F)} >= 25 || {Flue (F)} <= -25",
                                                "column_id": "Flue (F)",
                                                "row_index": 1,
                                            },
                                            "backgroundColor": "tomato",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Flue (F)} > -25 && {Flue (F)} < 25",
                                                "column_id": "Flue (F)",
                                                "row_index": 1,
                                            },
                                            "backgroundColor": "rgb(30, 30, 30)",
                                            "color": "white",
                                        },
                                        # STT thresholds
                                        {
                                            "if": {
                                                "filter_query": "{STT (F)} >= 675",
                                                "column_id": "STT (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "tomato",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{STT (F)} <= 300",
                                                "column_id": "STT (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "dodgerblue",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{STT (F)} > 300 && {STT (F)} < 675",
                                                "column_id": "STT (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "rgb(30, 30, 30)",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{STT (F)} >= 25 || {STT (F)} <= -25",
                                                "column_id": "STT (F)",
                                                "row_index": 1,
                                            },
                                            "backgroundColor": "tomato",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{STT (F)} > -25 && {STT (F)} < 25",
                                                "column_id": "STT (F)",
                                                "row_index": 1,
                                            },
                                            "backgroundColor": "rgb(30, 30, 30)",
                                            "color": "white",
                                        },
                                        # Room thresholds
                                        {
                                            "if": {
                                                "filter_query": "{Room (F)} >= 90",
                                                "column_id": "Room (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "tomato",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Room (F)} <= 66",
                                                "column_id": "Room (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "dodgerblue",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Room (F)} > 66 && {Room (F)} < 90",
                                                "column_id": "Room (F)",
                                                "row_index": 0,
                                            },
                                            "backgroundColor": "rgb(30, 30, 30)",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Room (F)} >= 1 || {Room (F)} <= -1",
                                                "column_id": "Room (F)",
                                                "row_index": 1,
                                            },
                                            "backgroundColor": "tomato",
                                            "color": "white",
                                        },
                                        {
                                            "if": {
                                                "filter_query": "{Room (F)} > -1 && {Room (F)} < 1",
                                                "column_id": "Room (F)",
                                                "row_index": 1,
                                            },
                                            "backgroundColor": "rgb(30, 30, 30)",
                                            "color": "white",
                                        },
                                    ],
                                )
                            ],
                            span=12,
                        ),
                        
                        # --- Controls row (full width, optional) ---
                        # GRAPH column
                        dmc.Col(
                            [
                                # centered control with fixed height + bottom gap
                                html.Div(
                                    dmc.Select(
                                        id="timeframe-dropdown",
                                        data=[
                                            {"label": "Last 15 minutes", "value": 15},
                                            {"label": "Last 30 minutes", "value": 30},
                                            {"label": "Last 1 hour", "value": 60},
                                            {"label": "Last 3 hours", "value": 180},
                                            {"label": "Last 6 hours", "value": 360},
                                            {"label": "Last 12 hours", "value": 720},
                                            {"label": "Last 24 hours", "value": 1440},
                                        ],
                                        value=360,
                                        label="Select Timeframe",
                                        style={"width": 250, "color": "white"},
                                        sx={
                                                "label": {"color": "white"},
                                                "input": {"color": "white"},
                                                ".mantine-Select-item": {"color": "white"},
                                            },
                                    ),
                                    style={
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "alignItems": "center",
                                        "height": "56px",        # control height
                                        "marginBottom": "12px",  # gap above graph border
                                    },
                                ),

                                # 4:3 graph box with border; graph fills it
                                html.Div(
                                    style={
                                        "position": "relative",
                                        "width": "100%",
                                        "aspectRatio": "16/9",
                                        "boxSizing": "border-box",
                                        "border": "1px solid white",
                                    },
                                    children=dcc.Graph(
                                        id="graph",
                                        figure=update_graph(compact=True),
                                        config={"responsive": True, "displayModeBar": False},
                                        style={"position": "absolute", "inset": 0, "height": "100%", "width": "100%"},
                                    ),
                                ),
                            ],
                            span=12, md=6,
                        ),

                        # VIDEO column
                        dmc.Col(
                            [
                                # empty spacer that matches the dropdown box so the media boxes align
                                html.Div(
                                        html.H4(
                                            "Live Stove Feed",
                                            style={"margin": 0, "color": "white", "textAlign": "center"},
                                        ),
                                        style={
                                            "display": "flex",
                                            "justifyContent": "center",
                                            "alignItems": "center",
                                            "height": "56px",        # ← match the dropdown wrapper height
                                            "marginBottom": "12px",  # ← same gap as graph
                                        },
                                ),

                                html.Div(
                                    style={
                                        "position": "relative",
                                        "width": "100%",
                                        "aspectRatio": "16/9",
                                        "boxSizing": "border-box",
                                        "border": "1px solid white",
                                        "overflow": "hidden",
                                    },
                                    children=html.Img(
                                        id="video",
                                        src="/video.mjpg",
                                        draggable=False,
                                        style={"position": "absolute", "inset": 0, "width": "100%", "height": "100%", "objectFit": "contain"},
                                    ),
                                ),
                            ],
                            span=12, md=6,
                        ),

                    ],
                    justify="center",
                    align="stretch",
                ),
               # Interval is set dynamically from settings via callback below
                dcc.Interval(
                    id="interval-component",
                    interval=_get_settings_raw()[0] * 1000,
                    n_intervals=0,
                ),
                dcc.Store(id="ui-flags", data={"compact": False}, storage_type="memory"),
                dcc.Interval(id="ui-detect", interval=300, n_intervals=0, max_intervals=1),
            ],
        ),
        
    ],
)


def main():
    try:
        app.run(host="0.0.0.0", port=8050, debug=False)
    finally:
        # Ensure clean shutdown
        stop_event.set()
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=3)


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

app.clientside_callback(
    ClientsideFunction(namespace="ui", function_name="detectCompact"),
    Output("ui-flags", "data"),
    Input("ui-detect", "n_intervals"),
)


@app.callback(
    Output("metrics", "columns"),
    Input("ui-flags", "data"),
)
def metrics_columns(ui_flags):
    compact = bool(ui_flags and ui_flags.get("compact"))
    if compact:
        # Short, mobile-friendly header labels (ids stay the same)
        return [
            {"name": "Data",     "id": "Data-Type"},
            {"name": "Flue", "id": "Flue (F)"},
            {"name": "STT",  "id": "STT (F)"},
            {"name": "Room", "id": "Room (F)"},
            {"name": "Humid", "id": "Humidity (%)"},
        ]
    # Desktop / default labels
    return [
        {"name": "Data-Type",    "id": "Data-Type"},
        {"name": "Flue (F)",     "id": "Flue (F)"},
        {"name": "STT (F)",      "id": "STT (F)"},
        {"name": "Room (F)",     "id": "Room (F)"},
        {"name": "Humidity (%)", "id": "Humidity (%)"},
    ]


@app.callback(
    Output("graph", "figure"),
    Input("interval-component", "n_intervals"),
    Input("timeframe-dropdown", "value"),
    Input("ui-flags", "data"),
)
def refresh_graph(_n, timeframe_minutes, ui_flags):
    compact = bool(ui_flags.get("compact")) if ui_flags else True
    return update_graph(timeframe_minutes, compact=compact)


@app.callback(
    Output("metrics", "data"),
    Input("interval-component", "n_intervals"),
    Input("ui-flags", "data"),
)
def refresh_metrics(_n, ui_flags):
    rows = update_metrics()  # [{ "Data-Type": "...", "Flue (F)": ..., ... }, ...]
    compact = bool(ui_flags and ui_flags.get("compact"))

    if compact:
        # Short mobile-friendly labels for the first column
        label_map = {
            "Current Values": "Live",
            "Rate/minute": "Δ/min",   # or "Rate/min"
        }
        for r in rows:
            if "Data-Type" in r:
                r["Data-Type"] = label_map.get(r["Data-Type"], r["Data-Type"])
    else:
        # Ensure desktop labels are restored (especially after rotate)
        label_map = {
            "Live": "Current Values",
            "Δ/min": "Rate/minute",
        }
        for r in rows:
            if "Data-Type" in r:
                r["Data-Type"] = label_map.get(r["Data-Type"], r["Data-Type"])

    return rows


@app.callback(Output("interval-component", "interval"), Input("interval-component", "n_intervals"))
def refresh_interval(_n):
    """
    Re-read settings periodically so changes on-disk (sample period)
    propagate without restarting the app.
    """
    sample_period_s, _ = get_settings_cached()
    return max(1, int(sample_period_s)) * 1000
    
    
def _shutdown(*_args):
    """Handle SIGINT/SIGTERM: stop capture thread and exit cleanly."""
    stop_event.set()
    try:
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=3)
    except Exception:
        pass
    sys.exit(0)

# Register handlers
signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)
    

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
