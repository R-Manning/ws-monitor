# dash_app.py
import os
import time
import threading
import sqlite3
import math
import re
import json
import asyncio
import traceback
import uuid
from contextlib import closing
from typing import Tuple, List, Dict, Any, Optional

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)

import signal, sys
import plotly.graph_objects as go

import dash_mantine_components as dmc
from dash import Dash, dash_table, dcc, html, Input, Output
from dash.dependencies import ClientsideFunction
from flask import Response, request
from paths import get_db_path
from database import connect, initialize
from metrics import get_metrics
from threading import Lock, Timer

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay

# Track viewers and manage the WebRTC session lazily
active_viewers = 0
_viewers_lock = Lock()
_idle_stop_timer = None
IDLE_GRACE_S = 10  # wait this long after last viewer disconnects before stopping the stream


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DB_PATH = str(get_db_path())
with connect(DB_PATH) as _conn:
    initialize(_conn)

DATA_TABLE = "stove_room"
SETTINGS_TABLE = "settings"
MAX_POINTS = 5000
GRAPH_INTERVAL_S = 15

# Column order we expect from stove_room for plotting/metrics
DATA_COLS = ["datetime", "flueF", "sttF", "tempF", "humid"]

# Example Wyze RTSP URL pattern (adjust for your camera):
# rtsp://username:password@CAMERA_IP/live
RTSP_URL = os.getenv("WYZE_RTSP_URL")
CAMERA_ENABLED = os.getenv("WSM_CAMERA_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
CAMERA_ENABLED = CAMERA_ENABLED and bool(RTSP_URL)

RECONNECT_DELAY_S = float(os.getenv("RECONNECT_DELAY_S", "3"))

# =========================
# WebRTC shared state
# =========================
_webrtc_player = None       # aiortc MediaPlayer — single RTSP connection
_webrtc_relay = None        # aiortc MediaRelay — fan-out to multiple viewers
_webrtc_lock = Lock()
_webrtc_loop = None         # asyncio event loop for aiortc
_webrtc_thread = None       # daemon thread running the event loop
_webrtc_supervisors_running = False
_active_peer_connections = set()
_pending_peer_connections = {}
_pc_started: Dict[int, float] = {}       # id(pc) -> creation time, for stale cleanup

# Hanging-chad / stale-connection protection
PENDING_PEER_TTL_S = 30.0    # offer handed out but browser never answered
STUCK_PEER_TTL_S = 60.0      # PC stuck in new/connecting this long gets closed
SWEEP_INTERVAL_S = 15.0

# Camera self-heal
HEALTH_CHECK_INTERVAL_S = 10.0
HEALTH_RECV_TIMEOUT_S = 5.0
HEALTH_MAX_MISSES = 2

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
            datetime, flueF, sttF, tempF, humid
        FROM windowed
    )
    SELECT
        bucket_epoch AS t_epoch,
        MIN(datetime) AS datetime,
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
        if prev_key == cache_key and (now - prev_time) < GRAPH_INTERVAL_S:
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
    rows = get_metrics(DB_PATH, ensure_schema=False)
    return [
        {key: ("Unavailable" if value is None else value) for key, value in row.items()}
        for row in rows
    ]


# ──────────────────────────────────────────────────────────────────────────────
# WebRTC camera pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _stop_player_tracks(player):
    """Starve the publisher task so loop.stop() and restarts return promptly."""
    track = getattr(player, "video", None)
    if track is not None:
        try:
            track.stop()
        except Exception:
            pass
    audio = getattr(player, "audio", None)
    if audio is not None:
        try:
            audio.stop()
        except Exception:
            pass


def _ensure_webrtc_running():
    """Start the RTSP MediaPlayer and asyncio loop on first viewer."""
    global _webrtc_player, _webrtc_relay, _webrtc_loop, _webrtc_thread

    with _webrtc_lock:
        if _webrtc_player is not None:
            return

        # Reuse a live loop (e.g. after a restart); only build a new one if the
        # previous loop was stopped during teardown.
        if _webrtc_loop is None or _webrtc_loop.is_closed():
            _webrtc_loop = asyncio.new_event_loop()

            def _run_loop():
                asyncio.set_event_loop(_webrtc_loop)
                _webrtc_loop.run_forever()

            _webrtc_thread = threading.Thread(target=_run_loop, daemon=True)
            _webrtc_thread.start()

        try:
            future = asyncio.run_coroutine_threadsafe(
                _init_player_if_missing(), _webrtc_loop
            )
            future.result(timeout=15)
        except Exception as exc:
            print(f"WEBRTC: player init failed: {exc}", flush=True)
            return

        # Launch camera-health monitor + stale-peer sweeper on the loop.
        _webrtc_loop.call_soon_threadsafe(_start_supervisors)


async def _init_player_if_missing():
    """Create the MediaPlayer (and relay) on the running loop if absent."""
    global _webrtc_player, _webrtc_relay
    if _webrtc_player is None:
        _webrtc_player = MediaPlayer(RTSP_URL, decode=False)
    if _webrtc_relay is None:
        _webrtc_relay = MediaRelay()


def _teardown_webrtc():
    """Close the RTSP player and event loop when no viewers remain."""
    global _webrtc_player, _webrtc_relay, _webrtc_loop, _webrtc_thread
    global _webrtc_supervisors_running

    with _webrtc_lock:
        if _webrtc_player is None:
            _webrtc_supervisors_running = False
            return

        loop = _webrtc_loop
        thread = _webrtc_thread

        # Close all active peer connections
        for pc in list(_active_peer_connections):
            try:
                future = asyncio.run_coroutine_threadsafe(pc.close(), loop)
                future.result(timeout=5)
            except Exception:
                pass
        _active_peer_connections.clear()
        _pc_started.clear()
        _pending_peer_connections.clear()

        # Stop MediaPlayer tracks BEFORE the loop stops (prevents a busy loop on stop).
        _stop_player_tracks(_webrtc_player)

        # Stop the event loop (this tears down MediaPlayer tracks)
        try:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
        except Exception:
            pass

        # Null out references — GC handles MediaPlayer/Relay cleanup
        _webrtc_player = None
        _webrtc_relay = None
        _webrtc_loop = None
        _webrtc_thread = None
        _webrtc_supervisors_running = False


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
        with _viewers_lock:
            if active_viewers == 0:
                _teardown_webrtc()

    _idle_stop_timer = Timer(IDLE_GRACE_S, _maybe_stop)
    _idle_stop_timer.daemon = True
    _idle_stop_timer.start()


def _cancel_pending_stop():
    """Cancel any scheduled idle stop (a new viewer arrived)."""
    global _idle_stop_timer
    if _idle_stop_timer is not None:
        try:
            _idle_stop_timer.cancel()
        except Exception:
            pass
        _idle_stop_timer = None


# ──────────────────────────────────────────────────────────────────────────────
# WebRTC supervisors (camera self-heal + stale-connection cleanup)
# ──────────────────────────────────────────────────────────────────────────────

def _start_supervisors():
    """Launch background coroutines on the WebRTC loop (runs on the loop)."""
    global _webrtc_supervisors_running
    if _webrtc_supervisors_running:
        return
    _webrtc_supervisors_running = True

    async def _launch():
        asyncio.create_task(_monitor_player_health())
        asyncio.create_task(_sweep_stale_peers())

    asyncio.ensure_future(_launch())


async def _restart_player():
    """Drop stale peer connections and recreate relay/player in-place.

    Runs on the WebRTC loop; browsers retry within ~3s and get a fresh relay.
    """
    global _webrtc_player, _webrtc_relay

    for pc in list(_active_peer_connections):
        try:
            await pc.close()
        except Exception:
            pass
    _active_peer_connections.clear()
    _pc_started.clear()
    _pending_peer_connections.clear()

    if _webrtc_player is not None:
        _stop_player_tracks(_webrtc_player)
        _webrtc_player = None
    if _webrtc_relay is not None:
        try:
            await _webrtc_relay.stop()
        except Exception:
            pass
        _webrtc_relay = None

    try:
        _webrtc_relay = MediaRelay()
        _webrtc_player = MediaPlayer(RTSP_URL, decode=False)
    except Exception as exc:
        print(f"WEBRTC: player restart failed: {exc}", flush=True)
        _webrtc_player = None


async def _monitor_player_health():
    """Periodically verify the camera is producing frames and restart the
    pipeline in-place if it has stalled."""
    misses = 0
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL_S)
        player = _webrtc_player
        relay = _webrtc_relay
        if player is None or relay is None or player.video is None:
            misses += 1
            if misses >= HEALTH_MAX_MISSES:
                print("WEBRTC: camera pipeline unavailable; restarting player", flush=True)
                await _restart_player()
                misses = 0
            continue
        probe = None
        try:
            probe = relay.subscribe(player.video)
            await asyncio.wait_for(probe.recv(), timeout=HEALTH_RECV_TIMEOUT_S)
            misses = 0
        except Exception:
            misses += 1
        finally:
            if probe is not None:
                try:
                    probe.stop()
                except Exception:
                    pass
        if misses >= HEALTH_MAX_MISSES:
            print("WEBRTC: camera pipeline stalled; restarting player", flush=True)
            await _restart_player()
            misses = 0


async def _sweep_stale_peers():
    """Close orphaned / half-open peer connections (hanging-chad guard).

    Handles two leaks: offered PCs the browser never answered, and active PCs
    stuck in new/connecting that can never reach a usable state on their own.
    """
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_S)
        now = time.time()

        # Unanswered offers (signaling abandoned mid-handshake)
        for pc_id in list(_pending_peer_connections):
            entry = _pending_peer_connections.get(pc_id)
            if entry is None:
                continue
            pc, ts = entry
            if now - ts > PENDING_PEER_TTL_S:
                _pending_peer_connections.pop(pc_id, None)
                _active_peer_connections.discard(pc)
                _pc_started.pop(id(pc), None)
                try:
                    await pc.close()
                except Exception:
                    pass

        # Active PCs stuck before they ever connected
        for pc in list(_active_peer_connections):
            if pc.connectionState in ("new", "connecting"):
                ts = _pc_started.get(id(pc))
                if ts is not None and now - ts > STUCK_PEER_TTL_S:
                    _active_peer_connections.discard(pc)
                    _pc_started.pop(id(pc), None)
                    try:
                        await pc.close()
                    except Exception:
                        pass


# ──────────────────────────────────────────────────────────────────────────────
# Dash App & WebRTC signaling
# ──────────────────────────────────────────────────────────────────────────────

app = Dash(__name__, title="Wood Stove Monitor", update_title=None,
           suppress_callback_exceptions=True)
server = app.server  # <-- get the Flask app from Dash


@server.route("/webrtc/offer", methods=["GET"])
def webrtc_offer():
    """Server-initiated: create a PC with a video track, return an SDP offer."""
    if not CAMERA_ENABLED:
        return Response("Camera unavailable", status=503, mimetype="text/plain")

    try:
        _ensure_webrtc_running()
    except Exception as e:
        return Response(f"Camera connection failed: {e}", status=503)

    async def _handle():
        global active_viewers
        pc = RTCPeerConnection()
        _active_peer_connections.add(pc)
        _pc_started[id(pc)] = time.time()

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            global active_viewers
            if pc.connectionState in ("failed", "closed", "disconnected"):
                _active_peer_connections.discard(pc)
                _pc_started.pop(id(pc), None)
                with _viewers_lock:
                    active_viewers = max(0, active_viewers - 1)
                    if active_viewers == 0:
                        _schedule_stop_if_idle()
                try:
                    await pc.close()
                except Exception:
                    pass

        with _webrtc_lock:
            if _webrtc_relay is None or _webrtc_player is None:
                await pc.close()
                return {"error": "Camera not ready"}
            pc.addTrack(_webrtc_relay.subscribe(_webrtc_player.video))

        offer = await pc.createOffer()

        # Strip VP8 from the SDP offer before setting it, so Chrome is
        # forced to pick H.264. Chrome always prefers VP8, but our raw
        # H.264 packets won't decode under a VP8 codec — every frame
        # gets dropped.
        sdp = offer.sdp
        vp8_pt_match = re.search(r"a=rtpmap:(\d+) VP8/", sdp)
        if vp8_pt_match:
            vp8_pt = vp8_pt_match.group(1)
            sdp = re.sub(rf"a=rtpmap:{vp8_pt} VP8/\d+\r\n", "", sdp)
            sdp = re.sub(rf"a=rtcp-fb:{vp8_pt} [^\r]*\r\n", "", sdp)
            sdp = re.sub(rf"a=fmtp:{vp8_pt}[^\r]*\r\n", "", sdp)
            # Remove VP8's RTX payload type (VP8 + next PT is RTX by aiortc)
            sdp = re.sub(rf"a=rtpmap:\d+ rtx/\d+\r\n.*?a=fmtp:\d+ apt={vp8_pt}\r\n",
                         "", sdp, flags=re.DOTALL)
            # Clean the m= line to only list H.264 payload types
            remaining_pts = re.findall(r"a=rtpmap:(\d+) (H264|rtx)/", sdp)
            pt_list = [pt for pt, _ in remaining_pts]
            if pt_list:
                sdp = re.sub(r"m=video \d+ UDP/TLS/RTP/SAVPF[^\r]*",
                             f"m=video 0 UDP/TLS/RTP/SAVPF " + " ".join(pt_list), sdp)
        offer.sdp = sdp

        await pc.setLocalDescription(offer)

        for _ in range(100):
            if pc.iceGatheringState == "complete":
                break
            await asyncio.sleep(0.05)

        # Store the PC so the answer endpoint can find it
        pc_id = uuid.uuid4().hex
        _pending_peer_connections[pc_id] = (pc, time.time())

        # Mark viewer connected
        with _viewers_lock:
            active_viewers += 1
            _cancel_pending_stop()

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "pc_id": pc_id,
        }

    try:
        future = asyncio.run_coroutine_threadsafe(_handle(), _webrtc_loop)
        result = future.result(timeout=10)
    except Exception as e:
        traceback.print_exc()
        return Response(json.dumps({"error": str(e)}), status=500, mimetype="application/json")

    return Response(
        json.dumps(result),
        mimetype="application/json",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@server.route("/webrtc/answer", methods=["POST"])
def webrtc_answer():
    """Receive the browser's SDP answer to complete signaling."""
    params = request.json
    answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc_id = params.get("pc_id", "")

    entry = _pending_peer_connections.pop(pc_id, None)
    pc = entry[0] if entry else None
    if pc is None:
        return Response(json.dumps({"error": "Unknown PC"}), status=404, mimetype="application/json")

    async def _handle():
        await pc.setRemoteDescription(answer)

    try:
        future = asyncio.run_coroutine_threadsafe(_handle(), _webrtc_loop)
        future.result(timeout=5)
    except Exception as e:
        traceback.print_exc()
        return Response(json.dumps({"error": str(e)}), status=500, mimetype="application/json")

    return Response(
        json.dumps({"ok": True}),
        mimetype="application/json",
    )


@server.route("/_healthz", methods=["GET"])
def healthz():
    """Liveness probe for the external watchdog.

    Returns 200 while Flask can serve requests at all; the state payload is
    for humans/forensics, not gating.
    """
    return Response(
        json.dumps({
            "status": "ok",
            "webrtc_player": _webrtc_player is not None,
            "webrtc_loop_alive": bool(_webrtc_loop is not None and _webrtc_loop.is_running()),
            "viewers": active_viewers,
            "active_peers": len(_active_peer_connections),
        }),
        mimetype="application/json",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dash App
# ──────────────────────────────────────────────────────────────────────────────

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
                                    children=html.Video(
                                        id="video",
                                        autoPlay=True,
                                        muted=True,
                                        controls=False,
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
                dcc.Interval(
                    id="graph-interval",
                    interval=GRAPH_INTERVAL_S * 1000,
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
        _teardown_webrtc()


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
    Input("graph-interval", "n_intervals"),
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
    """Handle SIGINT/SIGTERM: tear down WebRTC and exit cleanly."""
    try:
        _teardown_webrtc()
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
