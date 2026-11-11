import asyncio
import contextlib
import importlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(
    all(importlib.util.find_spec(package) for package in ("aiortc", "dash", "dash_mantine_components")),
    "dashboard dependencies are required",
)
class WebRTCErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        database = str(Path(cls.directory.name) / "samples.db")
        with patch.dict(os.environ, {"WSM_DB_PATH": database}):
            cls.dashboard = importlib.import_module("dashwebapp")
        cls.client = cls.dashboard.server.test_client()

    def setUp(self):
        from av.error import FileNotFoundError as CameraOpenError

        self.camera_url = "rtsp://test-user:test-password@example.invalid/ch0"
        self.error = CameraOpenError(2, "Unable to open camera", self.camera_url)

    def test_camera_startup_error_is_safe_in_response_and_log(self):
        log = io.StringIO()
        with patch.object(self.dashboard, "CAMERA_ENABLED", True):
            with patch.object(self.dashboard, "_ensure_webrtc_running", side_effect=self.error):
                with contextlib.redirect_stdout(log):
                    response = self.client.get("/webrtc/offer")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_data(as_text=True), "Camera connection failed")
        self.assertIn("FileNotFoundError", log.getvalue())
        self.assertNotIn(self.camera_url, log.getvalue())

    def test_camera_restart_error_logs_type_only(self):
        log = io.StringIO()
        with patch.multiple(self.dashboard, _webrtc_player=None, _webrtc_relay=None):
            with patch.object(self.dashboard, "MediaPlayer", side_effect=self.error):
                with contextlib.redirect_stdout(log):
                    asyncio.run(self.dashboard._restart_player())
        self.assertIn("FileNotFoundError", log.getvalue())
        self.assertNotIn(self.camera_url, log.getvalue())

    def test_webrtc_answer_failure_is_safe_in_response_and_log(self):
        def fail_signaling(coroutine, _loop):
            coroutine.close()
            raise RuntimeError(self.camera_url)

        log = io.StringIO()
        with patch.dict(self.dashboard._pending_peer_connections, {"test-peer": (object(), 0)}):
            with patch.object(self.dashboard.asyncio, "run_coroutine_threadsafe", side_effect=fail_signaling):
                with contextlib.redirect_stdout(log):
                    response = self.client.post("/webrtc/answer", json={
                        "sdp": "v=0", "type": "answer", "pc_id": "test-peer",
                    })
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {"error": "WebRTC answer failed"})
        self.assertIn("RuntimeError", log.getvalue())
        self.assertNotIn(self.camera_url, log.getvalue())

    def test_webrtc_answer_rejects_malformed_input(self):
        response = self.client.post("/webrtc/answer", json={"type": self.camera_url})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "Invalid WebRTC answer"})
