import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import telegramalert


class TelegramAlertTests(unittest.TestCase):
    def setUp(self):
        self.token = "test-token"
        self.chat = "test-chat"
        self.message = "test-alert"
        self.url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        for key, value in (("BOT_TOKEN", self.token), ("CHAT_ID", self.chat), ("BASE_URL", self.url.rsplit("/", 1)[0])):
            patcher = patch.object(telegramalert, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_http_error_logs_only_status_and_sends_json_body(self):
        response = requests.Response()
        response.status_code = 429
        response.reason = "Too Many Requests"
        response.url = f"{self.url}?chat_id={self.chat}&text={self.message}"
        with patch.object(telegramalert.session, "post", return_value=response) as post:
            with self.assertLogs("ws-monitor.telegram", level="WARNING") as captured:
                self.assertFalse(telegramalert.send_message(self.message))

        self.assertEqual(captured.output, ["WARNING:ws-monitor.telegram:Telegram delivery failed (HTTP 429)"])
        self.assertEqual(post.call_args.args, (self.url,))
        self.assertEqual(post.call_args.kwargs["json"], {"chat_id": self.chat, "text": self.message})
        self.assertNotIn("params", post.call_args.kwargs)

    def test_connection_failure_logs_exception_type_only(self):
        with patch.object(telegramalert.session, "post", side_effect=requests.ConnectionError(self.url)):
            with self.assertLogs("ws-monitor.telegram", level="WARNING") as captured:
                self.assertFalse(telegramalert.send_message(self.message))
        self.assertEqual(captured.output, ["WARNING:ws-monitor.telegram:Telegram delivery failed (ConnectionError)"])

    def test_success_returns_true(self):
        response = requests.Response()
        response.status_code = 200
        with patch.object(telegramalert.session, "post", return_value=response):
            self.assertTrue(telegramalert.send_message(self.message))

    def test_http_library_debug_logging_is_disabled(self):
        root_logger = logging.getLogger()
        self.addCleanup(root_logger.setLevel, root_logger.level)
        root_logger.setLevel(logging.DEBUG)
        for name in ("urllib3.connectionpool", "urllib3.util.retry"):
            self.assertFalse(logging.getLogger(name).isEnabledFor(logging.DEBUG))
