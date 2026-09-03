import os
import logging
try:
    import requests
except ImportError:  # Telegram is an optional alerting integration.
    requests = None

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
session = requests.Session() if requests is not None else None
logger = logging.getLogger("ws-monitor.telegram")

def send_message(message: str, verification: bool = True) -> bool:
    """Send a message via Telegram bot."""
    if session is None or not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        resp = session.post(
            f"{BASE_URL}/sendMessage",
            params={"chat_id": CHAT_ID, "text": message},
            verify=verification,
            timeout=float(os.getenv("TELEGRAM_REQUEST_TIMEOUT_SECONDS", "10")),
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Telegram delivery failed: %s", e)
        return False
