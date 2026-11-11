import logging
import os

try:
    import requests
except ImportError:  # Telegram is an optional alerting integration.
    requests = None

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
session = requests.Session() if requests is not None else None
logger = logging.getLogger("ws-monitor.telegram")

# urllib3 debug records can contain the request path, including the bot token.
logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)
logging.getLogger("urllib3.util.retry").setLevel(logging.INFO)


def send_message(message: str, verification: bool = True) -> bool:
    """Send a message via Telegram bot."""
    if session is None or not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        resp = session.post(
            f"{BASE_URL}/sendMessage",
            json={"chat_id": CHAT_ID, "text": message},
            verify=verification,
            timeout=float(os.getenv("TELEGRAM_REQUEST_TIMEOUT_SECONDS", "10")),
        )
        resp.raise_for_status()
        return True
    except requests.HTTPError as exc:
        if exc.response is not None:
            logger.warning("Telegram delivery failed (HTTP %d)", exc.response.status_code)
        else:
            logger.warning("Telegram delivery failed (HTTP error)")
        return False
    except Exception as exc:
        logger.warning("Telegram delivery failed (%s)", type(exc).__name__)
        return False
