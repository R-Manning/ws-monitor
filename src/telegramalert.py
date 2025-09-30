import os
import requests

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "my_token")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "my_chat")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
session = requests.Session()

def send_message(message: str, verification: bool = True) -> bool:
    """Send a message via Telegram bot."""
    try:
        resp = session.post(
            f"{BASE_URL}/sendMessage",
            params={"chat_id": CHAT_ID, "text": message},
            verify=verification
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error sending message: {e}")
        return False
