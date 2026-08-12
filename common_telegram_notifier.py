"""Telegram alert notifier.

Shared by ezviz_cloud_monitor.py and nonezviz_rtsp_monitor.py so both
pipelines send alerts in the same format from a single place.

Credentials are read from environment variables only -- no defaults
are baked into source control. Set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
(e.g. via a .env file, see .env.example) before starting the program.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT_SECONDS = 20

_ALERT_TEMPLATE = (
    "🚨 HUMAN DETECTION ALERT 🚨\n\n"
    "Site      : {site_name}\n"
    "Device    : {device_name}\n"
    "IP        : {ip}\n"
    "Cluster   : {cluster}\n"
    "Waktu     : {timestamp}\n"
    "Source    : {source}\n\n"
    "--Notification CCTV PTT Network--"
)


def send_alert(
    site_name: str,
    device_name: str,
    timestamp: str,
    photo_path: str = None,
    source: str = "EZVIZ",
    cluster: str = None,
    ip: str = None,
) -> bool:
    """Send a human-detection alert to Telegram.

    Sends a photo (with caption) when photo_path points to an existing
    file, otherwise sends a text-only message.

    Returns:
        True if Telegram accepted the message, False otherwise.
    """
    if not BOT_TOKEN or not CHAT_ID:
        logger.error(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID is not set; cannot send "
            "alert. Set both as environment variables (see .env.example)."
        )
        return False

    caption = _ALERT_TEMPLATE.format(
        site_name=site_name,
        device_name=device_name,
        ip=ip or "-",
        cluster=cluster or "-",
        timestamp=timestamp,
        source=source,
    )

    try:
        if photo_path and os.path.exists(photo_path):
            url = f"{API_BASE}/bot{BOT_TOKEN}/sendPhoto"
            with open(photo_path, "rb") as photo_file:
                resp = requests.post(
                    url,
                    data={"chat_id": CHAT_ID, "caption": caption},
                    files={"photo": photo_file},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
        else:
            if photo_path:
                logger.warning("Photo not found (%s), sending text-only alert.", photo_path)
            url = f"{API_BASE}/bot{BOT_TOKEN}/sendMessage"
            resp = requests.post(
                url,
                data={"chat_id": CHAT_ID, "text": caption},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        if resp.status_code == 200 and resp.json().get("ok"):
            return True

        logger.error("Telegram API error (HTTP %s): %s", resp.status_code, resp.text)
        return False

    except requests.exceptions.RequestException:
        logger.exception("Failed to reach Telegram API.")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ok = send_alert(
        site_name="TEST",
        device_name="Test Device",
        timestamp="2026-08-11 00:00:00",
        source="TEST",
        cluster="TEST",
        ip="10.0.0.1",
    )
    print("Test alert sent." if ok else "Test alert failed, see log above.")
