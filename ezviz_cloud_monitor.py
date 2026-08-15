#!/usr/bin/env python3
"""Human-detection monitor for EZVIZ cameras via the EZVIZ cloud API.

Companion to nonezviz_rtsp_monitor.py, which covers Hikvision/Amtek
cameras via direct RTSP. This script polls the EZVIZ cloud for alarm
messages, downloads the associated snapshot, and runs it through YOLO
verification before alerting.

The camera list, Site, Cluster and IP all come from
site_mapping_ezviz.json (see site_mapper_ezviz.py) -- edit that file
and restart to apply changes.

Credentials (EZVIZ_USERNAME / EZVIZ_PASSWORD, and the shared
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in common_telegram_notifier.py)
are read from environment variables, falling back to the defaults below.

Run:
    python ezviz_cloud_monitor.py
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

from common_human_verifier_yolo import contains_human_shape, draw_boxes_on_image
from common_telegram_notifier import send_alert
from site_mapper_ezviz import (
    get_data_source,
    get_detection_overrides,
    get_ip,
    get_location,
    get_repeater_code,
    get_site,
    is_monitored,
)

logger = logging.getLogger(__name__)

# --- EZVIZ account ---
EZVIZ_USERNAME = os.environ.get("EZVIZ_USERNAME", "Cctv.noc.ptt@gmail.com")
EZVIZ_PASSWORD = os.environ.get("EZVIZ_PASSWORD", "Punyaptt@2023")
EZVIZ_REGION = os.environ.get("EZVIZ_REGION", "apiisgp.ezvizlife.com")
PYEZVIZAPI_BIN = os.environ.get("PYEZVIZAPI_BIN", "/home/ptt/human_detection/venv/bin/pyezvizapi")
TOKEN_FILE = "ezviz_token.json"
TOKEN_REFRESH_INTERVAL_SECONDS = 3600 * 4

# --- Polling ---
POLL_INTERVAL_SECONDS = 60
MSG_LIMIT_PER_POLL = 50
ALLOWED_SUBTYPES = {2401}  # human detection alarm

# --- Anti-spam ---
# A camera that keeps re-triggering gets escalated to a longer cooldown
# after ESCALATION_THRESHOLD consecutive suppressed alerts.
NOTIFY_COOLDOWN_SECONDS = 300
ESCALATION_THRESHOLD = 3
NOISY_COOLDOWN_SECONDS = 1800

# --- Visual verification ---
REQUIRE_VISUAL_VERIFICATION = True
MIN_VERIFICATION_CONFIDENCE = 0.55

# --- Snapshots ---
SNAPSHOT_DIR = "ezviz_snapshots"
SNAPSHOT_RETENTION_DAYS = 1  # safety net for photos that failed to send
CLEANUP_INTERVAL_SECONDS = 3600 * 6

# --- Camera filter ---
# "whitelist": only cameras listed with status_monitor=true are processed.
# "monitor_all": cameras missing from site_mapping_ezviz.json are too.
FILTER_MODE = "whitelist"

# Some Hikvision cameras are registered on the EZVIZ account; their
# serials are longer and/or contain "DS" (e.g. "DS-2CD2xxx").
HIKVISION_SERIAL_MIN_LENGTH = 12

WIT_TO_WIB_OFFSET_HOURS = 2  # camera OSD clocks use WIT (UTC+9); alerts use WIB (UTC+7)
_TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"]

STATE_FILES = {
    "seen_msgids": "ezviz_seen_msgids.json",
    "last_notified": "ezviz_last_notified.json",
    "noisy_streak": "ezviz_noisy_streak.json",
}

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def should_monitor(device_name: str, serial: str = None) -> bool:
    default_if_unlisted = FILTER_MODE == "monitor_all"
    return is_monitored(device_name, serial=serial, default=default_if_unlisted)


def detect_camera_brand(serial: str) -> str:
    """Guess camera brand from the serial number pattern."""
    if not serial or serial == "?":
        return "EZVIZ"
    if "DS" in serial.upper() or len(serial) >= HIKVISION_SERIAL_MIN_LENGTH:
        return "HIKVISION"
    return "EZVIZ"


def convert_wit_to_wib(time_str: str) -> str:
    """Convert a WIT timestamp to WIB (-2 hours), same format.

    Returns the input unchanged if it doesn't match a known format.
    """
    if not time_str or time_str == "?":
        return time_str

    for fmt in _TIME_FORMATS:
        try:
            dt = datetime.strptime(time_str, fmt)
        except ValueError:
            continue
        return (dt - timedelta(hours=WIT_TO_WIB_OFFSET_HOURS)).strftime(fmt)

    logger.warning("Unrecognized timestamp format %r, WIT->WIB conversion skipped.", time_str)
    return time_str


def cleanup_old_snapshots():
    cutoff = time.time() - (SNAPSHOT_RETENTION_DAYS * 86400)
    deleted, freed_bytes = 0, 0
    try:
        for fname in os.listdir(SNAPSHOT_DIR):
            fpath = os.path.join(SNAPSHOT_DIR, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                freed_bytes += os.path.getsize(fpath)
                os.remove(fpath)
                deleted += 1
        if deleted:
            logger.info("Cleanup: removed %d old snapshot(s), freed %.1f MB.",
                        deleted, freed_bytes / 1024 / 1024)
    except OSError:
        logger.exception("Snapshot cleanup failed.")


# ---------------------------------------------------------------------------
# Persistent state (seen message IDs, last-notified times, noisy streaks)
# ---------------------------------------------------------------------------

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f)


def load_state() -> dict:
    return {
        "seen_msgids": set(load_json(STATE_FILES["seen_msgids"], [])),
        "last_notified": load_json(STATE_FILES["last_notified"], {}),
        "noisy_streak": load_json(STATE_FILES["noisy_streak"], {}),
    }


def save_state(state: dict, keys=("seen_msgids", "last_notified", "noisy_streak")):
    for key in keys:
        value = state[key]
        save_json(STATE_FILES[key], list(value)[-2000:] if key == "seen_msgids" else value)


# ---------------------------------------------------------------------------
# EZVIZ cloud API (via the pyezvizapi CLI)
# ---------------------------------------------------------------------------

def refresh_token():
    cmd = [
        PYEZVIZAPI_BIN, "-u", EZVIZ_USERNAME, "-p", EZVIZ_PASSWORD, "-r", EZVIZ_REGION,
        "--token-file", TOKEN_FILE, "--save-token",
        "--json", "devices", "status",
    ]
    logger.info("Refreshing EZVIZ token...")
    subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def get_recent_alarms(limit: int) -> list:
    cmd = [PYEZVIZAPI_BIN, "--token-file", TOKEN_FILE, "--json", "unifiedmsg", "--limit", str(limit)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.exception("Failed to fetch unifiedmsg.")
        return []


def download_snapshot(serial: str, image_url: str, msg_id: str, device_name: str = "", time_str: str = "") -> str:
    safe_time = re.sub(r"[^0-9]", "", time_str) if time_str else str(int(time.time()))
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", device_name).strip("_") or "unknown"
    safe_msgid = re.sub(r"[^A-Za-z0-9_-]", "_", msg_id)
    output_path = os.path.join(SNAPSHOT_DIR, f"{safe_time}_{safe_name}_{serial}_{safe_msgid}.jpg")

    cmd = [
        PYEZVIZAPI_BIN, "--token-file", TOKEN_FILE,
        "save", "image", "--serial", serial,
        "--image-url", image_url, "--output", output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if os.path.exists(output_path):
            return output_path
        logger.error("Snapshot download failed for %s (%s): %s", device_name, serial, result.stderr.strip())
    except subprocess.TimeoutExpired:
        logger.error("Snapshot download timed out for %s (%s).", device_name, serial)
    except Exception:
        logger.exception("Unexpected error downloading snapshot for %s (%s).", device_name, serial)
    return None


# ---------------------------------------------------------------------------
# Alarm processing
# ---------------------------------------------------------------------------

def process_alarm(msg: dict, last_notified: dict, noisy_streak: dict) -> str:
    """Returns one of: 'sent', 'skipped_cooldown', 'skipped_verification',
    'skipped_no_snapshot'. Assumes msg['subType'] has already been checked."""
    device_name = msg.get("from", "?").strip()
    serial = msg.get("deviceSerial", "?")
    time_str = convert_wit_to_wib(msg.get("timeStr", "?"))
    pic_url = msg.get("pic", "")

    now = time.time()
    last_time = last_notified.get(serial, 0)
    streak = noisy_streak.get(serial, 0)
    cooldown = NOISY_COOLDOWN_SECONDS if streak >= ESCALATION_THRESHOLD else NOTIFY_COOLDOWN_SECONDS

    if now - last_time < cooldown:
        noisy_streak[serial] = streak + 1
        logger.info("%s (%s): suppressed by cooldown (streak=%d).", device_name, serial, noisy_streak[serial])
        return "skipped_cooldown"

    noisy_streak[serial] = 0
    logger.info("%s (%s): new alarm @ %s.", device_name, serial, time_str)

    snapshot_path = None
    if pic_url:
        msg_id = msg.get("msgId", str(time.time()))
        snapshot_path = download_snapshot(serial, pic_url, msg_id, device_name, time_str)
        if not snapshot_path:
            time.sleep(2)  # brief retry for transient token/network hiccups
            snapshot_path = download_snapshot(serial, pic_url, msg_id, device_name, time_str)

    boxes = []
    if REQUIRE_VISUAL_VERIFICATION:
        if not snapshot_path:
            logger.info("%s (%s): no snapshot available, alert suppressed.", device_name, serial)
            return "skipped_no_snapshot"

        overrides = get_detection_overrides(device_name, serial=serial)
        min_confidence = overrides.pop("min_confidence", MIN_VERIFICATION_CONFIDENCE)
        detected, confidence, count, boxes = contains_human_shape(snapshot_path, min_confidence, **overrides)
        logger.info("%s (%s): verification detected=%s confidence=%.2f count=%d",
                    device_name, serial, detected, confidence, count)
        if not detected:
            if os.path.exists(snapshot_path):
                try:
                    os.remove(snapshot_path)
                    logger.info("%s (%s): no human detected, snapshot removed.", device_name, serial)
                except OSError:
                    logger.warning("Could not remove non-human snapshot: %s", snapshot_path)
            return "skipped_verification"

        annotated_path = snapshot_path.rsplit(".", 1)[0] + "_annotated.jpg"
        if draw_boxes_on_image(snapshot_path, boxes, annotated_path):
            snapshot_path = annotated_path

    # Site/Cluster/IP lookups use the raw device_name (not the SN-suffixed
    # display name below) and fall back to serial when the cloud's device
    # name doesn't match what's in site_mapping_ezviz.json.
    cluster = get_location(device_name, serial=serial, default=None)
    site_name = get_site(device_name, serial=serial, default=None)
    if not site_name:
        repeater_code = get_repeater_code(device_name)
        site_name = f"Repeater {repeater_code}" if repeater_code else (cluster or device_name)
    ip_address = get_ip(device_name, serial=serial, default="-")

    device_display = device_name
    if serial and serial != "?" and serial not in device_name:
        device_display = f"{device_name} (SN: {serial})"

    ok = send_alert(
        site_name=site_name,
        device_name=device_display,
        timestamp=time_str,
        photo_path=snapshot_path,
        source=detect_camera_brand(serial),
        cluster=cluster,
        ip=ip_address,
    )
    logger.info("%s: alert %s", device_name, "sent" if ok else "FAILED")

    if ok:
        last_notified[serial] = now
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                os.remove(snapshot_path)
            except OSError:
                logger.warning("Could not remove sent snapshot: %s", snapshot_path)

    return "sent"


def poll_once(state: dict):
    seen_msgids, last_notified, noisy_streak = state["seen_msgids"], state["last_notified"], state["noisy_streak"]
    counts = {"sent": 0, "skipped_cooldown": 0, "skipped_verification": 0,
              "skipped_no_snapshot": 0, "skipped_not_monitored": 0}

    for msg in reversed(get_recent_alarms(MSG_LIMIT_PER_POLL)):
        msg_id = msg.get("msgId")
        if not msg_id or msg_id in seen_msgids:
            continue
        seen_msgids.add(msg_id)

        if msg.get("subType") not in ALLOWED_SUBTYPES:
            continue

        device_name = msg.get("from", "?").strip()
        serial = msg.get("deviceSerial", "?")
        if not should_monitor(device_name, serial=serial):
            counts["skipped_not_monitored"] += 1
            continue

        result = process_alarm(msg, last_notified, noisy_streak)
        counts[result] = counts.get(result, 0) + 1

    save_state(state)

    logger.info(
        "Poll summary: sent=%d cooldown=%d not_human=%d no_snapshot=%d not_monitored=%d",
        counts["sent"], counts["skipped_cooldown"], counts["skipped_verification"],
        counts["skipped_no_snapshot"], counts["skipped_not_monitored"],
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Starting EZVIZ cloud monitor.")
    source = get_data_source()
    logger.info("Site mapping source: %s", os.path.basename(source) if source else "NOT FOUND")

    refresh_token()
    last_token_refresh = time.time()
    last_cleanup = time.time()
    cleanup_old_snapshots()

    state = load_state()
    logger.info("Loaded %d previously seen message(s).", len(state["seen_msgids"]))

    while True:
        try:
            if time.time() - last_token_refresh > TOKEN_REFRESH_INTERVAL_SECONDS:
                refresh_token()
                last_token_refresh = time.time()

            if time.time() - last_cleanup > CLEANUP_INTERVAL_SECONDS:
                cleanup_old_snapshots()
                last_cleanup = time.time()

            poll_once(state)

        except Exception:
            logger.exception("Unexpected error during poll.")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C).")
        sys.exit(0)
