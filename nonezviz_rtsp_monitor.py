#!/usr/bin/env python3
"""Human-detection monitor for Hikvision/Amtek cameras via direct RTSP.

Companion to ezviz_cloud_monitor.py, which covers EZVIZ cameras via the
cloud API. This script polls each camera's RTSP stream directly and
never depends on a third-party cloud service.

Camera list and connection details come from nonezviz_cameras_config.json.
Cameras with isapi_enabled=true are also monitored in real time via
motion push events (see nonezviz_isapi_motion_listener.py); they're
additionally re-polled every ISAPI_FALLBACK_EVERY_N_CYCLES cycles as a
safety net in case the push connection silently drops.

Run:
    python nonezviz_rtsp_monitor.py
"""

import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

# Must be set before cv2 is imported. Forces RTSP over TCP (avoids the
# packet loss / decoding artifacts UDP can cause on unstable radio/
# microwave links) and sets an explicit socket-level timeout, since
# OpenCV's CAP_PROP_*_TIMEOUT_MSEC is not honored by FFmpeg on some
# builds -- without this, a hung camera can block for FFmpeg's default
# 30s regardless of the timeout configured in capture_snapshot().
RTSP_TIMEOUT_SECONDS = 15
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    f"rtsp_transport;tcp|stimeout;{RTSP_TIMEOUT_SECONDS * 1_000_000}"
)

import cv2  # noqa: E402

from common_human_verifier_yolo import contains_human_shape, draw_boxes_on_image  # noqa: E402
from common_telegram_notifier import send_alert  # noqa: E402
from nonezviz_isapi_motion_listener import start_listeners  # noqa: E402

logger = logging.getLogger(__name__)

WIB = timezone(timedelta(hours=7))

CONFIG_FILE = "nonezviz_cameras_config.json"
STATE_FILE = "nonezviz_last_notified.json"
SNAPSHOT_DIR = "nonezviz_snapshots"

# --- Polling / anti-spam ---
CHECK_INTERVAL_SECONDS = 45
NOTIFY_COOLDOWN_SECONDS = 300

# Cameras on ISAPI push are skipped from normal polling, but are still
# re-polled every N cycles as a safety net in case the push connection
# silently fails. Set to 0 to disable this fallback.
ISAPI_FALLBACK_EVERY_N_CYCLES = 8  # ~6 min at the default 45s interval

# Cameras checked concurrently per cycle. Bounded by both CPU (one YOLO
# inference per camera) and per-site network bandwidth on remote links
# -- tune down if "failed to grab snapshot" counts rise after raising this.
MAX_PARALLEL_WORKERS = 20

# Snapshots are normally deleted right after each check completes; this
# retention is only a safety net for files left behind by a failed send
# or an unexpected error mid-cycle.
SNAPSHOT_RETENTION_DAYS = 1
CLEANUP_INTERVAL_SECONDS = 3600 * 6

# Visual verification thresholds. Per-camera overrides are supported via
# "min_confidence" / "min_box_height_ratio" / "min_aspect_ratio" in
# nonezviz_cameras_config.json, for cameras with atypical mounting
# (e.g. high/far angles where people always appear small in frame).
MIN_VERIFICATION_CONFIDENCE = 0.55
MIN_BOX_HEIGHT_RATIO = 0.04
MIN_ASPECT_RATIO = 0.8
INFERENCE_IMGSZ = 960

_state_lock = threading.Lock()

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Snapshot cleanup
# ---------------------------------------------------------------------------

def cleanup_old_snapshots():
    """Delete snapshots older than SNAPSHOT_RETENTION_DAYS."""
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
# Config & state
# ---------------------------------------------------------------------------

def load_config(path=CONFIG_FILE) -> dict:
    if not os.path.exists(path):
        logger.error("%s not found. Aborting.", path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state(path=STATE_FILE) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}


def save_state(state: dict, path=STATE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_active_cameras(config: dict):
    """Cameras eligible for RTSP polling: monitored, with an RTSP URL,
    and not handled instead via ISAPI push."""
    active, skipped = [], []
    for cam in config["cameras"]:
        if not cam.get("status_monitor", True):
            skipped.append((cam["name"], cam.get("note") or "status_monitor=false"))
            continue
        if not cam.get("rtsp_url"):
            skipped.append((cam["name"], cam.get("note") or "no rtsp_url"))
            continue
        if cam.get("isapi_enabled"):
            continue
        active.append(cam)
    return active, skipped


def get_isapi_cameras(config: dict):
    """Cameras opted into real-time ISAPI motion push."""
    return [
        cam for cam in config["cameras"]
        if cam.get("status_monitor", True) and cam.get("rtsp_url") and cam.get("isapi_enabled")
    ]


# ---------------------------------------------------------------------------
# RTSP capture
# ---------------------------------------------------------------------------

def capture_snapshot(rtsp_url: str, camera_name: str, timeout_seconds: int = RTSP_TIMEOUT_SECONDS) -> str:
    """Grab a single frame from an RTSP stream and save it to disk.

    Returns the saved file path, or None on failure.
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_seconds * 1000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_seconds * 1000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always read the freshest frame

    if not cap.isOpened():
        cap.release()
        return None

    # The first frame or two off a fresh RTSP connection is often
    # corrupt/black; read a few and keep the last good one.
    frame = None
    for _ in range(3):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()

    if frame is None:
        return None

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in camera_name)
    timestamp = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(SNAPSHOT_DIR, f"{timestamp}_{safe_name}.jpg")
    cv2.imwrite(output_path, frame)
    return output_path if os.path.exists(output_path) else None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def check_camera(camera: dict, state: dict):
    """Capture, verify, and alert for a single camera."""
    name = camera["name"]
    site = camera.get("site") or name
    cluster = camera.get("cluster", "-")
    serial = camera.get("serial_number") or ""
    ip = camera.get("ip_address") or ""
    source = (camera.get("source") or "hikvision").strip().upper()
    rtsp_url = camera["rtsp_url"]

    min_confidence = camera.get("min_confidence", MIN_VERIFICATION_CONFIDENCE)
    min_box_height_ratio = camera.get("min_box_height_ratio", MIN_BOX_HEIGHT_RATIO)
    min_aspect_ratio = camera.get("min_aspect_ratio", MIN_ASPECT_RATIO)
    ignore_zones = camera.get("ignore_zones")

    snapshot_path = capture_snapshot(rtsp_url, name)
    if not snapshot_path:
        logger.warning("%s: failed to grab snapshot (camera offline or RTSP unreachable).", name)
        return

    detected, confidence, count, boxes = contains_human_shape(
        snapshot_path,
        min_confidence,
        min_box_height_ratio,
        min_aspect_ratio=min_aspect_ratio,
        imgsz=INFERENCE_IMGSZ,
        **({"ignore_zones": ignore_zones} if ignore_zones else {}),
    )
    logger.debug("%s: detected=%s confidence=%.2f count=%d", name, detected, confidence, count)

    if not detected:
        _safe_remove(snapshot_path)
        return

    # Cooldown check must be atomic across threads, since two cameras
    # can finish concurrently right at the cycle boundary.
    now = time.time()
    with _state_lock:
        last_notified = state.get(name, 0)
        in_cooldown = (now - last_notified) < NOTIFY_COOLDOWN_SECONDS

    if in_cooldown:
        remaining = int(NOTIFY_COOLDOWN_SECONDS - (now - last_notified))
        logger.info("%s: detection suppressed by cooldown (%ds remaining).", name, remaining)
        _safe_remove(snapshot_path)
        return

    annotated_path = snapshot_path.rsplit(".", 1)[0] + "_annotated.jpg"
    final_photo_path = snapshot_path
    if draw_boxes_on_image(snapshot_path, boxes, annotated_path):
        final_photo_path = annotated_path

    device_display = f"{name} (SN: {serial})" if serial else name
    ok = send_alert(
        site_name=site,
        device_name=device_display,
        timestamp=datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S"),
        photo_path=final_photo_path,
        source=source,
        cluster=cluster,
        ip=ip,
    )
    logger.info("%s: alert %s", name, "sent" if ok else "FAILED")

    if ok:
        with _state_lock:
            state[name] = now
        _safe_remove(snapshot_path)
        _safe_remove(annotated_path)
    else:
        # Leave the photo on disk for later retry/inspection; the
        # retention cleanup will eventually remove it.
        logger.warning("%s: snapshot kept on disk after failed send.", name)


def _safe_remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def run_cycle(config: dict, state: dict, cycle_number: int = 0, isapi_cameras: list = None):
    active_cameras, skipped = get_active_cameras(config)

    fallback_cameras = []
    if isapi_cameras and ISAPI_FALLBACK_EVERY_N_CYCLES > 0 and cycle_number % ISAPI_FALLBACK_EVERY_N_CYCLES == 0:
        fallback_cameras = isapi_cameras
        logger.info("ISAPI fallback poll: %s", ", ".join(c["name"] for c in fallback_cameras))

    cameras_to_check = active_cameras + fallback_cameras
    logger.info("Cycle start: %d camera(s), %d worker(s)%s",
                len(cameras_to_check), MAX_PARALLEL_WORKERS,
                f", {len(skipped)} skipped" if skipped else "")

    cycle_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = {executor.submit(check_camera, camera, state): camera for camera in cameras_to_check}
        for future in as_completed(futures):
            camera = futures[future]
            try:
                future.result()
            except Exception:
                logger.exception("Unexpected error checking %s", camera.get("name", "?"))

    elapsed = time.time() - cycle_start
    logger.info("Cycle done in %.1fs (%d cameras).", elapsed, len(cameras_to_check))
    save_state(state)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Starting non-EZVIZ RTSP monitor.")
    config = load_config()
    state = load_state()
    cleanup_old_snapshots()
    last_cleanup = time.time()

    active_cameras, skipped = get_active_cameras(config)
    isapi_cameras = get_isapi_cameras(config)

    logger.info("%d camera(s) polled via RTSP every %ds (%d worker(s)).",
                len(active_cameras), CHECK_INTERVAL_SECONDS, MAX_PARALLEL_WORKERS)
    if isapi_cameras:
        logger.info("%d camera(s) monitored via ISAPI motion push: %s",
                    len(isapi_cameras), ", ".join(c["name"] for c in isapi_cameras))
    for cam_name, reason in skipped:
        logger.info("Skipping %s: %s", cam_name, reason)

    if active_cameras:
        batches = math.ceil(len(active_cameras) / MAX_PARALLEL_WORKERS)
        logger.info("~%d wave(s) per cycle to cover all polled cameras.", batches)

    def on_motion(camera):
        check_camera(camera, state)
        save_state(state)

    start_listeners(isapi_cameras, on_motion, log_fn=logger.info)

    cycle_number = 0
    while True:
        try:
            run_cycle(config, state, cycle_number=cycle_number, isapi_cameras=isapi_cameras)
        except Exception:
            logger.exception("Unexpected error during cycle.")
        cycle_number += 1

        if time.time() - last_cleanup > CLEANUP_INTERVAL_SECONDS:
            cleanup_old_snapshots()
            last_cleanup = time.time()

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user (Ctrl+C).")
        sys.exit(0)
