"""Real-time motion event listener for Hikvision/Amtek cameras (ISAPI).

Used by nonezviz_rtsp_monitor.py. Opens a long-lived HTTP connection to
each camera's ISAPI alertStream endpoint and pushes a callback the
moment motion (VMD) is detected, instead of waiting for the next
polling cycle. Not used for EZVIZ cameras (see ezviz_cloud_monitor.py).

Enable per camera in nonezviz_cameras_config.json:
    "isapi_enabled": true
    "isapi_port": 80   # optional, defaults to 80

Credentials are reused from the camera's existing "username"/"password"
fields (the same ones used for RTSP).
"""

import logging
import re
import threading
import time

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

logger = logging.getLogger(__name__)

# VMD (Video Motion Detection) is the only event type currently
# handled. Add more here (e.g. "fielddetection", "linedetection") if a
# camera exposes other relevant smart-detection events.
TRIGGER_EVENT_TYPES = {"VMD"}

# Cameras resend "active" every ~1s while motion continues, so trigger
# callbacks are debounced to avoid re-capturing on every event.
DEBOUNCE_SECONDS = 10
RECONNECT_DELAY_SECONDS = 15


class IsapiMotionListener:
    """Motion listener for a single camera, running in its own thread."""

    def __init__(self, camera: dict, on_motion_callback, log_fn=None):
        self.camera = camera
        self.name = camera.get("name", "?")
        self.ip = camera.get("ip_address") or camera.get("ip")
        self.port = camera.get("isapi_port", 80)
        self.username = camera.get("username") or camera.get("rtsp_username")
        self.password = camera.get("password") or camera.get("rtsp_password")
        self.on_motion_callback = on_motion_callback
        self._log = log_fn or logger.info

        self._stop_event = threading.Event()
        self._thread = None
        self._last_trigger_time = 0.0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"isapi-{self.name}")
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        url = f"http://{self.ip}:{self.port}/ISAPI/Event/notification/alertStream"
        auth_methods = [
            ("Digest", HTTPDigestAuth(self.username, self.password)),
            ("Basic", HTTPBasicAuth(self.username, self.password)),
        ]

        while not self._stop_event.is_set():
            connected = False
            for auth_name, auth in auth_methods:
                if self._stop_event.is_set():
                    return
                try:
                    self._log(f"[ISAPI:{self.name}] Connecting ({auth_name})...")
                    resp = requests.get(url, auth=auth, stream=True, timeout=30)
                    if resp.status_code == 200:
                        self._log(f"[ISAPI:{self.name}] Connected, listening for motion events.")
                        connected = True
                        self._listen(resp)
                        break
                    if resp.status_code == 401:
                        continue
                    self._log(f"[ISAPI:{self.name}] Unexpected status: {resp.status_code}")
                except requests.exceptions.RequestException as e:
                    self._log(f"[ISAPI:{self.name}] Connection error: {e}")

            if self._stop_event.is_set():
                return

            reason = "connection lost" if connected else "all auth methods failed"
            self._log(f"[ISAPI:{self.name}] {reason}, retrying in {RECONNECT_DELAY_SECONDS}s...")
            self._stop_event.wait(RECONNECT_DELAY_SECONDS)

    def _listen(self, resp):
        buffer = ""
        try:
            for chunk in resp.iter_content(chunk_size=1024, decode_unicode=True):
                if self._stop_event.is_set():
                    return
                if not chunk:
                    continue
                buffer += chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")

                while "<EventNotificationAlert" in buffer and "</EventNotificationAlert>" in buffer:
                    start = buffer.index("<EventNotificationAlert")
                    end = buffer.index("</EventNotificationAlert>") + len("</EventNotificationAlert>")
                    self._handle_event(buffer[start:end])
                    buffer = buffer[end:]
        except requests.exceptions.RequestException as e:
            self._log(f"[ISAPI:{self.name}] Stream error: {e}")

    def _handle_event(self, event_xml: str):
        event_type = self._extract_tag(event_xml, "eventType")
        event_state = self._extract_tag(event_xml, "eventState")

        if event_type not in TRIGGER_EVENT_TYPES or event_state != "active":
            return

        now = time.time()
        if now - self._last_trigger_time < DEBOUNCE_SECONDS:
            return

        self._last_trigger_time = now
        self._log(f"[ISAPI:{self.name}] Motion detected ({event_type}), triggering check.")
        try:
            self.on_motion_callback(self.camera)
        except Exception:
            logger.exception("[ISAPI:%s] Error in motion callback", self.name)

    @staticmethod
    def _extract_tag(xml: str, tag: str) -> str:
        match = re.search(f"<{tag}>(.*?)</{tag}>", xml)
        return match.group(1) if match else None


def start_listeners(cameras: list, on_motion_callback, log_fn=None) -> list:
    """Start a listener thread for every camera with isapi_enabled=True.

    Returns the list of running IsapiMotionListener instances.
    """
    listeners = []
    for camera in cameras:
        if not camera.get("isapi_enabled"):
            continue
        listener = IsapiMotionListener(camera, on_motion_callback, log_fn=log_fn)
        listener.start()
        listeners.append(listener)
    return listeners
