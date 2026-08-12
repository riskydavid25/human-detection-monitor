"""Site/cluster lookup for EZVIZ cameras, backed by site_mapping_ezviz.json.

This is the single source of truth for which EZVIZ cameras are
monitored and how they map to Site/Cluster/IP. Edit the JSON and
restart the program to apply changes.

Camera lookups are normalized (case/spacing/BAWAH-BWH/ATAS-ATS
tolerant) and matched by device name first, falling back to serial
number, since EZVIZ cloud alarms sometimes report a device name that
differs from the one in the JSON.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_mapping_ezviz.json")
REQUIRED_KEYS = ["name device", "site", "cluster", "serial_number", "ip_address", "status_monitor"]
OVERRIDE_FIELDS = ("ignore_zones", "min_confidence", "min_box_height_ratio", "min_aspect_ratio", "imgsz")

_REPEATER_CODE_PATTERN = re.compile(r"^([A-Za-z]{1,2}\d{1,3})[\s\-_]")

_cache = None  # normalized name/serial -> entry dict


def get_repeater_code(camera_name: str) -> str:
    """Extract a repeater code from a device name, e.g. 'B14-BWH' -> 'B14'."""
    if not camera_name:
        return None
    match = _REPEATER_CODE_PATTERN.match(camera_name.strip())
    return match.group(1).upper() if match else None


def _normalize(name) -> str:
    """Normalize for tolerant matching: case, spacing, and the
    BAWAH/BWH, ATAS/ATS abbreviations used inconsistently between the
    JSON and EZVIZ device names."""
    if not name:
        return ""
    s = str(name).upper().replace("BAWAH", "BWH").replace("ATAS", "ATS")
    return re.sub(r"[^A-Z0-9]", "", s)


def _load_mapping() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    mapping = {}

    if not os.path.exists(JSON_PATH):
        logger.warning("%s not found; camera mapping will be empty.", JSON_PATH)
        _cache = mapping
        return mapping

    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        logger.exception("Failed to read/parse %s", JSON_PATH)
        _cache = mapping
        return mapping

    if not isinstance(records, list):
        logger.error("%s must contain a JSON list, got %s", JSON_PATH, type(records).__name__)
        _cache = mapping
        return mapping

    loaded, skipped = 0, 0
    for row in records:
        if not isinstance(row, dict) or any(k not in row for k in REQUIRED_KEYS) or not row.get("name device"):
            skipped += 1
            continue

        entry = {
            "name": str(row["name device"]).strip(),
            "serial": str(row.get("serial_number") or "").strip(),
            "site": str(row.get("site") or "").strip(),
            "cluster": str(row.get("cluster") or "").strip(),
            "ip": str(row.get("ip_address") or "").strip(),
            "monitor": bool(row.get("status_monitor")),
        }
        for field in OVERRIDE_FIELDS:
            entry[field] = row.get(field)

        mapping[_normalize(entry["name"])] = entry
        if entry["serial"] and entry["serial"].upper() != "N/A":
            mapping[_normalize(entry["serial"])] = entry
        loaded += 1

    _cache = mapping
    logger.info(
        "Loaded %d cameras from %s%s", loaded, os.path.basename(JSON_PATH),
        f" ({skipped} rows skipped, incomplete data)" if skipped else "",
    )
    return mapping


def _find_entry(camera_name: str, serial: str = None) -> dict:
    """Find a camera's entry by name, then serial, then partial name match."""
    mapping = _load_mapping()
    key = _normalize(camera_name)

    if key in mapping:
        return mapping[key]
    if serial and _normalize(serial) in mapping:
        return mapping[_normalize(serial)]
    for mapped_key, entry in mapping.items():
        if key and (key in mapped_key or mapped_key in key):
            return entry
    return None


def get_location(camera_name: str, serial: str = None, default: str = None) -> str:
    """Return the Cluster for a camera.

    Note: despite the name, this returns Cluster, not Site -- use
    get_site() for the Site field. Kept as-is to avoid breaking
    existing callers.
    """
    entry = _find_entry(camera_name, serial)
    fallback = default if default is not None else camera_name
    return (entry["cluster"] if entry else "") or fallback


def get_site(camera_name: str, serial: str = None, default: str = None) -> str:
    """Return the Site for a camera, e.g. 'KIGAMANI'."""
    entry = _find_entry(camera_name, serial)
    fallback = default if default is not None else camera_name
    return (entry["site"] if entry else "") or fallback


def get_ip(camera_name: str, serial: str = None, default: str = "-") -> str:
    """Return the IP address for a camera."""
    entry = _find_entry(camera_name, serial)
    return (entry["ip"] if entry else "") or default


def is_monitored(device_name: str, serial: str = None, default: bool = False) -> bool:
    """Return whether a camera should be monitored (status_monitor field)."""
    entry = _find_entry(device_name, serial)
    return entry["monitor"] if entry else default


def get_detection_overrides(camera_name: str, serial: str = None) -> dict:
    """Return per-camera YOLO verification overrides set in the JSON
    ('ignore_zones', 'min_confidence', 'min_box_height_ratio',
    'min_aspect_ratio', 'imgsz'), for cameras with a known recurring
    false positive/negative that a global threshold shouldn't have to
    compromise for. Only fields actually set are returned, so callers
    can pass the result straight through as **kwargs.
    """
    entry = _find_entry(camera_name, serial)
    if entry is None:
        return {}
    overrides = {f: entry[f] for f in OVERRIDE_FIELDS if entry.get(f) is not None}
    if "ignore_zones" in overrides:
        overrides["ignore_zones"] = [tuple(zone) for zone in overrides["ignore_zones"]]
    return overrides


def get_data_source() -> str:
    """Return the path to the mapping file, or None if it doesn't exist."""
    return JSON_PATH if os.path.exists(JSON_PATH) else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    test_names = ["B14-BWH", "C3-BWH", "WAMENA OUT", "not_in_json", "B10-BWH",
                  "B10-BAWAH", "A2-ATAS", "A2-BAWAH", "B7-BAWAH", "B6-III-GENSET"]
    print(f"Data source: {get_data_source()}\n")
    for n in test_names:
        print(f"{n!r:22} -> site={get_site(n)!r:20} cluster={get_location(n)!r:15} "
              f"ip={get_ip(n)!r:16} monitor={is_monitored(n)!r} overrides={get_detection_overrides(n)}")
