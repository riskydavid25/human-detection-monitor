"""Human-shape verification using YOLOv8n.

Runs as a second-stage visual check before an alert is sent, on top of
whatever triggered the capture (motion detection, cloud alarm, etc.).
Shared by ezviz_cloud_monitor.py and nonezviz_rtsp_monitor.py.

Detections are filtered by three signals, tuned for outdoor CCTV:
  - confidence      : YOLO's own person-class confidence.
  - box height ratio: filters tiny artifacts (e.g. OSD timestamp text).
  - aspect ratio     : a standing person is always taller than wide;
                        text/watermarks are always wider than tall, so
                        this catches false positives that height alone
                        would miss (and lets a genuinely small/distant
                        person through, which a height-only cutoff would
                        wrongly discard).

Frames that are too dark (typical of IR night mode) get a contrast
boost (CLAHE) before inference, since YOLOv8n was trained mostly on
daylight imagery and tends to under-detect low-contrast night frames.

Requires the `ultralytics` package and a local yolov8n.pt weights file
(downloaded automatically on first use if not already present).
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0  # COCO dataset class index for "person"

# Frames with mean grayscale brightness below this are treated as
# low-light/IR and get a contrast boost before inference.
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 70

_model = None
_model_load_attempted = False


def _load_model() -> bool:
    global _model, _model_load_attempted

    if _model_load_attempted:
        return _model is not None

    _model_load_attempted = True
    try:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
        return True
    except Exception:
        logger.exception(
            "Failed to load YOLOv8n model. Verification will fail closed "
            "(always report no detection) until this is resolved."
        )
        return False


def _enhance_low_light(img_bgr: np.ndarray) -> np.ndarray:
    """Boost local contrast (CLAHE on the L channel) for dark frames."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab_enhanced = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _in_ignore_zone(x1, y1, x2, y2, ignore_zones) -> bool:
    """True if the detection's center point falls inside any ignore zone."""
    if not ignore_zones:
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return any(zx1 <= cx <= zx2 and zy1 <= cy <= zy2 for zx1, zy1, zx2, zy2 in ignore_zones)


def contains_human_shape(
    image_path: str,
    min_confidence: float = 0.4,
    min_box_height_ratio: float = 0.04,
    min_aspect_ratio: float = 0.8,
    imgsz: int = 960,
    enhance_low_light: bool = True,
    ignore_zones: list = None,
) -> tuple:
    """Check whether an image contains a person.

    Args:
        image_path: Path to the snapshot to analyze.
        min_confidence: Minimum YOLO person-class confidence to accept.
        min_box_height_ratio: Minimum detection box height, as a
            fraction of frame height.
        min_aspect_ratio: Minimum box height/width ratio; rejects
            flat/wide boxes typical of watermark text.
        imgsz: Inference resolution. Higher values improve recall for
            small/distant subjects at the cost of speed.
        enhance_low_light: Apply contrast enhancement to dark frames.
        ignore_zones: Optional list of (x1, y1, x2, y2) pixel rectangles
            (in the original image) where detections are discarded.
            Useful for a fixed static object that repeatedly triggers a
            false positive on one specific camera.

    Returns:
        (detected, best_confidence, person_count, boxes) where boxes is
        a list of (x1, y1, x2, y2, confidence) for accepted detections.
    """
    if not _load_model() or not os.path.exists(image_path):
        return False, 0.0, 0, []

    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not read image: %s", image_path)
        return False, 0.0, 0, []

    inference_input = img
    if enhance_low_light:
        mean_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        if mean_brightness < LOW_LIGHT_BRIGHTNESS_THRESHOLD:
            inference_input = _enhance_low_light(img)

    results = _model.predict(inference_input, verbose=False, conf=min_confidence, imgsz=imgsz)

    best_confidence = 0.0
    boxes_out = []

    for result in results:
        if result.boxes is None:
            continue
        img_height = result.orig_shape[0]
        for box in result.boxes:
            if int(box.cls[0]) != _PERSON_CLASS_ID:
                continue
            confidence = float(box.conf[0])
            if confidence < min_confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            height_ratio = (y2 - y1) / img_height if img_height else 0
            aspect_ratio = (y2 - y1) / max(x2 - x1, 1e-6)

            if height_ratio < min_box_height_ratio or aspect_ratio < min_aspect_ratio:
                continue
            if _in_ignore_zone(x1, y1, x2, y2, ignore_zones):
                continue

            best_confidence = max(best_confidence, confidence)
            boxes_out.append((int(x1), int(y1), int(x2), int(y2), confidence))

    return len(boxes_out) > 0, best_confidence, len(boxes_out), boxes_out


def draw_boxes_on_image(image_path: str, boxes: list, output_path: str) -> bool:
    """Draw green boxes with confidence labels and save to output_path.

    Returns True on success, False if the source image can't be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not read image to annotate: %s", image_path)
        return False

    GREEN = (0, 255, 0)
    for x1, y1, x2, y2, confidence in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        label = f"Person {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 6, text_h + 4)
        cv2.rectangle(img, (x1, label_y - text_h - 4), (x1 + text_w + 4, label_y + 2), GREEN, -1)
        cv2.putText(img, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.imwrite(output_path, img)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python common_human_verifier_yolo.py <image.jpg>")
        sys.exit(1)

    path = sys.argv[1]
    detected, confidence, count, boxes = contains_human_shape(path)
    print(f"File: {path}")
    print(f"Detected: {detected}  Confidence: {confidence:.3f}  Count: {count}")
    print(f"Boxes: {boxes}")

    if boxes:
        out_path = path.rsplit(".", 1)[0] + "_annotated.jpg"
        if draw_boxes_on_image(path, boxes, out_path):
            print(f"Annotated image saved to: {out_path}")
