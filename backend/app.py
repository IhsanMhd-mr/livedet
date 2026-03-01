"""
LIVEDET — Flask Backend API
Road Defect Detection with YOLOv8s (Finetuned)
"""

import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from detector import ObjectDetector
from utils import (
    compute_heuristic_measurements,
    classify_severity,
    encode_image_base64,
    convert_numpy_types,
    SEVERITY_COLORS_BGR,
)
from storage_manager import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Init ─────────────────────────────────────────────────────────────────
StorageManager.initialize()
app = Flask(__name__)
CORS(app)

cfg = Config()
detector: ObjectDetector = None

try:
    logger.info("[app] Loading ObjectDetector …")
    detector = ObjectDetector(
        model_path=cfg.BEST_MODEL_PATH,
        model_type=cfg.MODEL_TYPE,
        device=cfg.DEVICE,
        confidence_threshold=cfg.CONFIDENCE_THRESHOLD,
    )
    logger.info(f"[app] ✓ Detector ready — {detector.loaded_model_name}")
except Exception as exc:
    logger.error(f"[app] Detector init failed: {exc}")
    detector = None


# ── Annotation helper ─────────────────────────────────────────────────────
def annotate_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw severity-coloured boxes and labels onto a copy of the image."""
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x, y, w, h = det["bbox"]
        severity  = det.get("severity", "Low")
        cls_name  = det.get("class_name", "object")
        conf      = det.get("confidence", 0.0)

        color = SEVERITY_COLORS_BGR.get(severity, (0, 255, 0))

        # Bounding box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # Label: "ClassName | Severity conf%"
        label = f"{cls_name} | {severity} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
        cv2.rectangle(out, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(out, label, (x + 3, y - 4), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Sub-label: depth / width
        sub = f"D:{det.get('depth_cm', 0):.1f}cm  W:{det.get('width_cm', 0):.1f}cm"
        cv2.putText(out, sub, (x + 3, y + h + 14), font, 0.45, color, 1, cv2.LINE_AA)

    return out


# ── /predict ──────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Form: image (file), confidence (float, optional)
    Returns: JSON with detections, annotated image (base64), stats
    """
    session_id = str(uuid.uuid4())[:8]

    if detector is None:
        return jsonify({"status": "error", "error": "Detector not initialised"}), 500

    if "image" not in request.files:
        return jsonify({"status": "error", "error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"status": "error", "error": "Empty filename"}), 400

    # ── Decode ───────────────────────────────────────────────────────────
    raw = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"status": "error", "error": "Invalid image format"}), 400

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Save original ─────────────────────────────────────────────────────
    try:
        StorageManager.save_original_image_array(image_bgr, session_id, image_format="jpg")
    except Exception:
        pass

    # ── Confidence threshold ──────────────────────────────────────────────
    try:
        conf_thresh = float(request.form.get("confidence", cfg.CONFIDENCE_THRESHOLD))
        conf_thresh = max(0.0, min(1.0, conf_thresh))
    except (ValueError, TypeError):
        conf_thresh = cfg.CONFIDENCE_THRESHOLD

    # ── YOLO inference ────────────────────────────────────────────────────
    yolo_result = detector.detect(image_rgb, confidence_threshold=conf_thresh)
    if yolo_result.get("status") != "success":
        return jsonify(yolo_result), 500

    raw_dets = yolo_result.get("detections", [])

    # ── Enrich detections ─────────────────────────────────────────────────
    detections = []
    for i, det in enumerate(raw_dets):
        bbox = det["bbox"]
        m = compute_heuristic_measurements(bbox, (H, W))
        depth_cm = m["depth_cm"]
        width_cm = m["width_cm"]
        severity_label, severity_score = classify_severity(depth_cm, width_cm, det["confidence"])

        detections.append({
            "id":             i + 1,
            "class_id":       det["class_id"],
            "class_name":     det["class_name"],
            "confidence":     round(det["confidence"], 3),
            "bbox":           det["bbox"],
            "depth_cm":       round(depth_cm, 1),
            "width_cm":       round(width_cm, 1),
            "severity":       severity_label,
            "severity_score": round(severity_score, 3),
            "area":           int(bbox[2] * bbox[3]),
        })

    # ── Annotate & encode ─────────────────────────────────────────────────
    annotated = annotate_detections(image_bgr, detections)
    image_b64 = encode_image_base64(annotated)

    # ── Save processed ────────────────────────────────────────────────────
    try:
        StorageManager.save_processed_image(annotated, session_id, image_format="jpg")
    except Exception:
        pass

    # ── Stats ─────────────────────────────────────────────────────────────
    depths = [d["depth_cm"] for d in detections]
    widths = [d["width_cm"] for d in detections]
    sev_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for d in detections:
        sev_counts[d["severity"]] = sev_counts.get(d["severity"], 0) + 1

    summary = {
        "total":          len(detections),
        "severity_counts": sev_counts,
        "avg_depth_cm":   round(float(np.mean(depths)), 1) if depths else 0.0,
        "avg_width_cm":   round(float(np.mean(widths)), 1) if widths else 0.0,
        "max_depth_cm":   round(float(np.max(depths)), 1) if depths else 0.0,
        "max_width_cm":   round(float(np.max(widths)), 1) if widths else 0.0,
    }

    body = {
        "status":      "success",
        "session_id":  session_id,
        "timestamp":   datetime.now().isoformat(),
        "model":       detector.loaded_model_name or "unknown",
        "image_shape": [H, W, image_bgr.shape[2]],
        "image":       image_b64,
        "count":       len(detections),
        "detections":  detections,
        "summary":     summary,
        # Convenience aliases kept for frontend compatibility
        "num_potholes":      len(detections),
        "average_depth_cm":  summary["avg_depth_cm"],
        "average_width_cm":  summary["avg_width_cm"],
        "max_depth_cm":      summary["max_depth_cm"],
        "max_width_cm":      summary["max_width_cm"],
    }

    logger.info(
        f"[/predict] [{session_id}] ✓ {len(detections)} detection(s) | "
        f"sev={sev_counts} | model={detector.loaded_model_name}"
    )

    return jsonify(convert_numpy_types(body)), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info(f"[app] Starting on http://{cfg.FLASK_HOST}:{cfg.FLASK_PORT}")
    app.run(host=cfg.FLASK_HOST, port=cfg.FLASK_PORT, debug=False, threaded=True)
