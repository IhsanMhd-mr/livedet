"""
LIVEDET — Flask Backend API
=============================
REST API for road damage detection with image upload,
YOLO inference, and annotated image response.

Endpoints:
    GET  /              → Serves frontend UI
    POST /predict       → Run detection on uploaded image
    GET  /health        → Health check
"""

import os
import sys
import base64
import logging
from datetime import datetime
from pathlib import Path

# Ensure backend/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import uuid

from config import config
from detector import ObjectDetector
from storage_manager import StorageManager
from utils import (
    convert_numpy_types,
    compute_heuristic_measurements,
    classify_severity,
    encode_image_base64,
    SEVERITY_COLORS_BGR,
)

# ── Logging ─────────────────────────────────────────────────────────────
log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level, format="[%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Storage ─────────────────────────────────────────────────────────────
StorageManager.initialize()

# ── Flask App ───────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
)
CORS(app)

# ── Detector ────────────────────────────────────────────────────────────
detector = None
try:
    logger.info("[app] Initializing YOLO detector...")
    detector = ObjectDetector(
        model_path=config.BEST_MODEL_PATH,
        model_type=config.MODEL_TYPE,
        device=config.DEVICE,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )
    logger.info("[app] ✓ Detector initialized successfully")
except Exception as e:
    logger.error(f"[app] Detector initialization failed: {e}")
    detector = None


# ── Annotation helper ────────────────────────────────────────────────────
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
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{cls_name} | {severity} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
        cv2.rectangle(out, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(out, label, (x + 3, y - 4), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        sub = f"D:{det.get('depth_cm', 0):.1f}cm  W:{det.get('width_cm', 0):.1f}cm"
        cv2.putText(out, sub, (x + 3, y + h + 14), font, 0.45, color, 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════


@app.route("/")
def index():
    """Serve the frontend UI."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": detector is not None and detector.is_ready,
            "model_name": detector.loaded_model_name if detector else None,
            "device": str(config.DEVICE),
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Detect road damage in an uploaded image.

    POST /predict
    Form Data:
        - image (required): Image file to analyze
        - confidence (optional): Detection threshold 0.0-1.0 (default 0.5)

    Returns:
        JSON with detections, annotated image (base64), statistics.
    """
    session_id = str(uuid.uuid4())[:8]

    try:
        logger.info(f"[/predict] [{session_id}] Request received")

        if detector is None or not detector.is_ready:
            return jsonify({"error": "Detector not initialized", "status": "error"}), 500

        # ── 1. Parse image ──────────────────────────────────────────────
        if "image" not in request.files:
            return jsonify({"error": "No image provided", "status": "error"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image selected", "status": "error"}), 400

        image_data = file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format", "status": "error"}), 400

        H, W, C = image.shape
        logger.info(f"[/predict] [{session_id}] Image: {file.filename}, shape: {image.shape}")

        # ── 2. Save original ────────────────────────────────────────────
        try:
            original_path, _ = StorageManager.save_original(image, session_id)
        except Exception:
            original_path = None

        # ── 3. Confidence ────────────────────────────────────────────────
        try:
            confidence = float(request.form.get("confidence", config.CONFIDENCE_THRESHOLD))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = config.CONFIDENCE_THRESHOLD

        # ── 4. Run detection ─────────────────────────────────────────────
        result = detector.detect(image, confidence_threshold=confidence)

        if result.get("status") != "success":
            return jsonify(result), 500

        detections = result.get("detections", [])
        logger.info(f"[/predict] [{session_id}] ✓ {len(detections)} detection(s)")

        # ── 5. Depth / width / severity per detection ────────────────────
        for i, det in enumerate(detections):
            det["id"] = i + 1
            bbox = det.get("bbox", [])
            if len(bbox) >= 4:
                measurements = compute_heuristic_measurements(bbox, (H, W))
                det["depth_cm"] = round(measurements["depth_cm"], 1)
                det["width_cm"] = round(measurements["width_cm"], 1)
                det["area"] = int(bbox[2] * bbox[3])

                sev_label, sev_score = classify_severity(
                    det["depth_cm"], det["width_cm"], det["confidence"]
                )
                det["severity"] = sev_label
                det["severity_score"] = round(sev_score, 3)
            else:
                det["depth_cm"] = 0.0
                det["width_cm"] = 0.0
                det["area"] = 0
                det["severity"] = "Low"
                det["severity_score"] = 0.0

        # ── 6. Annotate image ────────────────────────────────────────────
        annotated = annotate_detections(image, detections)

        # ── 7. Save processed ────────────────────────────────────────────
        try:
            processed_path = StorageManager.save_processed(annotated, session_id)
        except Exception:
            processed_path = None

        # ── 8. Encode response ───────────────────────────────────────────
        image_b64 = encode_image_base64(annotated)

        # ── 9. Build response ────────────────────────────────────────────
        depths = [d["depth_cm"] for d in detections if d["depth_cm"] > 0]
        widths = [d["width_cm"] for d in detections if d["width_cm"] > 0]
        sev_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        for d in detections:
            sev_counts[d["severity"]] = sev_counts.get(d["severity"], 0) + 1

        response = {
            "status": "success",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "model": detector.loaded_model_name,
            "detections": detections,
            "count": len(detections),
            "num_potholes": len(detections),
            "image_shape": [H, W, C],
            "image": image_b64,
            "storage": {
                "original": original_path,
                "processed": processed_path,
            },
            "summary": {
                "total":           len(detections),
                "severity_counts": sev_counts,
                "avg_depth_cm":    round(float(np.mean(depths)), 1) if depths else 0.0,
                "avg_width_cm":    round(float(np.mean(widths)), 1) if widths else 0.0,
                "max_depth_cm":    round(float(np.max(depths)), 1) if depths else 0.0,
                "max_width_cm":    round(float(np.max(widths)), 1) if widths else 0.0,
            },
            "average_depth_cm": round(float(np.mean(depths)), 1) if depths else 0.0,
            "average_width_cm": round(float(np.mean(widths)), 1) if widths else 0.0,
            "max_depth_cm":     round(float(np.max(depths)), 1) if depths else 0.0,
            "max_width_cm":     round(float(np.max(widths)), 1) if widths else 0.0,
        }

        response = convert_numpy_types(response)
        logger.info(f"[/predict] [{session_id}] ✓ Response ready")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"[/predict] [{session_id}] Error: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}", "status": "error"}), 500


# ── Error handlers ──────────────────────────────────────────────────────


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("[app] Starting LIVEDET Flask server...")
    logger.info(f"[app] http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    logger.info("[app] Endpoints: GET / | POST /predict | GET /health")

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=False,
        threaded=True,
    )
