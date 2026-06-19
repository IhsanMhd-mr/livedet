"""
LIVEDET — Flask Backend API
Road Defect Detection with YOLOv11s (Finetuned)
"""

import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))

# pyrefly: ignore [missing-import]
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

# ── Configure request size limits ─────────────────────────────────────────
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB for file uploads
app.config['JSON_MAX_SIZE'] = 50 * 1024 * 1024       # 50 MB for JSON responses

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
    h_img, w_img = image.shape[:2]

    # Set scale factor relative to standard 960x541 image diagonal (1102.0)
    # to support different aspect ratios and orientations consistently
    diagonal = np.sqrt(w_img**2 + h_img**2)
    scale = max(diagonal / 1102.0, 0.5)
    font_scale_label = 0.55 * scale
    font_scale_sub = 0.40 * scale
    thickness = max(int(1 * scale), 1)
    box_thickness = max(int(2 * scale), 1)
    pad = max(int(6 * scale), 3)
    accent_bar_w = max(int(4 * scale), 2)

    for det in detections:
        x, y, w, h = det["bbox"]
        det_id = det.get("id", 1)
        severity  = det.get("severity", "Low")
        cls_name  = det.get("class_name", "object")
        conf      = det.get("confidence", 0.0)

        color = SEVERITY_COLORS_BGR.get(severity, (0, 255, 0))

        # Bounding box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, box_thickness)

        # Label: "#1 Pothole | Severity conf%"
        label = f"#{det_id} {cls_name} | {severity} {conf:.0%}"
        # Sub-label: depth / width / height
        sub = f"D:{det.get('depth_cm', 0):.1f}cm  W:{det.get('width_cm', 0):.1f}cm  H:{det.get('height_cm', 0):.1f}cm"

        (tw, th), _ = cv2.getTextSize(label, font, font_scale_label, thickness)
        (sw, sh), _ = cv2.getTextSize(sub, font, font_scale_sub, thickness)

        block_w = max(tw, sw) + 2 * pad + accent_bar_w
        block_h = th + sh + 3 * pad

        # Determine y position (draw inside box if label goes off-screen top)
        if y - block_h >= 0:
            by_start = y - block_h
        else:
            by_start = y
            
        by_end = by_start + block_h

        # Premium Dark navy/slate background block (BGR: 42, 23, 15)
        bg_color = (42, 23, 15)
        text_color = (248, 250, 252) # Off-white
        
        cv2.rectangle(out, (x, by_start), (x + block_w, by_end), bg_color, -1)

        # Draw left vertical accent bar of the severity color
        cv2.rectangle(out, (x, by_start), (x + accent_bar_w, by_end), color, -1)

        # Write text lines
        tx = x + accent_bar_w + pad
        ty1 = by_start + pad + th
        cv2.putText(out, label, (tx, ty1), font, font_scale_label, text_color, thickness, cv2.LINE_AA)

        ty2 = by_start + 2 * pad + th + sh
        cv2.putText(out, sub, (tx, ty2), font, font_scale_sub, color, thickness, cv2.LINE_AA)

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

    if "image" in request.files:
        file = request.files["image"]
    elif "file" in request.files:
        file = request.files["file"]
    else:
        return jsonify({"status": "error", "error": "No image or file provided"}), 400
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
        height_cm = max((bbox[3] / max(H, 1)) * 100.0, 3.0)
        x_cm = (bbox[0] / max(W, 1)) * 100.0
        y_cm = (bbox[1] / max(H, 1)) * 100.0
        severity_label, severity_score = classify_severity(depth_cm, width_cm, det["confidence"])

        detections.append({
            "id":             i + 1,
            "class_id":       det["class_id"],
            "class_name":     det["class_name"],
            "confidence":     round(det["confidence"], 3),
            "bbox":           det["bbox"],
            "x":              round(x_cm, 1),
            "y":              round(y_cm, 1),
            "width":          round(width_cm, 1),
            "height":         round(height_cm, 1),
            "depth_cm":       round(depth_cm, 1),
            "width_cm":       round(width_cm, 1),
            "height_cm":      round(height_cm, 1),
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
        "annotated_image": f"data:image/jpeg;base64,{image_b64}",
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


@app.route("/video/process", methods=["POST"])
def process_video():
    """
    POST /video/process
    File: file
    Returns: JSON with output_url, total_frames, total_detections, fps, duration
    """
    session_id = str(uuid.uuid4())[:8]

    if detector is None:
        return jsonify({"status": "error", "error": "Detector not initialised"}), 500

    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No video file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"status": "error", "error": "Empty filename"}), 400

    # Save original video file
    ext = Path(file.filename).suffix or ".mp4"
    orig_filename = f"{session_id}_original{ext}"
    orig_path = StorageManager.ORIGINAL_DIR / orig_filename
    file.save(str(orig_path))

    # Track file in StorageManager
    StorageManager._stored_files[session_id] = {
        'original': str(orig_path),
        'processed': None,
        'created': datetime.now(),
        'accessed': datetime.now()
    }

    # Open video
    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened():
        return jsonify({"status": "error", "error": "Could not open video file"}), 400

    # Read video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1
    duration = total_frames / orig_fps

    # Set up processed video output path
    processed_filename = f"{session_id}_processed.mp4"
    processed_path = StorageManager.PROCESSED_DIR / processed_filename

    # Read first frame to determine shape
    ret, frame_bgr = cap.read()
    if not ret:
        cap.release()
        return jsonify({"status": "error", "error": "Empty video file"}), 400

    # Resize first frame to get output dimensions
    MAX_WIDTH = 640
    h_orig, w_orig = frame_bgr.shape[:2]
    if w_orig > MAX_WIDTH:
        scale = MAX_WIDTH / w_orig
        out_w = MAX_WIDTH
        out_h = int(h_orig * scale)
    else:
        out_w = w_orig
        out_h = h_orig

    # Set up VideoWriter with H.264/AVC1 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(processed_path), fourcc, orig_fps, (out_w, out_h))
    
    # Fallback to 'mp4v' if avc1 init fails
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(processed_path), fourcc, orig_fps, (out_w, out_h))

    total_detections = 0
    frames_processed = 0

    try:
        # Loop over all frames
        while ret:
            # Resize
            if w_orig > MAX_WIDTH:
                frame_resized = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame_bgr

            H, W = frame_resized.shape[:2]
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # YOLO detection
            yolo_result = detector.detect(frame_rgb, confidence_threshold=cfg.CONFIDENCE_THRESHOLD)
            raw_dets = yolo_result.get("detections", [])

            detections = []
            for i, det in enumerate(raw_dets):
                bbox = det["bbox"]
                m = compute_heuristic_measurements(bbox, (H, W))
                depth_cm = m["depth_cm"]
                width_cm = m["width_cm"]
                height_cm = max((bbox[3] / max(H, 1)) * 100.0, 3.0)
                x_cm = (bbox[0] / max(W, 1)) * 100.0
                y_cm = (bbox[1] / max(H, 1)) * 100.0
                severity_label, severity_score = classify_severity(depth_cm, width_cm, det["confidence"])

                detections.append({
                    "id":             i + 1,
                    "class_id":       det["class_id"],
                    "class_name":     det["class_name"],
                    "confidence":     round(det["confidence"], 3),
                    "bbox":           det["bbox"],
                    "x":              round(x_cm, 1),
                    "y":              round(y_cm, 1),
                    "width":          round(width_cm, 1),
                    "height":         round(height_cm, 1),
                    "depth_cm":       round(depth_cm, 1),
                    "width_cm":       round(width_cm, 1),
                    "height_cm":      round(height_cm, 1),
                    "severity":       severity_label,
                    "severity_score": round(severity_score, 3),
                    "area":           int(bbox[2] * bbox[3]),
                })

            total_detections += len(detections)

            # Annotate
            annotated = annotate_detections(frame_resized, detections)

            # Write to output video
            out.write(annotated)
            frames_processed += 1

            # Read next frame
            ret, frame_bgr = cap.read()

    finally:
        cap.release()
        out.release()

    # Track processed file in storage manager
    StorageManager._stored_files[session_id]['processed'] = str(processed_path)

    # Return summary
    return jsonify({
        "status": "success",
        "output_url": f"/static/processed/{processed_filename}",
        "total_frames": frames_processed,
        "total_detections": total_detections,
        "fps": round(orig_fps, 1),
        "duration": round(duration, 1)
    }), 200


@app.route("/static/processed/<path:filename>")
def serve_processed_file(filename):
    return send_from_directory(StorageManager.PROCESSED_DIR, filename)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"status": "error", "error": "File too large — max 50 MB"}), 413


@app.errorhandler(431)
def request_header_fields_too_large(e):
    return jsonify({"status": "error", "error": "Request headers too large"}), 431


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info(f"[app] Starting on http://{cfg.FLASK_HOST}:{cfg.FLASK_PORT}")
    app.run(host=cfg.FLASK_HOST, port=cfg.FLASK_PORT, debug=False, threaded=True)
