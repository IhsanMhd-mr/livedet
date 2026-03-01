"""
LIVEDET — Live Camera Detection
==================================
Real-time webcam detection using YOLO + MiDaS depth estimation.

Features:
  - Webcam capture via OpenCV
  - YOLO per-frame detection
  - MiDaS depth map (every N frames for performance)
  - Real-world width estimation
  - Severity labels (Low / Medium / High / Critical)
  - FPS display
  - Press 'q' to quit

Usage:
    python backend/live_detect.py
    python backend/live_detect.py --camera 1 --focal-length 600
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from detector import ObjectDetector
from utils import (
    DepthEstimator,
    extract_median_depth,
    compute_real_width,
    compute_depth_cm,
    classify_severity,
    SEVERITY_COLORS_BGR,
    DEFAULT_FOCAL_LENGTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveDetect")

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

MAX_FRAME_WIDTH = 640
DEPTH_INTERVAL = 3
FPS_SMOOTHING_WINDOW = 30


# ═══════════════════════════════════════════════════════════════════════════
#  Drawing Helpers
# ═══════════════════════════════════════════════════════════════════════════

def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
    depth_map: Optional[np.ndarray],
    focal_length: float,
) -> np.ndarray:
    """Draw annotated bounding boxes with depth, width, severity."""
    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det.get("confidence", 0.0)

        if depth_map is not None:
            med = extract_median_depth(depth_map, (x, y, w, h))
            d_cm = compute_depth_cm(med)
            w_cm = compute_real_width(float(w), med, focal_length)
        else:
            d_cm = 0.0
            w_cm = 0.0

        severity, _ = classify_severity(d_cm, w_cm, conf)
        color = SEVERITY_COLORS_BGR.get(severity, (0, 255, 0))

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        lines = [
            f"{severity} ({conf:.0%})",
            f"D:{d_cm:.1f}cm  W:{w_cm:.1f}cm",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(lines):
            ty = y - (len(lines) - i - 1) * 18 - 4
            cv2.putText(frame, line, (x + 4, ty), font, 0.50, color, 1, cv2.LINE_AA)

    return frame


def draw_hud(
    frame: np.ndarray, fps: float, num_detections: int, depth_active: bool
) -> np.ndarray:
    """Draw FPS and status bar at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 26), font, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detections: {num_detections}", (w // 2 - 70, 26), font, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

    depth_text = "Depth: ON" if depth_active else "Depth: loading..."
    depth_col = (0, 255, 0) if depth_active else (0, 180, 255)
    cv2.putText(frame, depth_text, (w - 160, 26), font, 0.55, depth_col, 1, cv2.LINE_AA)

    cv2.putText(frame, "Press 'q' to quit", (8, h - 10), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def resize_frame(frame: np.ndarray, max_width: int = MAX_FRAME_WIDTH) -> np.ndarray:
    """Resize keeping aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_live_detection(
    camera_index: int = 0,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    depth_interval: int = DEPTH_INTERVAL,
    max_width: int = MAX_FRAME_WIDTH,
    confidence: float = 0.5,
):
    """Launch the live camera detection window."""

    cfg = Config()
    device = cfg.DEVICE

    # ── YOLO ────────────────────────────────────────────────────────────
    logger.info("Initialising YOLO...")
    det = ObjectDetector(
        model_path=cfg.BEST_MODEL_PATH,
        model_type=cfg.MODEL_TYPE,
        device=device,
        confidence_threshold=confidence,
    )
    logger.info("YOLO ✓")

    # ── MiDaS ───────────────────────────────────────────────────────────
    logger.info("Initialising MiDaS depth estimator...")
    midas = DepthEstimator(model_type="MiDaS_small", device=device)
    depth_active = midas.initialized

    # ── Camera ──────────────────────────────────────────────────────────
    logger.info(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info(
        f"Camera opened — {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    frame_count = 0
    fps_times: deque = deque(maxlen=FPS_SMOOTHING_WINDOW)
    cached_depth: Optional[np.ndarray] = None
    window = "LIVEDET — Live Detection"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    logger.info("▶  Live detection started.  Press 'q' to quit.\n")

    try:
        while True:
            t0 = time.perf_counter()

            ret, raw = cap.read()
            if not ret:
                continue

            frame = resize_frame(raw, max_width)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO
            result = det.detect(frame_rgb, confidence_threshold=confidence)
            detections = result.get("detections", [])

            # MiDaS
            if depth_active and frame_count % depth_interval == 0:
                new_depth = midas.estimate(frame_rgb)
                if new_depth is not None:
                    cached_depth = new_depth

            # Draw
            frame = draw_detections(frame, detections, cached_depth, focal_length)

            dt = time.perf_counter() - t0
            fps_times.append(dt)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

            frame = draw_hud(frame, fps, len(detections), depth_active and cached_depth is not None)

            cv2.imshow(window, frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("'q' pressed — shutting down.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Goodbye.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="LIVEDET — Live Camera Detection"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--focal-length", type=float, default=DEFAULT_FOCAL_LENGTH)
    parser.add_argument("--depth-interval", type=int, default=DEPTH_INTERVAL)
    parser.add_argument("--max-width", type=int, default=MAX_FRAME_WIDTH)
    parser.add_argument("--confidence", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_live_detection(
        camera_index=args.camera,
        focal_length=args.focal_length,
        depth_interval=args.depth_interval,
        max_width=args.max_width,
        confidence=args.confidence,
    )
