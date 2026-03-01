"""
LIVEDET — Live Camera Detection System
=======================================
Real-time detection using YOLO + MiDaS depth estimation.

Features:
  - Webcam capture via OpenCV
  - YOLO per-frame detection
  - MiDaS depth map (every N frames for performance)
  - Real-world width: RealWidth = (PixelWidth × Depth) / FocalLength
  - Severity labels (Low / Medium / High / Critical)
  - Bounding box annotations with depth, width, severity
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
import torch

# ── Ensure backend/ is on the path ──────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from detector import ObjectDetector
from utils import DepthEstimator, extract_median_depth, compute_real_width, compute_depth_cm, classify_severity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveDetect")


# ═══════════════════════════════════════════════════════════════════════════
#  Constants & Defaults
# ═══════════════════════════════════════════════════════════════════════════

MAX_FRAME_WIDTH = 640          # Resize for performance
DEPTH_INTERVAL  = 3            # Run MiDaS every N frames
DEFAULT_FOCAL_LENGTH = 600.0   # Pixels – typical webcam at 640px width
DEFAULT_CAMERA_INDEX = 0
FPS_SMOOTHING_WINDOW = 30      # Rolling average over N frames

# Severity thresholds (combined score 0-100)
SEVERITY_THRESHOLDS = {
    "Low":      (0,  25),
    "Medium":   (25, 50),
    "High":     (50, 75),
    "Critical": (75, 100),
}

# Colours (BGR)
COLOR_BOX      = (0, 255, 0)
COLOR_LOW      = (0, 200, 0)
COLOR_MEDIUM   = (0, 200, 255)
COLOR_HIGH     = (0, 100, 255)
COLOR_CRITICAL = (0, 0, 255)
COLOR_FPS      = (255, 255, 0)
COLOR_INFO     = (200, 200, 200)

SEVERITY_COLORS = {
    "Low":      COLOR_LOW,
    "Medium":   COLOR_MEDIUM,
    "High":     COLOR_HIGH,
    "Critical": COLOR_CRITICAL,
}


# ═══════════════════════════════════════════════════════════════════════════
#  MiDaS Depth Estimator  (self-contained, no external download at import)
# ═══════════════════════════════════════════════════════════════════════════

class LiveMiDaSDepth:
    """
    Lightweight MiDaS wrapper optimised for live video.
    Downloads the model on first call, caches it for subsequent runs.
    """

    def __init__(self, model_type: str = "MiDaS_small", device: str = "cuda:0"):
        self.device = device
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.initialized = False
        self._initialize(model_type, device)

    # ── init ────────────────────────────────────────────────────────────
    def _initialize(self, model_type: str, device: str):
        try:
            logger.info(f"[MiDaS] Loading {model_type} on {device} …")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.eval().to(device)

            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if "small" in model_type.lower():
                self.transform = transforms.small_transform
            else:
                self.transform = transforms.dpt_transform

            self.initialized = True
            logger.info("[MiDaS] ✓ Model ready")
        except Exception as exc:
            logger.error(f"[MiDaS] Init failed: {exc}")
            self.initialized = False

    # ── inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def estimate(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return normalised depth map (H, W) in [0, 1]. Higher = farther."""
        if not self.initialized:
            return None
        inp = self.transform(frame_rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = pred.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        return depth  # 0 = near, 1 = far


# ═══════════════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def resize_frame(frame: np.ndarray, max_width: int = MAX_FRAME_WIDTH) -> np.ndarray:
    """Resize keeping aspect ratio so width ≤ max_width."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_median_depth(depth_map: np.ndarray,
                         bbox: Tuple[int, int, int, int]) -> float:
    """
    Extract the **median** depth inside a bounding box.

    Args:
        depth_map: (H, W) normalised depth.
        bbox: (x, y, w, h) as returned by DeploymentPipeline.

    Returns:
        Median depth value (0-1). Falls back to 0.5 on error.
    """
    x, y, w, h = bbox
    h_map, w_map = depth_map.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_map, x + w)
    y2 = min(h_map, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.5
    region = depth_map[y1:y2, x1:x2]
    return float(np.median(region))


def compute_real_width(pixel_width: float,
                       depth: float,
                       focal_length: float) -> float:
    """
    RealWidth = (PixelWidth × Depth) / FocalLength

    `depth` comes from MiDaS (relative 0-1); we scale it to a plausible
    metric range (metres) so the width output is in **centimetres**.
    A depth value of 1.0 ≈ ~5 m away (rough heuristic for road cameras).
    """
    depth_metres = max(depth * 5.0, 0.3)            # clamp minimum 30 cm
    real_width_m = (pixel_width * depth_metres) / focal_length
    return real_width_m * 100.0                      # → centimetres


def compute_depth_cm(depth_value: float) -> float:
    """
    Convert normalised MiDaS depth into an approximate depth-of-pothole
    estimate in centimetres.

    Logic: a pothole closer to the camera (lower depth value) is deeper
    because we are looking *down* at the road.  We invert and scale.
    """
    inverted = 1.0 - depth_value          # invert: 0=far → 1=near surface
    depth_cm = inverted * 15.0            # max ~15 cm pothole depth
    return max(depth_cm, 0.5)


def classify_severity(depth_cm: float, width_cm: float, confidence: float) -> str:
    """
    Severity = 50 % depth + 30 % width + 20 % confidence.
    Normalise depth to 0-100 (max 15 cm) and width to 0-100 (max 100 cm).
    """
    d_score = min(depth_cm / 15.0, 1.0) * 100
    w_score = min(width_cm / 100.0, 1.0) * 100
    c_score = confidence * 100
    score = d_score * 0.50 + w_score * 0.30 + c_score * 0.20
    for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= score < hi:
            return label
    return "Critical"


def draw_detections(frame: np.ndarray,
                    detections: List[Dict],
                    depth_map: Optional[np.ndarray],
                    focal_length: float) -> np.ndarray:
    """
    Draw annotated bounding boxes with depth, width, severity.

    Args:
        frame:      BGR image to draw on (will be modified in-place).
        detections: List from DeploymentPipeline.detect()['detections'].
        depth_map:  Latest MiDaS depth map (may be None on first frames).
        focal_length: Camera focal length in pixels.

    Returns:
        Annotated frame.
    """
    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det.get("confidence", 0.0)

        # ── Depth / width / severity ────────────────────────────────
        if depth_map is not None:
            med_depth = extract_median_depth(depth_map, (x, y, w, h))
            depth_cm  = compute_depth_cm(med_depth)
            width_cm  = compute_real_width(float(w), med_depth, focal_length)
        else:
            depth_cm = 0.0
            width_cm = 0.0

        severity = classify_severity(depth_cm, width_cm, conf)
        color = SEVERITY_COLORS.get(severity, COLOR_BOX)

        # ── Bounding box ────────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # ── Label background ────────────────────────────────────────
        label_lines = [
            f"{severity} ({conf:.0%})",
            f"D:{depth_cm:.1f}cm  W:{width_cm:.1f}cm",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        line_h = 18
        max_tw = 0
        for line in label_lines:
            tw, _ = cv2.getTextSize(line, font, font_scale, thickness)[0:1][0], 0
            tw = cv2.getTextSize(line, font, font_scale, thickness)[0][0]
            max_tw = max(max_tw, tw)

        # Semi-transparent background
        overlay = frame.copy()
        box_y1 = max(y - line_h * len(label_lines) - 6, 0)
        cv2.rectangle(overlay, (x, box_y1), (x + max_tw + 8, y), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Text
        for i, line in enumerate(label_lines):
            ty = y - (len(label_lines) - i - 1) * line_h - 4
            cv2.putText(frame, line, (x + 4, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def draw_hud(frame: np.ndarray,
             fps: float,
             num_potholes: int,
             depth_active: bool) -> np.ndarray:
    """Draw FPS and status bar at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 26),
                font, 0.65, COLOR_FPS, 2, cv2.LINE_AA)

    mid_text = f"Potholes: {num_potholes}"
    cv2.putText(frame, mid_text, (w // 2 - 60, 26),
                font, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

    depth_text = "Depth: ON" if depth_active else "Depth: loading…"
    depth_col  = (0, 255, 0) if depth_active else (0, 180, 255)
    cv2.putText(frame, depth_text, (w - 160, 26),
                font, 0.55, depth_col, 1, cv2.LINE_AA)

    # Help text at bottom
    cv2.putText(frame, "Press 'q' to quit", (8, h - 10),
                font, 0.45, COLOR_INFO, 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_live_detection(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    depth_interval: int = DEPTH_INTERVAL,
    max_width: int = MAX_FRAME_WIDTH,
    confidence: float = 0.5,
):
    """
    Launch the live camera detection window.

    Args:
        camera_index:   cv2.VideoCapture device id.
        focal_length:   Approx. focal length in pixels.
        depth_interval: Run MiDaS every N frames.
        max_width:      Resize frames to this width max.
        confidence:     YOLO confidence threshold.
    """

    # ── Load config ─────────────────────────────────────────────────────
    config = Config()
    device = config.DEVICE if hasattr(config, "DEVICE") else "cpu"

    # ── Initialise YOLO ─────────────────────────────────────────────────
    detector = ObjectDetector(
        model_path=config.BEST_MODEL_PATH,
        device=device,
        confidence_threshold=confidence,
    )
    logger.info("YOLO ✓ ready")

    # ── Initialise MiDaS ───────────────────────────────────────────────
    logger.info("Initialising MiDaS depth estimator …")
    depth_estimator = DepthEstimator(device=device)
    depth_active = depth_estimator is not None

    # ── Open camera ─────────────────────────────────────────────────────
    logger.info(f"Opening camera {camera_index} …")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_index}")
        sys.exit(1)

    # Try to set resolution hints (camera may ignore these)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info(f"Camera opened  –  native res "
                f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # ── State ───────────────────────────────────────────────────────────
    frame_count = 0
    fps_times: deque = deque(maxlen=FPS_SMOOTHING_WINDOW)
    cached_depth_map: Optional[np.ndarray] = None
    window_name = "LIVEDET – Live Detection"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    logger.info("▶  Live detection started.  Press 'q' to quit.\n")

    try:
        while True:
            t_start = time.perf_counter()

            # ── 1. Capture ──────────────────────────────────────────────
            ret, raw_frame = cap.read()
            if not ret:
                logger.warning("Frame grab failed – retrying …")
                continue

            frame = resize_frame(raw_frame, max_width)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── 2. YOLO Detection ───────────────────────────────────────
            result = detector.detect(frame_rgb)
            detections = result.get("detections", [])

            # ── 3. MiDaS Depth (every N frames) ────────────────────────
            if depth_estimator and frame_count % depth_interval == 0:
                new_depth = depth_estimator.estimate(frame_rgb)
                if new_depth is not None:
                    cached_depth_map = new_depth

            # ── 4. Draw annotations ─────────────────────────────────────
            frame = draw_detections(frame, detections, cached_depth_map, focal_length)

            # ── 5. FPS ──────────────────────────────────────────────────
            t_end = time.perf_counter()
            fps_times.append(t_end - t_start)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

            frame = draw_hud(frame, fps, len(detections), depth_active and cached_depth_map is not None)

            # ── 6. Display ──────────────────────────────────────────────
            cv2.imshow(window_name, frame)
            frame_count += 1

            # ── 7. Quit on 'q' ──────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("'q' pressed – shutting down.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted – shutting down.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Goodbye.")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="LIVEDET – Live Camera Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--camera",       type=int,   default=DEFAULT_CAMERA_INDEX,
                        help=f"Camera index (default {DEFAULT_CAMERA_INDEX})")
    parser.add_argument("--focal-length", type=float, default=DEFAULT_FOCAL_LENGTH,
                        help=f"Focal length in pixels (default {DEFAULT_FOCAL_LENGTH})")
    parser.add_argument("--depth-interval", type=int, default=DEPTH_INTERVAL,
                        help=f"Run depth estimation every N frames (default {DEPTH_INTERVAL})")
    parser.add_argument("--max-width",    type=int,   default=MAX_FRAME_WIDTH,
                        help=f"Max frame width (default {MAX_FRAME_WIDTH})")
    parser.add_argument("--confidence",   type=float, default=0.5,
                        help="YOLO confidence threshold (default 0.5)")
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
