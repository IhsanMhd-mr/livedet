"""
LIVEDET — Utility Functions
=============================
Shared utilities for depth estimation, severity classification,
image encoding, and numpy type conversion.
"""

import base64
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_FOCAL_LENGTH = 600.0  # Pixels — typical webcam at 640px width

SEVERITY_LEVELS = {
    "Low": (0.00, 0.30),
    "Medium": (0.30, 0.55),
    "High": (0.55, 0.75),
    "Critical": (0.75, 1.01),
}

SEVERITY_COLORS_BGR = {
    "Low": (0, 200, 0),       # Green
    "Medium": (0, 200, 255),  # Yellow
    "High": (0, 100, 255),    # Orange
    "Critical": (0, 0, 255),  # Red
}


# ═══════════════════════════════════════════════════════════════════════════
#  MiDaS Depth Estimator
# ═══════════════════════════════════════════════════════════════════════════

class DepthEstimator:
    """
    MiDaS monocular depth estimation wrapper.
    Downloads the model on first use, caches it for subsequent runs.
    """

    def __init__(self, model_type: str = "MiDaS_small", device: str = "cpu"):
        self.device = device
        self.model = None
        self.transform = None
        self.initialized = False
        self._initialize(model_type, device)

    def _initialize(self, model_type: str, device: str):
        try:
            logger.info(f"[MiDaS] Loading {model_type} on {device}...")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.eval().to(device)

            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if "small" in model_type.lower():
                self.transform = transforms.small_transform
            else:
                self.transform = transforms.dpt_transform

            self.initialized = True
            logger.info("[MiDaS] ✓ Depth model ready")
        except Exception as exc:
            logger.error(f"[MiDaS] Init failed: {exc}")
            self.initialized = False

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
#  Depth / Width / Severity Functions
# ═══════════════════════════════════════════════════════════════════════════

def extract_median_depth(
    depth_map: np.ndarray, bbox: Tuple[int, int, int, int]
) -> float:
    """
    Extract median depth inside a bounding box (x, y, w, h format).
    Falls back to 0.5 on error.
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


def compute_real_width(
    pixel_width: float, depth: float, focal_length: float = DEFAULT_FOCAL_LENGTH
) -> float:
    """
    RealWidth = (PixelWidth × Depth) / FocalLength

    `depth` is MiDaS normalised (0-1); scale to ~5 m range.
    Returns width in centimetres.
    """
    depth_metres = max(depth * 5.0, 0.3)
    real_width_m = (pixel_width * depth_metres) / focal_length
    return real_width_m * 100.0


def compute_depth_cm(depth_value: float) -> float:
    """
    Convert normalised MiDaS depth into approximate pothole depth in cm.
    Inverted: closer to camera → deeper pothole (looking down at road).
    """
    inverted = 1.0 - depth_value
    depth_cm = inverted * 15.0  # Max ~15 cm
    return max(depth_cm, 0.5)


def classify_severity(
    depth_cm: float, width_cm: float, confidence: float
) -> Tuple[str, float]:
    """
    Severity = 50% depth + 30% width + 20% confidence.
    Returns (label, score).
    """
    d_norm = min(depth_cm / 15.0, 1.0)
    w_norm = min(width_cm / 100.0, 1.0)
    score = d_norm * 0.50 + w_norm * 0.30 + confidence * 0.20
    for label, (lo, hi) in SEVERITY_LEVELS.items():
        if lo <= score < hi:
            return label, score
    return "Critical", score


# ═══════════════════════════════════════════════════════════════════════════
#  Simple Heuristic Depth/Width (no MiDaS needed)
# ═══════════════════════════════════════════════════════════════════════════

def compute_heuristic_measurements(
    bbox: List[int], image_shape: Tuple[int, int]
) -> Dict:
    """
    Compute depth/width/severity using simple heuristics (no MiDaS).

    Args:
        bbox: [x, y, w, h]
        image_shape: (H, W)

    Returns:
        dict with depth_cm, width_cm, severity, severity_score
    """
    x, y, w, h = bbox
    H, W = image_shape[:2]

    width_cm = (w / max(W, 1)) * 100.0
    height_ratio = h / max(w, 1)
    depth_cm = (h / max(H, 1)) * 100.0 + (height_ratio * 10.0)

    depth_cm = max(depth_cm, 5.0)
    width_cm = max(width_cm, 3.0)

    return {"depth_cm": depth_cm, "width_cm": width_cm}


# ═══════════════════════════════════════════════════════════════════════════
#  Image Encoding Helpers
# ═══════════════════════════════════════════════════════════════════════════

def encode_image_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR numpy image to base64 string."""
    _, buffer = cv2.imencode(fmt, image)
    return base64.b64encode(buffer).decode("utf-8")


def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """Decode a base64 string to BGR numpy image."""
    try:
        img_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Numpy → JSON Conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
