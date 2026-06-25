"""
LIVEDET — Utility Functions
=============================
Shared utilities for depth estimation, severity classification,
image encoding, and numpy type conversion.

Architecture note
-----------------
This file contains ALL active depth logic used at runtime:
  - DepthEstimator      : wraps Intel MiDaS Small for monocular depth maps
  - extract_median_depth: crops a bounding box from a depth map → single value
  - compute_midas_confidence: judges how reliable MiDaS is for a given bbox
  - blend_depth         : hybrid MiDaS + heuristic weighted blend
  - compute_real_width  : converts pixel width → centimetres via focal length
  - compute_depth_cm    : converts normalised MiDaS value → centimetres
  - classify_severity   : scoring formula → Low / Medium / High / Critical
  - compute_heuristic_measurements: geometry-only fallback (no MiDaS needed)

The legacy files midas_depth_estimator.py and depth_pipeline.py are NOT
imported at runtime. They are retained as future-work starting points.
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

DEFAULT_FOCAL_LENGTH = 600.0  # Pixels — typical webcam at 640 px width

# Severity score thresholds (output of classify_severity scoring formula)
SEVERITY_LEVELS = {
    "Low":      (0.00, 0.30),
    "Medium":   (0.30, 0.55),
    "High":     (0.55, 0.75),
    "Critical": (0.75, 1.01),
}

# BGR colours used by the annotator for bounding box overlays
SEVERITY_COLORS_BGR = {
    "Low":      (0, 200,   0),   # Green
    "Medium":   (0, 200, 255),   # Yellow
    "High":     (0, 100, 255),   # Orange
    "Critical": (0,   0, 255),   # Red
}

# Hybrid blend thresholds
# If MiDaS confidence >= HIGH_CONF  → use MiDaS depth directly
# If MiDaS confidence >= LOW_CONF   → weighted blend of MiDaS + heuristic
# If MiDaS confidence <  LOW_CONF   → use heuristic depth directly
MIDAS_CONF_HIGH = 0.70   # trust MiDaS fully above this
MIDAS_CONF_LOW  = 0.30   # below this, heuristic takes over completely


# ═══════════════════════════════════════════════════════════════════════════
#  MiDaS Depth Estimator
# ═══════════════════════════════════════════════════════════════════════════

class DepthEstimator:
    """
    Wraps Intel MiDaS Small for monocular depth estimation.

    What it does
    ------------
    Takes a single RGB frame and returns a normalised inverse-depth map
    of the same spatial dimensions (H x W), where:
        0.0 = objects close to the camera
        1.0 = objects far from the camera

    Why monocular depth?
    --------------------
    A standard YOLO bounding box only gives 2D pixel coordinates — it
    cannot tell whether a pothole is shallow or deep (monocular scale
    ambiguity). MiDaS recovers relative depth cues from texture gradients,
    perspective, and shading without any additional hardware.

    Limitation
    ----------
    MiDaS outputs RELATIVE depth, not metric centimetres. The values are
    proportional rather than physically calibrated. We apply an empirical
    scale factor (15 cm max) to map them to approximate cavity depths.
    On textureless asphalt MiDaS confidence degrades — the hybrid blend
    in blend_depth() handles this by leaning on the heuristic fallback.

    Usage
    -----
        estimator = DepthEstimator(device="cuda:0")
        depth_map = estimator.estimate(frame_rgb)  # shape (H, W), float32
    """

    def __init__(self, model_type: str = "MiDaS_small", device: str = "cpu"):
        self.device = device
        self.model = None
        self.transform = None
        self.initialized = False
        self._initialize(model_type, device)

    def _initialize(self, model_type: str, device: str):
        """
        Load MiDaS weights from torch.hub (downloads on first run, then cached).
        Sets self.initialized = True only if loading fully succeeds.
        """
        try:
            logger.info(f"[MiDaS] Loading {model_type} on {device}...")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.eval().to(device)

            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            # Small variant uses a different input transform than DPT variants
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
        """
        Run MiDaS inference on one RGB frame.

        Steps
        -----
        1. Apply MiDaS input transform (resize + normalise to ImageNet stats)
        2. Run the model — output is a raw inverse-depth tensor
        3. Bicubic-upsample back to the original frame resolution
        4. Min-max normalise to [0, 1]

        Returns
        -------
        np.ndarray shape (H, W), float32 in [0, 1]
            0 = close to camera, 1 = far from camera
        None if the model is not initialised (load failed at startup)

        Why @torch.no_grad()?
        ---------------------
        Disables gradient tracking during inference — saves memory and
        speeds up the forward pass since we never need to backpropagate.
        """
        if not self.initialized:
            return None

        # Transform expects RGB numpy array; returns a batched tensor
        inp = self.transform(frame_rgb).to(self.device)

        # Forward pass through MiDaS
        pred = self.model(inp)

        # Upsample from MiDaS internal resolution back to original frame size
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Move to CPU and normalise to [0, 1]
        depth = pred.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        # 1e-6 prevents division by zero on uniform frames (e.g. pure white)

        return depth   # 0 = near, 1 = far


# ═══════════════════════════════════════════════════════════════════════════
#  Depth / Width / Hybrid Blend Functions
# ═══════════════════════════════════════════════════════════════════════════

def extract_median_depth(
    depth_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    """
    Sample the MiDaS depth map inside a YOLO bounding box.

    Why median?
    -----------
    The YOLO bounding box is rectangular and may include road-surface pixels
    around the pothole edges. The median is more robust than the mean to
    these outlier road pixels — it ignores the top and bottom halves of the
    distribution and returns the middle value.

    Parameters
    ----------
    depth_map : np.ndarray (H, W) — full-frame normalised depth map
    bbox      : (x, y, w, h) in pixel coordinates (YOLO format)

    Returns
    -------
    float in [0, 1] — median depth inside the bbox
    0.5 on error (neutral fallback, maps to ~7.5 cm via compute_depth_cm)
    """
    x, y, w, h = bbox
    h_map, w_map = depth_map.shape[:2]

    # Clamp to frame bounds (handles partial bboxes at frame edges)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_map, x + w)
    y2 = min(h_map, y + h)

    if x2 <= x1 or y2 <= y1:
        # Degenerate bbox — return neutral mid-range value
        return 0.5

    region = depth_map[y1:y2, x1:x2]
    return float(np.median(region))


def compute_midas_confidence(
    depth_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    """
    Estimate how reliable the MiDaS depth reading is inside a bounding box.

    Principle
    ---------
    MiDaS derives depth from texture gradients and shading cues. On a
    richly textured surface, the depth map shows high local variance —
    the model has strong signal to work with. On textureless asphalt
    (smooth, uniform grey), the depth map is nearly flat, indicating
    that MiDaS has little signal and the output is unreliable.

    We use the standard deviation of depth values inside the bounding box
    as a proxy for texture richness:
        high std  → textured surface → MiDaS is reliable → high confidence
        low std   → textureless      → MiDaS is unreliable → low confidence

    Formula
    -------
        confidence = clip(1.0 - std * 2.0, 0.0, 1.0)
    The factor 2.0 scales std (typically 0.0–0.5) to the full [0,1] range.

    Parameters
    ----------
    depth_map : np.ndarray (H, W) — full-frame normalised depth map
    bbox      : (x, y, w, h) in pixel coordinates

    Returns
    -------
    float in [0.0, 1.0]
        1.0 = MiDaS is fully reliable (rich texture)
        0.0 = MiDaS is unreliable (textureless surface)
        0.0 on degenerate bbox
    """
    x, y, w, h = bbox
    h_map, w_map = depth_map.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_map, x + w), min(h_map, y + h)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = depth_map[y1:y2, x1:x2]
    std = float(np.std(region))

    # Invert: high std → high confidence; clip to [0, 1]
    confidence = float(np.clip(1.0 - std * 2.0, 0.0, 1.0))
    return confidence


def blend_depth(
    midas_cm: float,
    heuristic_cm: float,
    midas_confidence: float,
) -> float:
    """
    Hybrid blend of MiDaS neural depth and geometry-based heuristic depth.

    Why blend instead of a hard switch?
    ------------------------------------
    A hard switch creates discontinuous jumps in depth_cm between frames
    as confidence crosses the threshold. A weighted blend produces smooth
    transitions, reducing severity tier oscillation at the boundaries.

    Decision logic
    --------------
    confidence >= MIDAS_CONF_HIGH (0.70):
        Trust MiDaS fully — texture is rich, gradient signal is strong.
        Return midas_cm directly.

    MIDAS_CONF_LOW (0.30) <= confidence < MIDAS_CONF_HIGH (0.70):
        Partial trust — surface has some texture but also smooth regions.
        Blend proportionally: higher confidence → more MiDaS weight.
        blend = confidence * midas_cm + (1 - confidence) * heuristic_cm

    confidence < MIDAS_CONF_LOW (0.30):
        Do not trust MiDaS — surface is textureless (smooth asphalt,
        wet road, painted markings). Fall back entirely to heuristic.
        Return heuristic_cm directly.

    Typical scenarios
    -----------------
    Textured pothole in good light   → confidence ≈ 0.8 → MiDaS wins
    Smooth asphalt, partial texture  → confidence ≈ 0.5 → 50/50 blend
    Uniform grey road, no texture    → confidence ≈ 0.1 → heuristic wins
    Night-time frame (flat MiDaS)    → confidence ≈ 0.0 → heuristic wins

    Parameters
    ----------
    midas_cm      : depth estimate from MiDaS pipeline (cm)
    heuristic_cm  : depth estimate from geometry heuristic (cm)
    midas_confidence : float in [0, 1] from compute_midas_confidence()

    Returns
    -------
    float : blended depth estimate in centimetres
    """
    if midas_confidence >= MIDAS_CONF_HIGH:
        # MiDaS is reliable — use it directly
        return midas_cm

    elif midas_confidence >= MIDAS_CONF_LOW:
        # Partial trust — weighted blend
        # Example: confidence=0.5 → 50% MiDaS + 50% heuristic
        return (midas_confidence * midas_cm
                + (1.0 - midas_confidence) * heuristic_cm)

    else:
        # MiDaS is unreliable — fall back to geometry heuristic
        return heuristic_cm


def compute_real_width(
    pixel_width: float,
    depth: float,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
) -> float:
    """
    Estimate physical width from pixel width using a pinhole camera model.

    Formula (similar triangles / pinhole projection)
    ------------------------------------------------
        RealWidth_m = (PixelWidth × DepthMetres) / FocalLength

    Where
    -----
    - PixelWidth   : bounding box pixel width (YOLO output)
    - DepthMetres  : estimated distance to the pothole in metres
                     (MiDaS normalised value scaled to a 0.3–5.0 m range)
    - FocalLength  : camera focal length in pixels (default 600 px for a
                     standard webcam at 640 px width)

    Why this formula?
    -----------------
    From the pinhole model: an object of physical width W at distance D
    projects to pixel_width = (W × f) / D, so W = (pixel_width × D) / f.
    This is the standard similar-triangles relationship used in
    monocular metric estimation (Zhang, 2000).

    Limitation
    ----------
    The depth value is MiDaS normalised (relative), scaled empirically to
    metres. The focal length is a default estimate, not camera-intrinsic
    calibrated. Width is therefore approximate, not engineering-grade.

    Returns
    -------
    float : estimated real width in centimetres
    """
    depth_metres = max(depth * 5.0, 0.3)   # scale [0,1] to [0.3, 5.0] m range
    real_width_m = (pixel_width * depth_metres) / focal_length
    return real_width_m * 100.0             # convert metres → centimetres


def compute_depth_cm(depth_value: float) -> float:
    """
    Convert a normalised MiDaS depth value to approximate pothole cavity
    depth in centimetres.

    Formula
    -------
        depth_cm = (1.0 - depth_value) × 15.0

    Why invert?
    -----------
    MiDaS output convention: 0 = close to camera, 1 = far from camera.
    When looking down at a road from a dashcam:
        - The road surface level has a certain baseline depth value.
        - A pothole cavity appears CLOSER to the camera than the flat road
          (the cavity floor is physically lower but optically nearer to the
          nadir of the camera's field of view).
    So a lower depth_value (closer) maps to a larger cavity — we invert.

    Why 15.0?
    ---------
    15 cm is the empirical maximum depth scale for the UK road defect
    classification scale. Most dangerous potholes are < 15 cm deep.
    This maps the full [0, 1] MiDaS range to [0, 15] cm.

    Floor of 0.5 cm
    ---------------
    Prevents returning zero for a perfectly flat MiDaS region
    (which would incorrectly imply no cavity at all).

    Returns
    -------
    float : estimated cavity depth in centimetres, minimum 0.5 cm
    """
    inverted = 1.0 - depth_value
    depth_cm = inverted * 15.0
    return max(depth_cm, 0.5)


def classify_severity(
    depth_cm: float,
    width_cm: float,
    confidence: float,
) -> Tuple[str, float]:
    """
    Compute a continuous severity score and classify into a severity tier.

    Scoring formula
    ---------------
        d_norm = min(depth_cm  / 15.0, 1.0)   ← normalise to [0, 1]
        w_norm = min(width_cm / 100.0, 1.0)   ← normalise to [0, 1]
        score  = d_norm × 0.50
               + w_norm × 0.30
               + confidence × 0.20

    Why these weights?
    ------------------
    Depth is weighted most heavily (0.50) because a deep pothole causes
    the most vehicle damage regardless of its lateral size.
    Width contributes 0.30 because a wide defect is harder to avoid and
    more likely to catch both wheels.
    Detection confidence contributes 0.20 as a quality signal — a low-
    confidence detection may be a false positive.

    Tier classification
    -------------------
    Critical : score > 0.65  (severe depth AND width combined)
    High     : width > 50 cm  OR  depth > 8 cm
    Medium   : width >= 20 cm OR  depth >= 3 cm
    Low      : otherwise

    Thresholds rationale
    --------------------
    The 3 cm / 8 cm depth thresholds and 20 cm / 50 cm width thresholds
    are grounded in UK Highways Agency pothole inspection guidelines,
    where a defect deeper than 40 mm (4 cm) is classified as requiring
    immediate repair. The thresholds are deliberately conservative to
    prioritise driver safety over minimising false alerts.

    Parameters
    ----------
    depth_cm   : estimated cavity depth (cm)
    width_cm   : estimated pothole width (cm)
    confidence : YOLO detection confidence score [0, 1]

    Returns
    -------
    (label, score) where label is one of Low / Medium / High / Critical
    and score is the continuous float in [0, 1]
    """
    d_norm = min(depth_cm  / 15.0, 1.0)
    w_norm = min(width_cm / 100.0, 1.0)

    score = d_norm * 0.50 + w_norm * 0.30 + confidence * 0.20
    score = float(np.clip(score, 0.0, 1.0))

    if score > 0.65:
        label = "Critical"
    elif width_cm > 50 or depth_cm > 8:
        label = "High"
    elif width_cm >= 20 or depth_cm >= 3:
        label = "Medium"
    else:
        label = "Low"

    return label, score


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry-Based Heuristic (no MiDaS required)
# ═══════════════════════════════════════════════════════════════════════════

def compute_heuristic_measurements(
    bbox: List[int],
    image_shape: Tuple[int, int],
) -> Dict:
    """
    Estimate pothole depth and width from bounding box geometry alone.
    Used by the REST API (app.py) for all requests, and by live_ws.py
    as the fallback component inside the hybrid blend when MiDaS
    confidence is low.

    Why geometry instead of MiDaS for the REST API?
    ------------------------------------------------
    The REST API processes single images or video files without a
    persistent GPU context or frame-to-frame cache. Running MiDaS on
    every REST call would add ~8 ms GPU scheduling overhead with no
    caching benefit. The heuristic provides immediate estimates at
    near-zero computational cost — a deliberate design trade-off.

    Width estimation
    ----------------
        width_cm = (bbox_width / frame_width) × 100.0
    Assumes the frame represents a ~1 m wide road section at a typical
    dashcam viewing distance. Clamped to [5, 80] cm to prevent outlier
    bboxes from producing unrealistic values.

    Depth estimation
    ----------------
        raw_depth = (bbox_height / frame_height) × 30.0

    Perspective damping
    -------------------
    Objects lower in the frame are closer to the camera and appear
    magnified by perspective. Without correction a close large pothole
    would be overestimated. The damping factor reduces depth for
    objects nearer the bottom of the frame (large y_norm):
        perspective_factor = max(1.0 - 0.6 × y_norm, 0.4)
    At y_norm = 0 (top of frame, far away) factor = 1.0 → no damping
    At y_norm = 1 (bottom of frame, close) factor = 0.4 → 60% reduction

    Depth clamped to [2, 12] cm to cap outlier bboxes.

    Limitation
    ----------
    This heuristic assumes a flat road surface and a forward-facing
    fixed-pitch dashcam. It does not account for actual texture, lighting,
    or road geometry. The 30.0 and 0.6 constants are empirically derived,
    not physically calibrated.

    Parameters
    ----------
    bbox        : [x, y, w, h] in pixels (YOLO format)
    image_shape : (H, W) of the source frame

    Returns
    -------
    dict with keys depth_cm (float) and width_cm (float)
    """
    x, y, w, h = bbox
    H, W = image_shape[:2]

    # ── Width ────────────────────────────────────────────────────────────
    width_cm = (w / max(W, 1)) * 100.0
    width_cm = float(np.clip(width_cm, 5.0, 80.0))

    # ── Depth: base estimate from bbox height ratio ───────────────────────
    raw_depth = (h / max(H, 1)) * 30.0

    # ── Perspective damping based on vertical position ───────────────────
    y_center = y + h / 2.0
    y_norm   = y_center / max(H, 1)
    perspective_factor = max(1.0 - 0.6 * y_norm, 0.4)

    depth_cm = raw_depth * perspective_factor
    depth_cm = float(np.clip(depth_cm, 2.0, 12.0))

    return {"depth_cm": depth_cm, "width_cm": width_cm}


# ═══════════════════════════════════════════════════════════════════════════
#  Image Encoding Helpers
# ═══════════════════════════════════════════════════════════════════════════

def encode_image_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """
    Encode a BGR numpy image to a base64 UTF-8 string.
    Used by the REST API to return annotated images in the JSON response
    without writing files to disk.
    """
    _, buffer = cv2.imencode(fmt, image)
    return base64.b64encode(buffer).decode("utf-8")


def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    """
    Decode a base64 string back to a BGR numpy image.
    Returns None on any decoding error (malformed input, wrong format).
    """
    try:
        img_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Numpy → JSON Type Conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert_numpy_types(obj):
    """
    Recursively convert numpy scalar and array types to Python native
    types so the result can be serialised to JSON.

    Why needed?
    -----------
    json.dumps() cannot serialise numpy.int64, numpy.float32, etc. by
    default. This helper walks any nested dict / list structure and
    converts every numpy type to its Python equivalent before serialisation.
    """
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
