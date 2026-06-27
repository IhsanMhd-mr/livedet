"""
LIVEDET – Live WebSocket Server
================================
Standalone WebSocket server for real-time pothole detection.

Architecture
------------
Browser  ──base64 JPEG──▶  Server  (one frame per message)
Browser  ◀──JSON detections──  Server

Each JSON response contains per-detection fields:
  bbox, confidence, depth_cm, width_cm, height_cm,
  distance_m, severity, severity_score

Depth pipeline (hybrid MiDaS + heuristic)
------------------------------------------
MiDaS runs every DEPTH_INTERVAL frames (default: 3).
On every detection, compute_midas_confidence() measures how reliable
MiDaS is for that specific bounding box region (based on texture variance).
blend_depth() then combines MiDaS and heuristic depth proportionally:

  confidence >= 0.70 → MiDaS depth only
  0.30 <= conf < 0.70 → weighted blend (more MiDaS as confidence rises)
  confidence < 0.30  → heuristic depth only (textureless / low-light)

This means the system automatically adapts: on textured potholes in
good light it trusts MiDaS; on smooth uniform asphalt or at night it
falls back gracefully to the geometry heuristic.

REST API comparison
-------------------
app.py (REST /predict, /video/process) uses compute_heuristic_measurements()
only — a deliberate design choice to avoid MiDaS GPU overhead on
stateless single-image requests. live_ws.py has a persistent frame loop
and cached depth map, making MiDaS much more efficient here.

Usage
-----
    python backend/live_ws.py
    python backend/live_ws.py --port 8765 --focal-length 600
"""

import asyncio
import argparse
import json
import base64
import sys
import time
import logging
from pathlib import Path
from collections import deque
from typing import Optional

import cv2
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
# Add backend/ and project root to sys.path so imports resolve regardless of
# where the script is launched from.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from detector import ObjectDetector
from utils import (
    DepthEstimator,
    extract_median_depth,
    compute_midas_confidence,     # NEW: measures how reliable MiDaS is
    blend_depth,                  # NEW: hybrid MiDaS + heuristic blend
    compute_real_width,
    compute_depth_cm,
    compute_heuristic_measurements,  # used inside the blend fallback
    classify_severity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveWS")


# ═══════════════════════════════════════════════════════════════════════════
#  Server Configuration Constants
# ═══════════════════════════════════════════════════════════════════════════

WS_HOST = "0.0.0.0"    # Listen on all interfaces
WS_PORT = 8765          # WebSocket port (must match React client config)
FPS_WINDOW = 30         # Rolling window size for FPS calculation (frames)

# MiDaS runs every DEPTH_INTERVAL frames.
# On the other frames the previous cached depth map is reused.
# Rationale: MiDaS takes ~8 ms on GPU. At 30 FPS, running every frame
# would consume 240 ms/s of GPU time. At interval=3 it uses ~80 ms/s,
# leaving headroom for YOLO inference and WebSocket I/O.
DEPTH_INTERVAL = 3

# Runtime config dict — allows CLI overrides without global declarations
runtime_config = {
    "focal_length": 600.0,   # Pixels — typical webcam focal length at 640 px width
}


# ═══════════════════════════════════════════════════════════════════════════
#  Global Model Singletons
# ═══════════════════════════════════════════════════════════════════════════
# Models are loaded once at server startup via initialize_models() and
# shared across all client connections. This avoids reloading weights
# (each ~18–190 MB) per connection.

detector: Optional[ObjectDetector] = None
depth_estimator: Optional[DepthEstimator] = None
cfg: Optional[Config] = None


def initialize_models():
    """
    Load YOLO11s and MiDaS Small once at server startup.

    Why load once globally?
    -----------------------
    Loading ML models is expensive: YOLO11s is ~18 MB and takes ~1 s to
    initialise on GPU; MiDaS Small is ~190 MB and takes ~2-3 s. Loading
    them per-connection would make the first detection after each browser
    reconnect unacceptably slow. The singleton pattern loads once and
    keeps the models resident in GPU VRAM for the lifetime of the server.

    Failure behaviour
    -----------------
    If MiDaS fails to load (e.g. no GPU, torch.hub network error), the
    server still starts and depth falls back to the heuristic. YOLO
    failure is fatal — the server cannot function without detection.
    """
    global detector, depth_estimator, cfg

    cfg = Config()
    device = getattr(cfg, "DEVICE", "cpu")

    # ── YOLO11s ──────────────────────────────────────────────────────────
    logger.info("Initialising YOLO …")
    detector = ObjectDetector(
        model_path=cfg.BEST_MODEL_PATH,
        device=device,
        confidence_threshold=getattr(cfg, "CONFIDENCE_THRESHOLD", 0.25),
        iou_threshold=getattr(cfg, "IOU_THRESHOLD", 0.45),
    )
    logger.info("YOLO ✓")

    # ── MiDaS Small ──────────────────────────────────────────────────────
    logger.info("Initialising MiDaS depth …")
    depth_estimator = DepthEstimator(device=device)
    if depth_estimator.initialized:
        logger.info("MiDaS ✓")
    else:
        logger.warning(
            "MiDaS failed to load – depth will use heuristic fallback only"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket Client Handler
# ═══════════════════════════════════════════════════════════════════════════

async def handle_client(websocket):
    """
    Handle one connected browser client for the duration of the session.

    Protocol
    --------
    Client sends: one Base64-encoded JPEG frame per message
    Server sends: one JSON object per frame with all detection results

    State per connection
    --------------------
    frame_count  : increments each frame, drives DEPTH_INTERVAL logic
    cached_depth : last MiDaS depth map (shape H×W), reused on throttled frames
    focal        : camera focal length in pixels (from CLI or default)
    fps_times    : rolling deque of per-frame processing times for FPS calc

    Error handling
    --------------
    Frame-level errors (bad base64, corrupt JPEG) are caught and sent back
    as {"error": "..."} without crashing the connection. Connection-level
    errors (client disconnect) are caught at the outer try/except and
    trigger a clean disconnect log.
    """
    addr = websocket.remote_address
    logger.info(f"[+] Client connected: {addr}")

    frame_count  = 0
    cached_depth: Optional[np.ndarray] = None
    focal        = runtime_config["focal_length"]
    fps_times: deque = deque(maxlen=FPS_WINDOW)

    try:
        async for message in websocket:
            t0 = time.perf_counter()

            try:
                # ── 1. Decode incoming Base64 JPEG frame ─────────────────
                # The browser captures a webcam frame, draws it on a canvas,
                # calls canvas.toDataURL("image/jpeg"), strips the data: prefix,
                # and sends the raw Base64 string over the WebSocket.
                img_bytes  = base64.b64decode(message)
                arr        = np.frombuffer(img_bytes, dtype=np.uint8)
                frame_bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame_bgr is None:
                    # imdecode returns None for corrupt/unrecognised data
                    await websocket.send(json.dumps({"error": "bad frame"}))
                    continue

                # MiDaS and YOLO both expect RGB; OpenCV decodes as BGR
                frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                fh, fw     = frame_bgr.shape[:2]   # frame height, width

                # ── 2. YOLO11s object detection ──────────────────────────
                # detector.detect() runs a single forward pass through YOLO11s
                # and returns a list of detections, each with:
                #   bbox (x,y,w,h), confidence, class_name
                result     = detector.detect(frame_rgb)
                detections = result.get("detections", [])

                # ── 3. MiDaS depth map (every DEPTH_INTERVAL frames) ─────
                # Running MiDaS on every frame would be too expensive on a
                # GTX 1650. We run it every 3rd frame and cache the result.
                # On the intermediate frames we reuse the cached map but
                # re-extract the median for each detection's NEW bbox position —
                # so the depth is from an older map but the spatial sampling
                # is always up-to-date with the current YOLO detections.
                if depth_estimator and frame_count % DEPTH_INTERVAL == 0:
                    cached_depth = depth_estimator.estimate(frame_rgb)

                # ── 4. Per-detection measurement and severity ─────────────
                det_out = []
                for i, det in enumerate(detections):
                    x, y, w, h = det["bbox"]
                    conf       = det.get("confidence", 0.0)
                    cls_name   = det.get("class_name", "Unknown")

                    if cached_depth is not None:
                        # ── MiDaS path ────────────────────────────────────
                        # Extract the median depth value inside this bbox
                        # from the cached MiDaS depth map
                        med = extract_median_depth(cached_depth, (x, y, w, h))

                        # Pure MiDaS depth estimate (before blending)
                        midas_depth_cm = compute_depth_cm(med)

                        # Geometry-based heuristic estimate (backup)
                        heuristic = compute_heuristic_measurements(
                            (x, y, w, h), (fh, fw)
                        )

                        # Measure how reliable MiDaS is for this specific
                        # bbox region (based on local depth map variance)
                        midas_conf = compute_midas_confidence(
                            cached_depth, (x, y, w, h)
                        )

                        # Hybrid blend: weight MiDaS vs heuristic by confidence
                        # High confidence  → MiDaS dominant
                        # Low confidence   → heuristic dominant (textureless)
                        depth_cm = blend_depth(
                            midas_depth_cm,
                            heuristic["depth_cm"],
                            midas_conf,
                        )

                        # Width via pinhole model (focal length based)
                        width_cm  = compute_real_width(float(w), med, focal)
                        height_cm = compute_real_width(float(h), med, focal)

                        # Distance: MiDaS outputs inverse depth (higher = closer).
                        # Invert so that far objects get larger distance values.
                        distance_m = max((1.0 - float(med)) * 5.0 + 0.3, 0.3)

                    else:
                        # ── Pure heuristic fallback ───────────────────────
                        # MiDaS has not run yet (first frames) or failed to
                        # load at startup. Use geometry-only estimates.
                        heuristic  = compute_heuristic_measurements(
                            (x, y, w, h), (fh, fw)
                        )
                        depth_cm   = heuristic["depth_cm"]
                        width_cm   = heuristic["width_cm"]
                        height_cm  = (h / max(fh, 1)) * 50.0

                        y_center   = y + h / 2.0
                        y_norm     = y_center / max(fh, 1)
                        distance_m = max(5.0 - (y_norm * 4.5), 0.5)

                        midas_conf = 0.0   # signal to client that MiDaS inactive

                    # ── 5. Severity classification ────────────────────────
                    # Combines depth, width, and YOLO confidence into a
                    # continuous score → Low / Medium / High / Critical label
                    severity_label, severity_score = classify_severity(
                        depth_cm, width_cm, conf
                    )

                    det_out.append({
                        "id":             i + 1,
                        "bbox":           [int(x), int(y), int(w), int(h)],
                        "class_name":     cls_name,
                        "confidence":     round(float(conf),          3),
                        "depth_cm":       round(float(depth_cm),      1),
                        "width_cm":       round(float(width_cm),      1),
                        "height_cm":      round(float(height_cm),     1),
                        "distance_m":     round(float(distance_m),    2),
                        "severity":       severity_label,
                        "severity_score": round(float(severity_score),3),
                        "midas_confidence": round(float(midas_conf),  2),
                    })

                # ── 6. FPS calculation ────────────────────────────────────
                # Rolling average over FPS_WINDOW frames to smooth spikes
                dt = time.perf_counter() - t0
                fps_times.append(dt)
                fps = (
                    1.0 / (sum(fps_times) / len(fps_times))
                    if fps_times else 0.0
                )
                frame_count += 1

                # ── 7. Send JSON response to browser ─────────────────────
                resp = {
                    "detections":   det_out,
                    "num_potholes": len(det_out),
                    "fps":          round(fps, 1),
                    "frame_count":  frame_count,
                    "depth_active": cached_depth is not None,
                }
                await websocket.send(json.dumps(resp))

            except Exception as exc:
                # Frame-level error — log, send error JSON, keep connection alive
                logger.error(f"Frame error: {exc}", exc_info=True)
                try:
                    await websocket.send(json.dumps({"error": str(exc)}))
                except Exception:
                    pass  # websocket itself broken — outer try handles

    except Exception:
        pass  # Connection closed or broken — normal browser disconnect

    finally:
        logger.info(f"[-] Client disconnected: {addr}")


# ═══════════════════════════════════════════════════════════════════════════
#  Server Entry-point
# ═══════════════════════════════════════════════════════════════════════════

async def main(host: str, port: int):
    """
    Start the WebSocket server and run indefinitely.

    max_size=10MB: allows high-resolution JPEG frames without truncation.
    ping_interval/timeout: detects stale connections after 20 s of silence.
    asyncio.Future() with no result keeps the coroutine alive forever —
    the server only stops on SIGINT (Ctrl+C) or process kill.
    """
    import websockets

    initialize_models()

    logger.info(f"WebSocket server starting on ws://{host}:{port}")
    async with websockets.serve(
        handle_client,
        host,
        port,
        max_size=10 * 1024 * 1024,   # 10 MB max message size
        ping_interval=20,
        ping_timeout=20,
    ):
        logger.info(f"✓ WebSocket server ready — ws://{host}:{port}")
        await asyncio.Future()        # run until cancelled


def cli():
    """
    Command-line interface for the WebSocket server.
    Allows overriding host, port, and focal length at launch:
        python live_ws.py --port 8766 --focal-length 720
    Focal length affects compute_real_width() — adjust if your camera
    has a different sensor/resolution combination.
    """
    parser = argparse.ArgumentParser(description="LIVEDET Live WebSocket Server")
    parser.add_argument("--host",         default=WS_HOST)
    parser.add_argument("--port",         type=int,   default=WS_PORT)
    parser.add_argument("--focal-length", type=float,
                        default=runtime_config["focal_length"])
    args = parser.parse_args()

    runtime_config["focal_length"] = args.focal_length
    asyncio.run(main(args.host, args.port))


if __name__ == "__main__":
    cli()
