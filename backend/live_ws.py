"""
LIVEDET — Live WebSocket Server
==================================
Real-time detection server for browser-based webcam streaming.

Flow:
  Browser  ──base64 JPEG──▶  Server
  Browser  ◀──JSON detections──  Server

Usage:
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
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveWS")

# ═══════════════════════════════════════════════════════════════════════════
#  Defaults
# ═══════════════════════════════════════════════════════════════════════════

WS_HOST = "0.0.0.0"
WS_PORT = 8765
DEPTH_INTERVAL = 3
FPS_WINDOW = 30

runtime_config = {
    "focal_length": 600.0,
}

# ═══════════════════════════════════════════════════════════════════════════
#  Global Model Singletons
# ═══════════════════════════════════════════════════════════════════════════

detector: Optional[ObjectDetector] = None
midas: Optional[DepthEstimator] = None
cfg: Optional[Config] = None


def initialize_models():
    """Load YOLO + MiDaS once on server start."""
    global detector, midas, cfg

    cfg = Config()
    device = cfg.DEVICE

    logger.info("Initialising YOLO...")
    detector = ObjectDetector(
        model_path=cfg.BEST_MODEL_PATH,
        model_type=cfg.MODEL_TYPE,
        device=device,
        confidence_threshold=cfg.CONFIDENCE_THRESHOLD,
    )
    logger.info("YOLO ✓")

    logger.info("Initialising MiDaS depth...")
    midas = DepthEstimator(model_type="MiDaS_small", device=device)
    if midas.initialized:
        logger.info("MiDaS ✓")
    else:
        logger.warning("MiDaS failed to load — using heuristic fallback")


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket Handler
# ═══════════════════════════════════════════════════════════════════════════

async def handle_client(websocket):
    """Handle one browser client — receive frames, return detections."""
    addr = websocket.remote_address
    logger.info(f"[+] Client connected: {addr}")

    frame_count = 0
    cached_depth: Optional[np.ndarray] = None
    focal = runtime_config["focal_length"]
    fps_times: deque = deque(maxlen=FPS_WINDOW)

    try:
        async for message in websocket:
            t0 = time.perf_counter()

            try:
                # Decode frame
                img_bytes = base64.b64decode(message)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame_bgr is None:
                    await websocket.send(json.dumps({"error": "bad frame"}))
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                fh, fw = frame_bgr.shape[:2]

                # YOLO
                result = detector.detect(frame_rgb)
                detections = result.get("detections", [])

                # MiDaS
                if midas and midas.initialized and frame_count % DEPTH_INTERVAL == 0:
                    depth = midas.estimate(frame_rgb)
                    if depth is not None:
                        cached_depth = depth

                # Per-detection measurements
                det_out = []
                for i, det in enumerate(detections):
                    x, y, w, h = det["bbox"]
                    conf      = det.get("confidence", 0.0)
                    cls_name  = det.get("class_name", "Unknown")

                    if cached_depth is not None:
                        med = extract_median_depth(cached_depth, (x, y, w, h))
                        depth_cm = compute_depth_cm(med)
                        width_cm = compute_real_width(float(w), med, focal)
                    else:
                        width_cm = (w / max(fw, 1)) * 50.0
                        depth_cm = (h / max(fh, 1)) * 15.0 + 2.0

                    severity, _ = classify_severity(depth_cm, width_cm, conf)

                    det_out.append(
                        {
                            "id":         i + 1,
                            "bbox":       [int(x), int(y), int(w), int(h)],
                            "class_name": cls_name,
                            "confidence": round(float(conf), 3),
                            "depth_cm":   round(float(depth_cm), 1),
                            "width_cm":   round(float(width_cm), 1),
                            "severity":   severity,
                        }
                    )

                # FPS
                dt = time.perf_counter() - t0
                fps_times.append(dt)
                fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0
                frame_count += 1

                resp = {
                    "detections": det_out,
                    "num_detections": len(det_out),
                    "fps": round(fps, 1),
                    "frame_count": frame_count,
                    "depth_active": cached_depth is not None,
                }
                await websocket.send(json.dumps(resp))

            except Exception as exc:
                logger.error(f"Frame error: {exc}", exc_info=True)
                try:
                    await websocket.send(json.dumps({"error": str(exc)}))
                except Exception:
                    pass

    except Exception:
        pass
    finally:
        logger.info(f"[-] Client disconnected: {addr}")


# ═══════════════════════════════════════════════════════════════════════════
#  Server Entry Point
# ═══════════════════════════════════════════════════════════════════════════

async def main(host: str, port: int):
    import websockets

    initialize_models()

    logger.info(f"WebSocket server starting on ws://{host}:{port}")
    async with websockets.serve(
        handle_client,
        host,
        port,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    ):
        logger.info(f"✓ WebSocket server ready — ws://{host}:{port}")
        await asyncio.Future()


def cli():
    parser = argparse.ArgumentParser(description="LIVEDET WebSocket Server")
    parser.add_argument("--host", default=WS_HOST)
    parser.add_argument("--port", type=int, default=WS_PORT)
    parser.add_argument("--focal-length", type=float, default=runtime_config["focal_length"])
    args = parser.parse_args()
    runtime_config["focal_length"] = args.focal_length
    asyncio.run(main(args.host, args.port))


if __name__ == "__main__":
    cli()
