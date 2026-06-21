Live Detection Backend Guide

This folder contains the core backend services, APIs, and prediction models for the road defect detection system.

1. config.py (Configuration Manager)
- Purpose: Globally manages project variables and settings.
- What it does: Automatically parses parameters from your project's .env file, including model weights file paths, confidence limits, directories, and server addresses.

2. detector.py (YOLO Wrapper)
- Purpose: Interface for loading and running the object detection model.
- What it does: Loads the custom trained YOLO model checkpoint (best.pt) onto the target GPU/CPU device. It processes input images, filters box overlaps, and outputs detection coordinates, confidence metrics, and class predictions.

3. midas_depth_estimator.py (Depth Estimator Wrapper)
- Purpose: Interface for generating monocular depth map frames.
- What it does: Downloads and caches PyTorch MiDaS model files on initialization. It executes depth inference on frames to output 2D relative depth arrays mapping closer vs farther pixel distances.

4. utils.py (Shared Mathematics & Fallbacks)
- Purpose: Houses coordinate formulas and hazard rating calculations.
- What it does:
  - compute_real_width: Calculates physical size in centimeters using camera variables and bounding box dimensions.
  - classify_severity: Generates severity index scores (0.0 to 1.0) and assigns category levels (Low, Medium, High, Critical) based on weighted parameters (50% depth, 30% width, 20% confidence).
  - compute_heuristic_measurements: Perspective fallback helper that estimates pothole dimension variables from frame coordinates if the neural depth model is unavailable.

5. calibration.py (Camera Alignment)
- Purpose: Calibrates camera pitch, height, and mounting setups.
- What it does: Aligns perspective calculations relative to vehicle mounting properties so raw pixel sizes can be translated into accurate centimeters.

6. storage_manager.py (Disk Cleaner & Storage)
- Purpose: Directs local upload caching and disk optimization.
- What it does: Saves raw uploads, processed image grids, and JSON reports. It spawns a background thread that periodically deletes temporary session files older than 24 hours to manage disk usage.

7. app.py (Flask REST API Server)
- Purpose: Serves static file processing routes.
- What it does: Serves endpoints for upload requests (/api/detect/image for static pictures, /api/detect/video for clips) and storage statistics requests.

8. live_ws.py (WebSocket Stream Server)
- Purpose: Serves real-time streaming connections.
- What it does: Serves a high-throughput websocket endpoint (ws://0.0.0.0:8765) that accepts streamed camera frames from the browser, processes YOLO/MiDaS detections, and returns real-time coordinate arrays.
