================================================================================
LIVEDET — BACKEND ARCHITECTURE & TECHNICAL GUIDE
================================================================================

This directory houses the core backend services, APIs, and prediction models
for the road defect detection system. It is designed around modular software 
engineering principles to handle concurrent streaming and real-time inference.

--------------------------------------------------------------------------------
1. SYSTEM DATA FLOW SEQUENCE
--------------------------------------------------------------------------------

   +--------------------------+
   |  React Client (Frontend) |
   +------------+-------------+
                │
                │ 1. Sends frame (HTTP POST or WebSocket stream)
                ▼
   +--------------------------+
   | Server Transport         | <--- (app.py / live_ws.py)
   +------------+-------------+
                │
                │ 2. Passes BGR image array
                ▼
   +--------------------------+
   | Pipeline Orchestrator    | <--- (depth_pipeline.py)
   +------------+-------------+
                │
                ├─► [3. Object Detection] ──► detector.py (YOLO inference)
                │                             (Returns 2D boxes [x,y,w,h])
                │
                ├─► [4. Neural Depth] ──────► midas_depth_estimator.py
                │                             (Returns scale-indeterminate map)
                │
                └─► [5. Geometry Scaling] ──► calibration.py
                                              (Applies pixel-to-cm ratio)
                │
                ▼
   +--------------------------+
   | Server Transport         |
   +------------+-------------+
                │
                │ 6. Calculates severity warning score & annotates BGR
                ▼
   +--------------------------+
   |  React Client (Frontend) | <--- (Triggers browser Web Audio Dual-Chirp)
   +--------------------------+

--------------------------------------------------------------------------------
2. DIRECTORY STRUCTURE & COMPONENT ROLES
--------------------------------------------------------------------------------

1. app.py (Flask REST API Server)
   - Path: backend/app.py
   - Purpose: Serves static file processing and analytics routes.
   - Key Handlers:
     * POST /predict: Processes uploaded static images, computes measurements,
       draws premium visual overlay HUD cards, and returns a JSON summary.
     * POST /video/process: Processes uploaded MP4 files frame-by-frame,
       saves the annotated frames using the H.264 codec, and outputs an MP4 URL.

2. live_ws.py (WebSocket Streaming Server)
   - Path: backend/live_ws.py
   - Purpose: Handles high-throughput live dashboard webcam video streams.
   - Key Handlers:
     * handler(): Receives binary frames over WS on port 8765, decodes them,
       runs YOLO + pipeline, and emits measurement packages.
     * Frame-Throttling: Uses "DEPTH_INTERVAL = 3" to execute monocular depth
       only on every 3rd frame to guarantee real-time latency (~28ms loop).

3. detector.py (Object Detector Wrapper)
   - Path: backend/detector.py
   - Purpose: Interface for loading and running the YOLO detector.
   - Key Handlers:
     * detect(): Converts frames to RGB, runs YOLO inference, filters boxes by
       confidence, and maps class labels (e.g. plain, pot -> Pothole).

4. midas_depth_estimator.py (Neural Depth & Blending)
   - Path: backend/midas_depth_estimator.py
   - Purpose: Wraps Intel MiDaS and implements hybrid fallback estimation.
   - Key Handlers:
     * estimate_pothole_depth(): Measures depth difference between the pothole
       bounding box crop and the road surface baseline (bottom-quarter slice).
     * HybridDepthEstimator: Blends MiDaS neural estimates with pixel shadow
       metrics if the neural model's confidence falls below 0.4.

5. calibration.py (Geometry & Custom Fallbacks)
   - Path: backend/calibration.py
   - Purpose: Translates 2D pixel bounding boxes into physical centimeters.
   - Key Handlers:
     * calibrate_from_marker(): HSV-filters a white 10cm reference marker on the
       road to compute pixels-per-cm scale ratio.
     * MonocularDepthEstimator: Custom fallback estimators (fusing shadow
       intensity difference, Laplacian texture roughness variance, and area size
       proxies) to evaluate depth when MiDaS is unavailable.

6. depth_pipeline.py (Pipeline Orchestrator)
   - Path: backend/depth_pipeline.py
   - Purpose: Main pipeline orchestrator class.
   - Key Handlers:
     * measure_pothole(): Coordinates detector outputs with calibration
       scaling and hybrid depth estimators to compile a unified result.

7. storage_manager.py (Disk Cleaner)
   - Path: backend/storage_manager.py
   - Purpose: Automatically manages disk space via an asynchronous background
     thread that purges temporary session files older than 10 minutes.

8. utils.py (Shared Mathematics & Helpers)
   - Path: backend/utils.py
   - Purpose: Holds severity calculations, BGR color charts, and base64 helper
     functions.

9. config.py (Config Manager)
   - Path: backend/config.py
   - Purpose: Automatically parses parameters and settings from your .env file.

--------------------------------------------------------------------------------
3. MODULE IMPORT DEPENDENCY MAP
--------------------------------------------------------------------------------

Below is the layout of how backend Python modules import and depend on each other:

  [app.py (REST)]       [live_ws.py (WebSockets)]
         │                         │
         ├─────────────────────────┤ (use detector wrappers)
         ▼                         ▼
  [detector.py]             [depth_pipeline.py]
         │                         │
         ▼ (config)                ├─► [midas_depth_estimator.py]
   [config.py]                     │
                                   └─► [calibration.py]
                                             │
                                             ▼ (loads scale configurations)
                                       [camera_calibration.json]

--------------------------------------------------------------------------------
4. HOW THE CORE TECHNICAL CONCEPTS WORK
--------------------------------------------------------------------------------

A. REFERENCE MARKER CALIBRATION (WIDTH SCALING)
A 10cm x 10cm physical white square is placed on the road surface.
1. The camera captures an image.
2. The HSV mask filters white regions.
3. cv2.findContours finds the square contour boundary:
   pixels_per_cm = Mean(width_px, height_px) / 10.0 cm
4. Any future bounding box width is scaled:
   width_cm = bbox_width_pixels / pixels_per_cm

B. MONOCULAR NEURAL DEPTH MAP (MIDAS)
1. The full image is input to MiDaS_small, producing relative depth map D (0.0-1.0).
2. The bottom-quarter horizontal slice of the depth map represents the road surface:
   Mean(D_road)
3. The average value inside the localized pothole box represents the cavity depth:
   Mean(D_pothole)
4. The relative depth is calculated as:
   depth_cm = (Mean(D_road) - Mean(D_pothole)) * 100

C. HEURISTIC FALLBACKS (SHADOW + TEXTURE + SIZE)
If MiDaS is unavailable, the pipeline blends three standard mathematical checks:
1. Shadow Intensity: depth_shadow = (brightness_road - brightness_pothole) / brightness_road * 100
2. Texture variance: Compares Laplacian variance between pothole and road (rougher surface = deeper).
3. Size proxy: depth_size = Sqrt(width_px * height_px) / 10.0
4. Fused Estimate:
   fused_depth = (depth_shadow * 0.5) + (depth_texture * 0.3) + (depth_size * 0.2)

D. SEVERITY SCORE FORMULA (utils.py)
When a pothole is detected, a severity score is computed as:
   Score = (depth_normalized * 0.50) + (width_normalized * 0.30) + (confidence * 0.20)
* Score < 0.30 -> Low (Green HUD card)
* 0.30 <= Score < 0.55 -> Medium (Yellow HUD card)
* 0.55 <= Score < 0.75 -> High (Orange HUD card)
* Score >= 0.75 -> Critical (Red HUD card + triggers browser ADAS warning beep)

--------------------------------------------------------------------------------
5. VIVA PRESENTATION / DEFENSE PREPARATION Q&A
--------------------------------------------------------------------------------

Q1: "Why did you use pre-trained models like MiDaS and YOLO instead of training
    your own models from scratch?"
A1: "As a software engineering student, my core research focus is the system
    integration, latency optimization, and calibration pipeline of the real-time
    warning system. Designing and training foundation models from scratch is a
    pure AI research task that requires thousands of GPU hours and millions of
    images. I utilized transfer learning to fine-tune YOLO11s on customized
    road-defect annotations, and utilized standard pre-trained weights for
    MiDaS to retrieve relative depth maps, focusing my engineering work on
    resolving the monocular scale ambiguity mathematically."

Q2: "How does your system resolve scale ambiguity? How do relative depth values
    become centimeters?"
A2: "Relative depth estimation models like MiDaS output scale-indeterminate
    maps. A value of 0.2 doesn't represent 20cm; it represents relative depth.
    I resolved this by designing a Reference Marker Calibration module. By
    detecting a known 10cm physical marker on the road, the system computes a
    pixels-per-centimeter ratio. For depth, the system calculates the
    differential between the relative depth of the road surface and the bottom
    of the cavity, converting that differential directly into physical
    centimeters."

Q3: "What happens at night or when the lighting conditions are very poor?"
A3: "In low-light or nighttime conditions, neural depth models like MiDaS
    suffer from a drop in confidence, and shadow intensity fallbacks become
    less reliable due to low gray-scale differentials. To handle this, the
    system uses a Hybrid Estimator. If the neural model's confidence drop below
    0.4, the pipeline switches to a blended fallback heuristic utilizing a
    size proxy and texture roughness metric, ensuring the driver still
    receives approximate severity notifications."
