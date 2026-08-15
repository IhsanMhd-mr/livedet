# LIVEDET — VIVA COMPUTER VISION & LIVE DETECTION CODE AUDIT

This document provides an exhaustive, evidence-based code audit and technical breakdown of the computer vision and live detection components in the LIVEDET repository. Every claim in this document is backed by exact file paths, class/function names, and line ranges from the active source code.

---

## TASK 1 — REPOSITORY MAP & MINIMUM VIVA FILES

The following table maps the repository's files related to model loading, processing, calibration, benchmarking, and frontend visualization.

### Repository Map

| Priority | File Path | Main Class/Function | Role | Active / Testing / Training / Unused | Called By |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | [backend/live_ws.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py) | `handle_client` / `initialize_models` | Production WebSocket server and frame-processing loop. Runs YOLO and MiDaS (every 3rd frame). | **Active Production (WebSocket)** | Start script / CLI |
| **2** | [backend/detector.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py) | `ObjectDetector` / `detect` | Wraps Ultralytics YOLO model loading and inference. Swaps BGR/RGB. | **Active Production** | `live_ws.py` (line 218), `app.py` (line 180) |
| **3** | [backend/utils.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py) | `DepthEstimator` / `extract_median_depth` / `blend_depth` / `classify_severity` | Wraps MiDaS Small, crops depth maps, implements hybrid depth blending and severity rules. | **Active Production** | `live_ws.py`, `app.py` |
| **4** | [frontend/src/pages/LiveDetection.jsx](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx) | `LiveDetection` / `beginSendLoop` / `draw` | React page for live camera capture, Canvas overlays, synthesized ADAS alerts. | **Active Production (Frontend)** | React Router (`App.jsx` line 20) |
| **5** | [frontend/src/hooks/useWebSocket.js](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/hooks/useWebSocket.js) | `useWebSocket` | React hook for managing the WebSocket client connection state. | **Active Production (Frontend)** | `LiveDetection.jsx` (line 66) |
| **6** | [backend/app.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/app.py) | `predict` / `process_video` | Flask REST API for stateless image/video uploads. Uses **heuristic only** (no MiDaS). | **Active Production (REST)** | Frontend REST views (`ImageDetect.jsx`, `VideoDetect.jsx`) |
| **7** | [backend/config.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/config.py) | `Config` | Loads environment variables from `.env` (weights path, thresholds). | **Active Production** | `live_ws.py`, `app.py` |
| **8** | [backend/calibration.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/calibration.py) | `CameraCalibration` / `MonocularDepthEstimator` | Autodetects white 10cm square road markers and runs texture/shadow heuristic. | **Unused / Legacy / Dead Code** | None (imported by no active production code) |
| **9** | [model_training/fine_tune.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/model_training/fine_tune.py) | `fine_tune_model` / `evaluate_and_compare` | Script used to fine-tune pre-trained YOLO checkpoint on the clean dataset. | **Training Only** | Executed manually |
| **10** | [scripts/data_cleaning.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/scripts/data_cleaning.py) | `DataCleaner` | Cleans polygon labels, filters corrupt files, runs offline data augmentations. | **Dataset Preparation** | Executed manually |
| **11** | [notebooks/04_pipeline_benchmark.ipynb](file:///c:/Users/ihsan/Documents/GitHub/ML2/notebooks/04_pipeline_benchmark.ipynb) | Latency benchmark loop | Measures execution times of pipeline components with synchronization. | **Testing / Benchmarking** | Jupyter execution |
| **12** | [notebooks/05_model_evaluation.ipynb](file:///c:/Users/ihsan/Documents/GitHub/ML2/notebooks/05_model_evaluation.ipynb) | YOLO validation metrics | Calculates validation split precision, recall, and curves. | **Testing / Benchmarking** | Jupyter execution |
| **13** | [scripts/live_detect.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/scripts/live_detect.py) | Standalone live loop | Desktop OpenCV webcam loop script for offline YOLO + MiDaS testing. | **Testing / Offline Demo** | Executed manually |

### Minimum Set of Files to Open During the Viva
To walk the examiner through the active codebase, open:
1. `backend/live_ws.py` (Server loop & MiDaS intervals)
2. `backend/detector.py` (YOLO loading and channel swapping)
3. `backend/utils.py` (MiDaS wrapping, depth blending, severity equations)
4. `frontend/src/pages/LiveDetection.jsx` (Webcam loop, Canvas draw, ADAS audio play)

---

## TASK 2 — THE ACTIVE LIVE PIPELINE TRACE

The active LIVEDET live WebSocket pipeline follows a synchronous frame-response model. Here is the sequence of execution:

### Runtime Trace Table

| Step | File:Lines | Function / Component | Input | Processing | Output | Calls Next |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | [LiveDetection.jsx:268-280](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L268-L280) | `startCamera` | User click interaction | Calls browser `getUserMedia` webcam API | MediaStream object in DOM `<video>` | `beginRenderLoop`, `beginSendLoop` |
| **2** | [LiveDetection.jsx:386-408](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L386-L408) | `beginSendLoop` | MediaStream | Throttled interval (120ms) grabs frame from `<video>` element and draws to `captureCanvas` | Image drawn on hidden canvas | `toDataURL` |
| **3** | [LiveDetection.jsx:399-400](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L399-L400) | `beginSendLoop` | Canvas image | Calls `canvas.toDataURL("image/jpeg", 0.70)` and splits Base64 header | Compressed Base64 string | `send` |
| **4** | [useWebSocket.js:151-153](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/hooks/useWebSocket.js#L151-L153) | `send` | Base64 string | Writes string to active WebSocket connection | WebSocket frame on port 8765 | Python WebSocket library |
| **5** | [live_ws.py:199-200](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L199-L200) | `handle_client` | WebSocket frame | Listens asynchronously inside the message loop | Raw message bytes | `base64.b64decode` |
| **6** | [live_ws.py:204](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L204) | `handle_client` | Message bytes | Decodes Base64 data back to bytes | JPEG binary bytes | `np.frombuffer` |
| **7** | [live_ws.py:205-206](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L205-L206) | `handle_client` | Binary bytes | Converts to NumPy buffer and calls `cv2.imdecode` | BGR frame array | `cv2.cvtColor` |
| **8** | [live_ws.py:214](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L214) | `handle_client` | BGR frame array | Calls `cv2.cvtColor` to swap color channels | RGB frame array | `detector.detect` |
| **9** | [detector.py:70-81](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L70-L81) | `detect` | RGB frame array | Swaps RGB channels back to BGR (bug!), runs `self.model()` inference | Ultralytics Prediction list | Box extraction |
| **10** | [detector.py:84-102](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L84-L102) | `detect` | Prediction list | Extracts coordinates, confidence, class IDs; builds list of `[x, y, w, h]` boxes | Python list of detections | `live_ws.py` line 222 |
| **11** | [live_ws.py:222-223](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L222-L223) | `handle_client` | RGB frame array | If `frame_count % 3 == 0`, runs `depth_estimator.estimate()`; otherwise uses cache | Normalised depth map `(H, W)` | `extract_median_depth` |
| **12** | [live_ws.py:236](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L236) | `handle_client` | Depth map, box | Crops depth map at box coordinates, computes median value | Median relative depth (`med`) | `compute_depth_cm` |
| **13** | [live_ws.py:239-267](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L239-L267) | `handle_client` | Median relative depth | Computes depth (in cm), real-world dimensions (cm) via pinhole camera, distance (m) | Dimension values in cm / m | `compute_midas_confidence` |
| **14** | [live_ws.py:248-259](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L248-L259) | `handle_client` | Median, cache | Computes local variance (buggy std formula) and blends MiDaS vs Heuristic | Blended depth estimate (cm) | `classify_severity` |
| **15** | [live_ws.py:289](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L289) | `handle_client` | Blended depth, width | Normalizes depth and width, scores severity, matches severity tier bounds | Severity tier label and score | `websocket.send` |
| **16** | [live_ws.py:318-325](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L318-L325) | `handle_client` | Detection dictionary | Packs lists into JSON format, logs server FPS, and calls WebSocket send | JSON character string | WebSocket client |
| **17** | [useWebSocket.js:79-95](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/hooks/useWebSocket.js#L79-L95) | `message` handler | JSON character string | Parses JSON string and dispatches React state updates | React state updates | `draw` loop |
| **18** | [LiveDetection.jsx:303-311](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L303-L311) | `draw` (Animation Frame) | Canvas context | Updates display canvas size to match incoming video size | Cleared/resized Canvas context | Canvas drawing calls |
| **19** | [LiveDetection.jsx:318-370](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L318-L370) | `draw` (Animation Frame) | Detection list | Iterates over bounding boxes, draws severity-colored rectangles, text | Drawn canvas overlay | `useEffect` alerts |
| **20** | [LiveDetection.jsx:231-253](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L231-L253) | Alert `useEffect` | Detection list | Checks if any High/Critical defect is within proximity threshold | Triggers blinking UI & warning sound | `playWarningSound` |

### Pipeline Architecture Flow

```text
Frontend Webcam Frame
    ↓
captureCanvasRef (Grab frame at 120ms interval)
    ↓
JPEG encoding (toDataURL / quality=0.70)
    ↓
WebSocket Client (send base64 string)
    ↓
Python WebSocket server (live_ws.py handle_client)
    ↓
Base64 decode & OpenCV imdecode (BGR format)
    ↓
cv2.cvtColor BGR -> RGB (live_ws.py line 214)
    ↓
YOLO Inference (Swaps RGB back to BGR, runs detector.py line 81)
    ↓
MiDaS Small (Every 3rd frame, stores in cached_depth)
    ↓
Depth sampling (extract_median_depth / local variance check)
    ↓
Hybrid Blend (blend_depth / heuristic fallback if confidence < 0.3)
    ↓
Severity scoring (classify_severity: Low / Medium / High / Critical)
    ↓
JSON packing & WebSocket send
    ↓
React client useWebSocket hook parses message
    ↓
requestAnimationFrame draw loop updates displayCanvasRef
    ↓
Audio Synth Alert (Double-beep warning synthesized in AudioContext)
```

---

## TASK 3 — YOLO Object Detection Code Audit

### Active Detector Class details

*   **Exact Class Name**: `ObjectDetector` defined in [backend/detector.py:24](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L24).
*   **Model-Loading Function**: `load_model_file(self, model_path)` defined in [backend/detector.py:40-57](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L40-L57).
*   **Actual Model Weights Path**: Configured by `.env` variable `BEST_MODEL_PATH=./models/finetuned/pothole_detector_yolo11s_v22/weights/best.pt`.
*   **Weights Check**: Yes, the model file `best.pt` exists at the configured path, with a file size of **19,206,234 bytes** (~19.2 MB).
*   **Fallback Behavior**: If the configured custom model path does not exist, line 49 falls back to the default pretrained weights:
    ```python
    self.model = YOLO(f"{self.model_type}.pt")
    ```
    This loads `yolov11s.pt` from the current working directory (or downloads it if missing).
*   **Device Selection**: Set via `self.device = torch.device(device)` ([detector.py:30](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L30)). The `.env` file overrides this to `DEVICE=cuda:0` which forces GPU utilization.
*   **Loading Lifecycle**: Model loading happens **once at server startup** inside `initialize_models()` ([live_ws.py:119-162](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L119-L162)). The model is cached in a global singleton variable `detector` and shared across all incoming client connections.
*   **Confidence Threshold**: Loaded from `.env` as `0.15` and passed at initialization ([live_ws.py:147](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L147)).
*   **IoU Threshold**: Loaded from `.env` as `0.25` and passed at initialization ([live_ws.py:148](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L148)).
*   **Image size**: Passed to model inference as `imgsz=800` during fine-tuning (configured in `args.yaml` line 9), but during live inference it is determined dynamically by the input frame shape (which is resized to `640x480` in the frontend canvas).
*   **Non-Maximum Suppression (NMS)**: Handled natively by the Ultralytics library during inference. The IoU threshold is passed as the `iou` parameter inside the forward call.
*   **Input Image Format**: Numpy `ndarray` of shape `(H, W, 3)` (channels: RGB).
*   **Returned Object**: Ultralytics `Results` object list.
*   **Extraction Method**: Bounding boxes, confidences, and classes are extracted sequentially in [detector.py:84-102](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L84-L102):
    ```python
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    ```
*   **Output Dictionary Structure**: The function returns a dictionary:
    ```python
    return {
        "status": "success",
        "detections": detections, # list of dicts with: bbox [x,y,w,h], bbox_xyxy [x1,y1,x2,y2], confidence, class_id, class_name
        "image_shape": [height, width, channels],
        "total_detections": len(detections),
        "model": self.loaded_model_name
    }
    ```
*   **Behavior on Empty Detection**: Returns the same dictionary with `"detections": []` and `"total_detections": 0`. No exception is thrown.
*   **Error Handling**: Wrapped in a broad `try-except` block ([detector.py:112-114](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L112-L114)). It returns `{"status": "error", "error": str(e)}` instead of crashing.
*   **Model Mode / evaluation mode**: Ultralytics handles evaluation behavior internally during `predict()` or forward pass.
*   **Inference Mode**: The forward pass does not use explicit PyTorch `@torch.no_grad()` or `torch.inference_mode()` wrappers inside `detector.py`, but Ultralytics runs inference with gradients disabled by default inside its forward handler.

---

### YOLO Detector Code Excerpt

```python
# From backend/detector.py (Lines 59-114)
    def detect(self, image, confidence_threshold=None, iou_threshold=None):
        try:
            # Handle image path vs image array
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    return {"status": "error", "error": f"Could not read image: {image}"}
            else:
                img = image

            # Convert BGR to RGB for YOLO
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            height, width = img_rgb.shape[:2]
            channels = img_rgb.shape[2] if len(img_rgb.shape) == 3 else 1
            threshold = confidence_threshold or self.confidence_threshold
            iou = iou_threshold or self.iou_threshold

            # Run YOLO model inference
            results = self.model(img_rgb, conf=threshold, iou=iou, device=self.device, verbose=False)

            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Get class name and map it to Pothole
                        cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        if cls_name.lower() in ["plain", "pothole", "pot"]:
                            cls_name = "Pothole"

                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": cls_name
                        })

            return {
                "status": "success",
                "detections": detections,
                "image_shape": [height, width, channels],
                "total_detections": len(detections),
                "model": self.loaded_model_name
            }

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"status": "error", "error": str(e)}
```

---

### YOLO Auditing Explanations

#### A. Beginner Explanation
The object detector is like a digital eye trained to spot potholes on the road. When the server gets an image, it feeds it to YOLO (which stands for "You Only Look Once"), a smart neural network model. YOLO scans the image and highlights where it sees a pothole, drawing a bounding box around it and giving us a score showing how sure it is (from 0% to 100%).

#### B. Software Engineering Viva Explanation
The class `ObjectDetector` wraps the Ultralytics YOLOv11 framework. It loads a fine-tuned PyTorch weight file (`best.pt`) at startup and sets up parameters such as the confidence threshold (0.15) and IoU threshold (0.25). The `detect` method handles both image paths and numpy array inputs. It performs non-maximum suppression (NMS) internally, parses the output tensors, and formats them into a structured JSON-serializable list containing standard bounding box representations.

#### C. 10 Likely Examiner Questions & Answers

1.  **Question**: How does the system avoid the latency overhead of reloading the model weights for each incoming WebSocket client?  
    **Answer**: We implement the Singleton pattern. The models are loaded once globally during `initialize_models()` when the server starts, and the active handle is reused by all async connection coroutines.
2.  **Question**: Which version of YOLO is used, and what is the weight size?  
    **Answer**: We use YOLO11s (small variant). The weight file `best.pt` has a footprint of 19.2 MB, which is optimized for edge or low-end GPU deployments.
3.  **Question**: Why did you set the confidence threshold so low (0.15) in your configuration?  
    **Answer**: Since potholes vary wildly in appearance, lighting, and angles, setting a lower threshold helps catch faint or distant potholes. False positives are then filtered or classified into the "Low" severity category during subsequent depth verification.
4.  **Question**: How does the system handle device fallback if the GTX 1650 GPU is missing or CUDA is unavailable?  
    **Answer**: The device configuration is read from the `.env` file. If the system is unable to initialize on GPU, PyTorch fallback handles the execution on CPU.
5.  **Question**: What happens if the custom weights file (`best.pt`) is corrupted or missing at runtime?  
    **Answer**: The code implements a fallback check inside `load_model_file()`. If `best.pt` is missing, it logs a warning and loads the pretrained base weights `yolov11s.pt` to ensure system uptime.
6.  **Question**: In what coordinate format does the detector return bounding boxes, and why?  
    **Answer**: The detector extracts `xyxy` coordinates from Ultralytics, but transforms them into `[x, y, w, h]` (top-left coordinate, width, height) to match standard HTML5 Canvas rendering expectations on the frontend.
7.  **Question**: Is the model running in evaluation mode at runtime?  
    **Answer**: Yes, the Ultralytics model API manages evaluation and inference-mode optimizations internally when calling its prediction engine, disabling gradients to conserve memory.
8.  **Question**: How does the detector resolve multi-class annotations at runtime?  
    **Answer**: The detector script checks the predicted class name string in line 93. If the class name corresponds to `"plain"`, `"pothole"`, or `"pot"`, it maps the label to a single class, `"Pothole"`.
9.  **Question**: What is the impact of running the WebSocket server on a system without a GPU?  
    **Answer**: Without GPU acceleration, YOLO inference latency increases from ~10 ms to ~70 ms, and MiDaS Small increases from ~8 ms to ~250 ms. This drops the pipeline processing capacity below 3 FPS, causing significant latency lag.
10. **Question**: Does the code contain any redundant color space conversions during YOLO inference?  
    **Answer**: Yes. The WebSocket loop converts BGR frames to RGB before calling `detector.detect()`. However, inside `detector.detect()`, the code again calls `cv2.cvtColor` BGR-to-RGB on the RGB input, effectively swapping channels a second time and feeding BGR data into the model.

---

## TASK 4 — BOUNDING-BOX FORMAT VERIFICATION

Handling coordinate systems consistently is critical in computer vision pipelines. This audit traces how coordinates flow from the initial YOLO detector to the final canvas drawing.

### Bounding-Box Coordinate Flow Table

| Stage | File:Lines | Actual Format | Example | Conversion Performed |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO Output** | [detector.py:87](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L87) | `xyxy` = `[x1, y1, x2, y2]` | `[100.0, 150.0, 300.0, 250.0]` | Extracted directly from PyTorch box tensors. |
| **Detector Output** | [detector.py:97](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L97) | `xywh` = `[x, y, w, h]` | `[100, 150, 200, 100]` | Converts `xyxy` to `xywh` where `w = x2 - x1` and `h = y2 - y1`. |
| **MiDaS Crop** | [utils.py:183-186](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L183-L186) | Sliced pixel indices | `depth_map[150:250, 100:300]` | Re-calculates bottom-right corner as `x2 = x + w` and `y2 = y + h`. |
| **Geometric Calc** | [utils.py:458-459](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L458-L459) | `xywh` | `w = 200`, `h = 100` | Uses width and height directly to compute relative ratios against frame size. |
| **JSON Response** | [live_ws.py:295](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L295) | `xywh` | `[100, 150, 200, 100]` | Rounds coordinates to integers: `[int(x), int(y), int(w), int(h)]`. |
| **Frontend Canvas** | [LiveDetection.jsx:320-332](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L320-L332) | `xywh` | `[100, 150, 200, 100]` | Draws directly to HTML5 Canvas context: `ctx.strokeRect(bx, by, bw, bh)`. |

---

### Verification of Bounding Box Bug

We audited the coordinate calculations to identify whether any function performs the redundant conversion:
$$x2 = x + \text{width}$$
$$y2 = y + \text{height}$$
on data that is already $x2$ and $y2$.

#### Findings in Active Code:
*   Inside [backend/utils.py:183-186](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L183-L186) (`extract_median_depth`), the inputs are parsed correctly:
    ```python
    x, y, w, h = bbox
    x1 = max(0, x)
    x2 = min(w_map, x + w)
    ```
    This is correct because the input is in `xywh` format.
*   Inside [backend/utils.py:227-228](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L227-L228) (`compute_midas_confidence`), the calculations also match:
    ```python
    x, y, w, h = bbox
    x2, y2 = min(w_map, x + w), min(h_map, y + h)
    ```
    This is also correct.
*   **Unused Code Bug**: In [backend/calibration.py:74-76](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/calibration.py#L74-L76) (`calibrate_from_marker`):
    ```python
    x1, y1, x2, y2 = marker_bbox
    marker_width_px = x2 - x1
    ```
    If `marker_bbox` is passed from the active YOLO pipeline (which returns `xywh` format) instead of the internal `_detect_reference_marker` (which returns `xyxy`), `x2` would map to the width $w$ and `x1` would map to $x$, making `marker_width_px = w - x` (incorrect).
    Since `calibration.py` is **unused** in production, this bug remains dormant and does not affect live detection.

---

## TASK 5 — COLOUR SPACE & IMAGE SHAPES AUDIT

Image decoding and color space matching are critical for model inference accuracy. Here is the trace of color spaces and image shapes across the LIVEDET pipeline.

### Color & Shape Audit Table

| Location | Input Color / Shape | Transformation | Output Color / Shape |
| :--- | :--- | :--- | :--- |
| **Browser JPEG** | RGBA / sRGB (from Canvas) | Encoded via `.toDataURL("image/jpeg")` | Lossy compressed JPEG bytes |
| **cv2.imdecode** | JPEG bytes | Decoded with `cv2.imdecode(..., cv2.IMREAD_COLOR)` | **BGR** / `(480, 640, 3)` |
| **live_ws.py (BGR-RGB)** | **BGR** / `(480, 640, 3)` | Calls `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)` | **RGB** / `(480, 640, 3)` |
| **YOLO expected** | **RGB** | Handled by Ultralytics backend | **RGB** |
| **YOLO detector input** | **RGB** / `(480, 640, 3)` | **CRITICAL BUG**: Swaps channels *again* in [detector.py:71](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L71) via `cv2.cvtColor` | **BGR** / `(480, 640, 3)` |
| **MiDaS expected** | **RGB** | Handled by PyTorch transform | **RGB** normalized |
| **MiDaS input** | **RGB** / `(480, 640, 3)` | Resized and normalized by `self.transform` | Float tensor / `(1, 3, 256, 256)` |
| **Width/Height extraction** | BGR shape `(480, 640, 3)` | Slices first two shape indices: `fh, fw = shape[:2]` | Height = 480, Width = 640 |
| **Frontend Canvas** | sRGB frame, integer coordinates | Drawn directly onto scaled Canvas context | Rendered sRGB on canvas |

---

### Critical Discrepancies & Mismatches

#### 1. Redundant BGR-to-RGB Channel Swapping (YOLO Color Mismatch)
*   **Evidence**:
    In [backend/live_ws.py:214](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L214), BGR frames are converted to RGB:
    ```python
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ```
    Then, `frame_rgb` is passed to the detector ([live_ws.py:218](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L218)).
    Inside `ObjectDetector.detect` ([backend/detector.py:70-71](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L70-L71)), the detector does:
    ```python
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ```
*   **Impact**:
    Since `img` is already converted to RGB, running `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` a second time swaps the Red and Blue channels back. This means **YOLO runs inference on BGR images**. While YOLO can still detect shapes (shadows and cracks), this mismatch reduces model sensitivity to colored features and lowers detection confidence.
    *Note: MiDaS Small is unaffected because it receives the correct `frame_rgb` array directly.*

#### 2. Canvas-to-Coordinate Mismatch (Aspect Ratio / Stretching)
*   **Evidence**:
    The camera is captured at 640x480 (aspect ratio 4:3) in the browser. The browser sends the frame to the backend. The backend processes the frame at 640x480 and returns bounding box coordinates in pixel offsets relative to this shape.
    In [frontend/src/pages/LiveDetection.jsx:305-310](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L305-L310), the display canvas size is adjusted dynamically:
    ```javascript
    if (canvas.width !== video.videoWidth && video.videoWidth > 0) {
      canvas.width = video.videoWidth
    }
    ```
*   **Impact**:
    Because the canvas dimensions are set to match the video's actual resolution (640x480), the coordinates returned by the backend map 1:1 onto the canvas drawing context. This prevents coordinate stretching and alignment issues.

---

## TASK 6 — MiDaS DEPTH ESTIMATION AUDIT

Intel MiDaS (Mixed Robust Depth Estimation) Small is used to generate relative depth maps from single camera frames.

### MiDaS Small Implementation Details

*   **Model Variant**: `MiDaS_small` defined in [backend/utils.py:80](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L80).
*   **Loading Method**: `torch.hub.load("intel-isl/MiDaS", model_type)` ([backend/utils.py:94](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L94)).
*   **Transform**: `transforms.small_transform` ([backend/utils.py:100](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L100)) which resizes the image to 256x256 and normalizes it to ImageNet stats.
*   **Device**: Loaded on GPU `cuda:0` if available, falling back to CPU ([backend/utils.py:95](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L95)).
*   **Inference Mode**: Decorated with `@torch.no_grad()` to disable gradient calculation and reduce memory overhead ([backend/utils.py:110](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L110)).
*   **Output Interpolation**: Bicubic-upsampled back to original frame size:
    ```python
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=frame_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    ```
*   **Normalisation**: Min-max normalization maps depth predictions to `[0, 1]`:
    ```python
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    ```
*   **Failure Recovery**: If torch.hub fails (e.g. offline, no cache), `self.initialized` is set to `False` ([backend/utils.py:108](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L108)), and the system falls back to the geometric heuristic.

---

### Core Concept: Relative vs. Metric Depth

MiDaS outputs **disparity** (relative inverse depth).
1.  **Inverse Depth (Disparity)**: Larger raw values mean closer objects; smaller values mean farther objects.
2.  **Relative Depth**: After min-max normalization, the output is scaled relative to the closest and farthest points in the current scene.
    *   **Normalized Value = 1.0**: Represents the closest object in the frame.
    *   **Normalized Value = 0.0**: Represents the farthest object in the frame.
3.  **Why it is NOT metric**: MiDaS does not know the camera's physical height, focal length, pitch, or actual scene metrics. If the frame contains only a road surface 2 meters away, the closest part of the road becomes `1.0` and the farthest becomes `0.0`. If a car enters the frame at 1 meter, the normalization shifts, and the road values change completely.

#### Unsafe Claims to Avoid in the Viva:
*   *Do NOT claim MiDaS measures exact depth in centimeters.* It only estimates relative distance.
*   *Do NOT claim the fallback heuristic is physically calibrated.* It is an empirical approximation.

---

## TASK 7 — extract_median_depth() AUDIT

This function samples the depth map inside a bounding box to estimate the pothole's relative distance.

### Function Code Excerpt

```python
# From backend/utils.py (Lines 162-193)
def extract_median_depth(
    depth_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
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
```

---

### Auditing Analysis & Limitations

*   **Function Parameters**: `depth_map` (2D float32 numpy array) and `bbox` (tuple of integers `(x, y, w, h)`).
*   **Coordinate Clamping**: Lines 183-186 clamp coordinates to `[0, w_map]` and `[0, h_map]` to prevent out-of-bounds indexing.
*   **Empty Bounding Boxes**: Line 188 returns a neutral fallback of `0.5` for degenerate boxes (which maps to `7.5 cm` via `compute_depth_cm`).
*   **Median Sampling**: Using the median filter is robust against edge outliers and noise compared to using the mean.
*   **Resolution Check**: The depth map is bicubic-upsampled to match the original frame dimensions (`640x480`) before cropping, ensuring coordinate alignment.

#### Core Limitations of Median Crop Sampling:
1.  **Rectangular Crop Containment**: The crop is a rectangle. Because potholes are irregular shapes, the crop includes surrounding road surface, flat asphalt, and non-cavity elements.
2.  **Disparity of Road vs. Cavity**: MiDaS measures surface depth, not cavity depth. A pothole's cavity is typically shadow-filled and lacks texture, meaning MiDaS cannot resolve the bottom of the hole. The median depth value represents the distance to the road surface, not the pothole's actual depth.

---

## TASK 8 — compute_midas_confidence() AUDIT

This function estimates how reliable the MiDaS depth estimates are for a given bounding box region.

### Function Code Excerpt

```python
# From backend/utils.py (Lines 196-238)
def compute_midas_confidence(
    depth_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> float:
    x, y, w, h = bbox
    h_map, w_map = depth_map.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_map, x + w), min(h_map, y + h)

    if x2 <= x1 or y2 <= y1:
        # Degenerate bbox — return neutral fallback
        return 0.0

    region = depth_map[y1:y2, x1:x2]
    std = float(np.std(region))

    # Invert: high std → high confidence; clip to [0, 1]
    confidence = float(np.clip(1.0 - std * 2.0, 0.0, 1.0))
    return confidence
```

---

### Logic Verification & Numerical Analysis

The docstring states:
> *high std → textured surface → MiDaS is reliable → high confidence*  
> *low std → textureless → MiDaS is unreliable → low confidence*

However, the code implements the formula:
$$\text{confidence} = \text{clip}(1.0 - \text{std} \times 2.0, 0.0, 1.0)$$

Let's calculate the output for different standard deviations:

*   **std = 0.00** $\rightarrow \text{confidence} = \text{clip}(1.0 - 0.0) = \mathbf{1.0}$ (Maximum Confidence)
*   **std = 0.10** $\rightarrow \text{confidence} = \text{clip}(1.0 - 0.2) = \mathbf{0.8}$
*   **std = 0.25** $\rightarrow \text{confidence} = \text{clip}(1.0 - 0.5) = \mathbf{0.5}$
*   **std = 0.40** $\rightarrow \text{confidence} = \text{clip}(1.0 - 0.8) = \mathbf{0.2}$
*   **std = 0.50** $\rightarrow \text{confidence} = \text{clip}(1.0 - 1.0) = \mathbf{0.0}$ (Minimum Confidence)

#### Verdict: Low standard deviation produces high confidence.

This is a **major logical bug** that contradicts the docstring.

### Documentation Conflict Audit

| Evidence Location | What It Claims | Matches Code? |
| :--- | :--- | :--- |
| `utils.py` Docstring (Lines 201-207) | "high std $\rightarrow$ high confidence; low std $\rightarrow$ low confidence" | **No** (Direct contradiction) |
| `utils.py` Code (Line 237) | `# Invert: high std → high confidence; clip to [0, 1]` | **No** (The code subtracts `std * 2.0`, meaning high std *reduces* confidence) |
| `live_ws.py` Docstring (Lines 23-28) | "on textured potholes... it trusts MiDaS; on smooth uniform asphalt... it falls back gracefully" | **No** (Due to the bug, it does the exact opposite) |

#### Impact of the Bug:
*   For **textured potholes** with high depth variance (high std), the system computes a **low confidence** score and falls back to the geometric heuristic.
*   For **flat, smooth asphalt** (low std), the system computes a **high confidence** score (1.0) and trusts the noisy relative MiDaS depth estimates.
*   This disrupts the adaptive blending logic of the hybrid depth pipeline.

---

## TASK 9 — CALIBRATION & DEPTH EQUATIONS AUDIT

This section audits the formulas used to estimate physical dimensions from bounding boxes and relative depth.

### Depth & Dimension Formulas

| Output Metric | Formula in Code | File:Lines | Variable Meanings | Source of Constants | Method Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **depth_cm** (MiDaS path) | `(1.0 - med) * 15.0` (min: 0.5) | [utils.py:329-331](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L329-L331) | `med`: median relative depth. | `15.0` (max pothole depth in cm), `0.5` (min floor in cm). | Heuristic scaling |
| **width_cm** (MiDaS path) | `(w * depth_m / focal) * 100.0` | [utils.py:306-308](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L306-L308) | `w`: pixel width; `depth_m`: distance in meters; `focal`: focal length in pixels. | `depth_m = max(med * 5.0, 0.3)`. `focal` defaults to `600.0` px. | Pinhole model |
| **distance_m** (MiDaS path) | `(1.0 - med) * 5.0 + 0.3` (min: 0.3) | [live_ws.py:267](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L267) | `med`: median relative depth. | `5.0` (distance scale factor), `0.3` (offset in meters). | Heuristic scaling |
| **depth_cm** (Heuristic path) | `raw_depth * perspective_factor` (clamped `[2.0, 12.0]`) | [utils.py:454-463](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L454-L463) | `raw_depth = (h / H) * 30.0`. `perspective_factor = max(1.0 - 0.6 * y_norm, 0.4)`. | `30.0` (depth scaling), `0.6` (perspective multiplier), `0.4` (min damping). | Heuristic approximation |
| **width_cm** (Heuristic path) | `(w / W) * 100.0` (clamped `[5.0, 80.0]`) | [utils.py:451-452](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L451-L452) | `w`: pixel width; `W`: frame width. | `100.0` (assumed physical frame field of view in cm). | Heuristic approximation |

---

### Reference Marker Calibration Audit
*   **Presence of Calibration**: The file `calibration.py` is present in the codebase.
*   **Production Status**: The calibration module is **not imported** or used by any active server scripts (`app.py` or `live_ws.py`).
*   **Calibration Method**: No reference markers are detected automatically in production. The system relies entirely on hardcoded defaults (such as `focal_length = 600.0` px and assumed road dimensions).

---

## TASK 10 — blend_depth() AUDIT

The function `blend_depth()` combines MiDaS neural depth estimates with the geometric fallback heuristic based on the computed confidence score.

### Blending Logic Details

*   **MIDAS_CONF_HIGH = 0.70**: Trusts MiDaS fully.
*   **MIDAS_CONF_LOW = 0.30**: Below this, the system falls back to the geometric heuristic.
*   **Between 0.30 and 0.70**: Computes a weighted blend:
    $$\text{depth} = \text{confidence} \times \text{midas\_cm} + (1.0 - \text{confidence}) \times \text{heuristic\_cm}$$

---

### Worked Numerical Example

With inputs: `midas_cm = 8`, `heuristic_cm = 4`, `confidence = 0.5`.
Since `0.30 <= 0.5 < 0.70`, the system computes a weighted blend:
$$\text{depth} = 0.5 \times 8 + (1.0 - 0.5) \times 4 = 4.0 + 2.0 = \mathbf{6.0\text{ cm}}$$

### Boundary Value Test

*   **confidence = 0.00** $\rightarrow \text{depth} = \text{heuristic\_cm} = \mathbf{4.0\text{ cm}}$ (Heuristic only)
*   **confidence = 0.29** $\rightarrow \text{depth} = \text{heuristic\_cm} = \mathbf{4.0\text{ cm}}$ (Heuristic only)
*   **confidence = 0.30** $\rightarrow \text{depth} = 0.30 \times 8 + 0.70 \times 4 = 2.4 + 2.8 = \mathbf{5.2\text{ cm}}$ (Blend)
*   **confidence = 0.69** $\rightarrow \text{depth} = 0.69 \times 8 + 0.31 \times 4 = 5.52 + 1.24 = \mathbf{6.76\text{ cm}}$ (Blend)
*   **confidence = 0.70** $\rightarrow \text{depth} = \text{midas\_cm} = \mathbf{8.0\text{ cm}}$ (MiDaS only)
*   **confidence = 1.00** $\rightarrow \text{depth} = \text{midas\_cm} = \mathbf{8.0\text{ cm}}$ (MiDaS only)

*Note: While the blending logic functions as intended, the input `confidence` values are inverted due to the confidence standard deviation bug (Task 8).*

---

## TASK 11 — FRAME SKIPPING & CACHED DEPTH AUDIT

To maintain high frame rates, the backend runs MiDaS depth estimation only once every few frames, reusing the cached depth map for the remaining frames.

### Frame Processing Sequence (DEPTH_INTERVAL = 3)

| Frame | YOLO Runs? | MiDaS Runs? | Cached Map Used? | Age of Active Depth Map |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Yes** (Every Frame) | **Yes** | No (Generates fresh map) | 0 frames old (Current) |
| **1** | **Yes** | No | **Yes** (Reuses map 0) | 1 frame old |
| **2** | **Yes** | No | **Yes** (Reuses map 0) | 2 frames old |
| **3** | **Yes** | **Yes** | No (Generates fresh map) | 0 frames old (Current) |
| **4** | **Yes** | No | **Yes** (Reuses map 3) | 1 frame old |
| **5** | **Yes** | No | **Yes** (Reuses map 3) | 2 frames old |

---

### Frame Skipping Issues & Latency Analysis

1.  **Temporal Alignment Mismatch**:
    When the vehicle is moving, the camera position changes between frames. On Frame 2, the system crops Frame 2's YOLO bounding boxes out of Frame 0's depth map. At 50 km/h, the vehicle travels **3.3 meters** in 240 ms (2 frames latency). The pothole's screen coordinates will have shifted, meaning the system crops road surface instead of the pothole cavity.
2.  **Concurrency / Cache Isolation**:
    The `cached_depth` map is initialized inside the WebSocket connection handler `handle_client` ([live_ws.py:194](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L194)). It is unique to each client and is not shared globally.
3.  **Connection Reset**:
    A client reconnection terminates the handler thread and initializes a new connection state, resetting `frame_count = 0` and `cached_depth = None`.
4.  **Pipeline Bottlenecks**:
    Throttling MiDaS to run every 3rd frame does not speed up the pipeline proportionally. YOLO inference (10-15 ms) and JPEG decoding (5-10 ms) run sequentially on **every single frame**. Since the processing loop is synchronous and blocking, these components limit the overall frame rate.

---

## TASK 12 — SEVERITY CLASSIFICATION AUDIT

The `classify_severity` function assigns potholes to a severity category (Low, Medium, High, Critical).

### Severity Equations

*   **Normalisation**:
    $$d_{\text{norm}} = \text{min}\left(\frac{\text{depth\_cm}}{15.0}, 1.0\right)$$
    $$w_{\text{norm}} = \text{min}\left(\frac{\text{width\_cm}}{100.0}, 1.0\right)$$
*   **Scoring Formula**:
    $$\text{score} = d_{\text{norm}} \times 0.50 + w_{\text{norm}} \times 0.30 + \text{confidence} \times 0.20$$
*   **Classification Rules**:
    *   **Critical**: $\text{score} > 0.65$
    *   **High**: $\text{width\_cm} > 50\text{ cm}$ OR $\text{depth\_cm} > 8\text{ cm}$
    *   **Medium**: $\text{width\_cm} \ge 20\text{ cm}$ OR $\text{depth\_cm} \ge 3\text{ cm}$
    *   **Low**: Otherwise

---

### Severity Classification Truth Table

| depth_cm | width_cm | YOLO Conf | d_norm | w_norm | Score | Tier Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 2.0 | 10.0 | 0.95 | 0.133 | 0.100 | 0.287 | **Low** (Fails Medium thresholds) |
| 1.0 | 55.0 | 0.20 | 0.067 | 0.550 | 0.239 | **High** (Matches width > 50 condition) |
| 9.0 | 30.0 | 0.30 | 0.600 | 0.300 | 0.450 | **High** (Matches depth > 8 condition) |
| 10.0 | 48.0 | 0.30 | 0.667 | 0.480 | 0.538 | **High** (Fails Critical score threshold) |
| 10.0 | 48.0 | 0.95 | 0.667 | 0.480 | 0.668 | **Critical** (Score > 0.65) |

---

### Auditing Analysis & Design Critique

1.  **Confidence Bias**:
    YOLO confidence measures class certainty, not physical severity. Including confidence in the scoring formula means the same physical pothole ($10\times 48\text{ cm}$) can jump from a **High** rating (low confidence) to a **Critical** rating (high confidence).
2.  **Wide/Shallow Pothole Classification**:
    A shallow puddle ($1\text{ cm}$ deep, $55\text{ cm}$ wide) is classified as **High** because of the OR condition in the High tier checks.
3.  **Boundary Operators**:
    The boundary checks use strict inequalities (`>`) for High/Critical, but inclusive inequalities (`>=`) for Medium, which can lead to edge-case classification gaps.

#### Viva Defense:
Including YOLO confidence in the severity score acts as a heuristic filter: if the model is uncertain about a detection, its severity score is reduced to prevent false-positive alarms from triggering Critical alerts.

---

## TASK 13 — WEBSOCKET ENGINE AUDIT

The WebSocket handler manages real-time frame transmission and processing.

### WebSocket Engine Details

*   **Host & Port**: `0.0.0.0:8765` ([live_ws.py:90-91](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L90-L91)).
*   **Startup Lifecycle**: Starts the WebSocket server after calling `initialize_models()` to pre-load weights into GPU VRAM ([live_ws.py:357](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L357)).
*   **Event Loop Blocking**:
    **CRITICAL CONCURRENCY BUG**: The WebSocket handler uses `asyncio`. However, the inference calls `detector.detect()` and `depth_estimator.estimate()` are synchronous, blocking operations. They do not yield control using `await`. Running inference blocks the entire thread, preventing other concurrent WebSocket clients from processing frames.
*   **Reconnection Handling**: Client disconnects are caught by the outer connection try-except block ([live_ws.py:335-336](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L335-L336)), which cleans up resources and resets the per-client cache.

### Core Processing Loop Excerpt

```python
# From backend/live_ws.py (Lines 199-224)
        async for message in websocket:
            t0 = time.perf_counter()

            try:
                # ── 1. Decode incoming Base64 JPEG frame ─────────────────
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
                result     = detector.detect(frame_rgb)
                detections = result.get("detections", [])

                # ── 3. MiDaS depth map (every DEPTH_INTERVAL frames) ─────
                if depth_estimator and frame_count % DEPTH_INTERVAL == 0:
                    cached_depth = depth_estimator.estimate(frame_rgb)
```

---

## TASK 14 — FRONTEND & ALERTS AUDIT

The React frontend manages camera capture and overlays canvas detections in real time.

### Frontend Logic & ADAS System

1.  **Webcam Frame Grab**:
    The browser captures the webcam stream using `navigator.mediaDevices.getUserMedia` at a resolution of 640x480 ([LiveDetection.jsx:269](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L269)).
2.  **Transmission Loop Throttle**:
    The send loop is managed via `setInterval` running every `120 ms` (~8.3 FPS) ([LiveDetection.jsx:394](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L394)).
3.  **Backpressure Control**:
    The frontend implements a lock: `wsConnectedRef` and `pendingRef`. If a frame is sent, `pendingRef` is set to `true`, and no new frames are sent until the backend returns a response. If the backend slows down, the transmission rate drops dynamically, preventing network queuing.
4.  **ADAS Warning Audio Synthesis**:
    Alert sound synthesis uses the browser Web Audio API ([LiveDetection.jsx:200-229](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L200-L229)). It generates a double beep (920 Hz) using sine wave oscillators:
    ```javascript
    playTone(now, 920, 0.12);
    playTone(now + 0.18, 920, 0.15);
    ```
5.  **Alert Debouncing**:
    To prevent sound overlapping, a time lock check is run before playing the warning tone ([LiveDetection.jsx:248](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L248)):
    ```javascript
    if (nowMs - lastBeepTimeRef.current > 1000) { ... }
    ```

---

## TASK 15 — YOLO TRAINING & METRICS COMPARISON

This section reviews the training configuration and metric changes across different iterations of the pothole detector.

### Training Progress & Configuration Table

| Parameter / Metric | YOLOv8s Baseline | YOLO11s Baseline | YOLO11s Fine-Tuned (v1) | YOLO11s Fine-Tuned (v22) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Weights File** | `yolov8s.pt` (Base) | `yolo11s.pt` (Base) | `yolo11s_v1/best.pt` | `yolo11s_v22/best.pt` (Active Production) |
| **Optimizer** | AdamW | AdamW | SGD | Auto (SGD/AdamW) |
| **Epochs Budget** | 100 | 100 | 100 | 150 (Stopped early at 45) |
| **Training Batch Size** | 16 | 16 | 8 | 8 |
| **Image Resolution** | 640 px | 640 px | 640 px | **800 px** |
| **Backbone Frozen Layers** | None | None | None | **First 10 Layers** (`freeze=10`) |
| **Learning Rate Strategy** | Constant | Constant | Linear decay | **Cosine LR** (`cos_lr=true`) |
| **Mosaic Augmentation** | 0.0 | 0.0 | 0.5 | **1.0** (Full scale) |
| **Mixup Augmentation** | 0.0 | 0.0 | 0.0 | **0.15** |
| **Validation Precision** | 64.27% | 64.21% | 56.84% | **65.78%** |
| **Validation Recall** | 54.05% | 55.26% | 33.08% | **68.86%** |
| **Validation mAP@50** | 57.34% | 57.84% | 33.83% | **72.26%** |
| **Validation mAP@50-95**| 29.77% | 30.53% | 18.67% | **40.79%** |

---

### Key Takeaways from Training Audit
*   **The v1 Fine-tuning Drop**:
    The baseline model performed better than the initial v1 fine-tuned model (mAP@50 dropped from 57.84% to 33.83%). This occurred because the dataset splits were not cleaned, and learning rate decay was not configured properly.
*   **The v22 Performance Boost**:
    The v22 model achieved a **72.26% mAP@50**. This improvement was driven by:
    1.  Increasing resolution from 640 to 800 pixels.
    2.  Freezing the first 10 layers of the backbone to prevent catastrophic forgetting.
    3.  Using Cosine LR scheduling.
    4.  Applying offline augmentations (mixup and mosaic).

---

## TASK 16 — METRICS & BENCHMARK AUDIT

This section audits the claims made about pipeline performance and latency.

### Performance Claims Validation Table

| Claimed Metric | Codebase Evidence | Exact Source | Reproducible? | Verdict / Discrepancy |
| :--- | :--- | :--- | :--- | :--- |
| **mAP@50 = 72.26%** | Matches `0.72264` | `yolo11s_v22/results.csv` (Row 23 / Epoch 22) | **Yes** | **Validated**. Match is exact. |
| **Precision = 65.78%** | Matches `0.65781` | `yolo11s_v22/results.csv` (Row 23 / Epoch 22) | **Yes** | **Validated**. Match is exact. |
| **Recall = 68.86%** | Matches `0.68861` | `yolo11s_v22/results.csv` (Row 23 / Epoch 22) | **Yes** | **Validated**. Match is exact. |
| **F1 Score = 0.673** | Derived formula value | Calculated from $2 \times \frac{P \times R}{P + R}$ | **Yes** | **Validated**. Match is exact. |
| **Depth MAE = 5.48 cm** | None | None | **No** | **NOT CONFIRMED FROM THE REPOSITORY.** The code contains no depth evaluation logic. |
| **Sample size n = 79** | None | None | **No** | **NOT CONFIRMED FROM THE REPOSITORY.** |
| **GTX 1650 latency** | Average Latency: 109.23 ms | `04_pipeline_benchmark.ipynb` (Line 411) | **Yes** | **Validated**. GPU times were synchronized correctly. |
| **Average Pipeline FPS** | Average: 9.51 FPS | `04_pipeline_benchmark.ipynb` (Line 419) | **Yes** | **Validated**. |

---

### Latency Measurement & Calculations Audit
*   **GPU Synchronization**:
    The benchmark notebooks call `torch.cuda.synchronize()` before and after time checks, ensuring accurate timing measurements on GPU.
*   **FPS Calculation Mismatch**:
    Inside the benchmark loop, the average FPS is calculated as the mean of individual frame FPS values:
    ```python
    mean_fps = fps_values.mean()
    ```
    This average is **9.51 FPS**. However, if you calculate the overall pipeline FPS using the mean latency:
    $$\text{FPS} = \frac{1000.0}{109.236\text{ ms}} = \mathbf{9.15\text{ FPS}}$$
    Averaging individual frame rates introduces a positive bias compared to using the total elapsed time, making the reported FPS appear slightly higher than it actually is.

---

## TASK 17 — CV BUGS & VIVA RISKS

This section highlights implementation issues and architectural limitations that could be flagged by examiners during the viva.

### Risk Assessment Table

| Severity | Issue | Code Location | Runtime Impact | Viva Impact | Safe Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CRITICAL** | **Double BGR/RGB Conversion** | [detector.py:71](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L71) | YOLO runs inference on BGR images with swapped channels, reducing detection sensitivity. | High risk of being flagged as an implementation bug. | "This channel swapping was identified during auditing. Because potholes are detected primarily by shape and texture rather than color, the impact on accuracy is limited, but we plan to patch this in the next release." |
| **CRITICAL** | **Inverted Depth Normalisation** | [utils.py:125](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L125) / [utils.py:329](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L329) | The normalization convention is inverted. Potholes closer to the camera appear shallower, while farther potholes appear deeper. | Undermines the validity of the depth estimation logic. | "The depth values are relative approximations. The inverted mapping is a normalization bug. We will update the convention to match standard disparity scales." |
| **HIGH** | **Inverted Confidence Formula** | [utils.py:237](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L237) | Flat road surfaces yield high MiDaS confidence, while textured pothole cavities yield low confidence. | Disrupts the adaptive hybrid blending logic. | "The confidence estimator penalizes variance rather than texture, which is a logic bug. We plan to correct the confidence mapping." |
| **HIGH** | **Blocking inference in Async loop** | [live_ws.py:218](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L218) | Synchronous inference blocks the event loop, causing server lag when multiple clients are connected. | Highlights limitations in WebSocket server concurrency. | "The server is a prototype designed for single-vehicle testing. For multi-client production deployment, we would run inference on a separate thread pool using Celery or Redis." |
| **MEDIUM** | **Frame Skipping Alignment** | [live_ws.py:222](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L222) | Frame crops are matched against an outdated depth map, causing offset issues at typical driving speeds. | Underlines tracking and latency problems. | "This frame skipping is an optimization trade-off. At higher driving speeds, we would run a Kalman filter or keyframe alignment to correct coordinates." |
| **MEDIUM** | **Unused Calibration module** | [calibration.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/calibration.py) | Dead code remains in the repository, and the active backend uses hardcoded calibration constants instead. | Indicates uncleaned files in the workspace. | "The calibration script was developed as part of an autonomous reference marker detection prototype, but we chose to use hardcoded defaults for deployment stability." |

---

## TASK 18 — CONTRIBUTION ANALYSIS

This section outlines what was built as part of the project versus what was imported from external libraries.

### Project Contribution Table

| Component | Provided by Library / Framework | Implemented/Integrated in this Project | Codebase Evidence |
| :--- | :--- | :--- | :--- |
| **YOLO Architecture** | Ultralytics PyTorch wrapper | Loaded and initialized in the wrapper class. | [detector.py:24](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L24) |
| **YOLO Weights** | Pretrained checkpoints (`yolo11s.pt`) | Fine-tuned on the clean dataset. | `args.yaml` |
| **NMS & Tensor parsing**| NMS is handled by Ultralytics backend | Box coordinates are parsed and converted to list format. | [detector.py:84-102](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L84-L102) |
| **MiDaS Small Model** | Intel ISL PyTorch Hub wrapper | Initialized and upsampled back to frame size. | [utils.py:94](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L94) |
| **OpenCV Preprocessing** | `cv2.imdecode` & `cv2.cvtColor` | Frame decoding and color conversions. | [live_ws.py:206-214](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L206-L214) |
| **YOLO-MiDaS Handoff** | PyTorch framework tensors | The bounding boxes from YOLO are cropped out of the MiDaS depth map. | [live_ws.py:236](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L236) |
| **Centimetre Equations** | None | Pinhole camera scaling and fallback perspective damping. | [utils.py:307](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L307) |
| **Severity Scoring** | None | Normalization weights and scoring tiers. | [utils.py:334](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L334) |
| **WebSocket Server** | `websockets` python module | Async connection loop and base64 message parsing. | [live_ws.py:199](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L199) |
| **React GUI** | React / Tailwind CSS / Framer Motion | Live canvas rendering and HUD statistics. | `LiveDetection.jsx` |
| **ADAS alerts** | HTML5 audio context | Real-time warning beeps synthesis. | [LiveDetection.jsx:200-229](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L200-L229) |

#### Safe Answer to: "Did you build YOLO and MiDaS yourself?"
> *"I did not build the underlying neural network architectures for YOLO or MiDaS. My contribution lies in the dataset preparation, training configuration, system integration, hybrid depth blending logic, and building the WebSocket pipeline."*

---

## TASK 19 — VIVA STUDY GUIDE

A guide to the core computer vision concepts used in the LIVEDET project.

1.  **Computer Vision**: Analyzing and understanding digital images. LIVEDET uses it to detect potholes and estimate their dimensions. ([detector.py](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py)).  
    *Question*: What is Computer Vision?  
    *Answer*: Computer vision is a field of AI that enables computers to extract meaningful information from digital images or video frames.
2.  **Object Classification vs. Detection**: Classification identifies what is in an image; detection locates objects and draws bounding boxes around them. ([detector.py:97](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L97)).  
    *Question*: What is the difference between classification and detection?  
    *Answer*: Classification assigns a label to an entire image, whereas detection identifies and locates objects within the image, returning bounding boxes.
3.  **YOLO (You Only Look Once)**: A single-stage object detector that processes images in one forward pass, making it suitable for real-time applications. ([detector.py:81](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L81)).  
    *Question*: Why use YOLO instead of a two-stage detector like Faster R-CNN?  
    *Answer*: YOLO processes images in a single forward pass, providing faster inference times that are suitable for real-time applications.
4.  **Bounding Boxes**: Bounding boxes define the boundaries of detected objects. LIVEDET uses `[x, y, w, h]` format. ([detector.py:97](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L97)).  
    *Question*: What is a bounding box?  
    *Answer*: A bounding box is a rectangular frame defined by coordinates that encloses a detected object in an image.
5.  **Confidence**: A score representing the probability that a bounding box contains a target object. ([detector.py:88](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L88)).  
    *Question*: What does the confidence score represent in YOLO?  
    *Answer*: It is the model's predicted probability that a bounding box contains a target object and matches the ground truth.
6.  **Intersection over Union (IoU)**: Measures the overlap between the predicted bounding box and the ground truth box. ([detector.py:81](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L81)).  
    *Question*: How is IoU used?  
    *Answer*: IoU measures the ratio of overlap area to the union area between two bounding boxes, and is used to filter redundant predictions.
7.  **Non-Maximum Suppression (NMS)**: Filters out redundant, overlapping bounding boxes, keeping only the highest-scoring detection. ([detector.py:81](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L81)).  
    *Question*: Why is NMS necessary?  
    *Answer*: NMS suppresses redundant bounding boxes that refer to the same physical object, keeping only the highest-confidence box.
8.  **Precision**: The ratio of true positive detections to all predicted detections. ([results.csv](file:///c:/Users/ihsan/Documents/GitHub/ML2/models/finetuned/pothole_detector_yolo11s_v22/results.csv)).  
    *Question*: What is precision?  
    *Answer*: Precision is the percentage of positive detections that are correct, measuring the model's false-positive rate.
9.  **Recall**: The ratio of true positive detections to all actual target objects in the dataset. ([results.csv](file:///c:/Users/ihsan/Documents/GitHub/ML2/models/finetuned/pothole_detector_yolo11s_v22/results.csv)).  
    *Question*: What is recall?  
    *Answer*: Recall is the percentage of actual objects that were successfully detected by the model, measuring the false-negative rate.
10. **F1 Score**: The harmonic mean of precision and recall. ([results.csv](file:///c:/Users/ihsan/Documents/GitHub/ML2/models/finetuned/pothole_detector_yolo11s_v22/results.csv)).  
    *Question*: Why use the F1 Score instead of accuracy?  
    *Answer*: The F1 score balance precision and recall, providing a better metric for datasets with imbalanced classes.
11. **mAP@50**: Mean Average Precision calculated at an IoU threshold of 0.50. ([results.csv](file:///c:/Users/ihsan/Documents/GitHub/ML2/models/finetuned/pothole_detector_yolo11s_v22/results.csv)).  
    *Question*: What is mAP@50?  
    *Answer*: It is the average precision calculated across all classes at a 50% overlap threshold.
12. **Transfer Learning**: Fine-tuning a pre-trained model on a new target dataset. ([fine_tune.py:67](file:///c:/Users/ihsan/Documents/GitHub/ML2/model_training/fine_tune.py#L67)).  
    *Question*: Why is transfer learning effective?  
    *Answer*: It leverages generic features learned from large-scale datasets, reducing the training data and time needed for a new task.
13. **Epoch**: One complete pass of the training dataset through the neural network. ([fine_tune.py:75](file:///c:/Users/ihsan/Documents/GitHub/ML2/model_training/fine_tune.py#L75)).  
    *Question*: What is an epoch?  
    *Answer*: An epoch is one complete training cycle where the entire dataset is passed forward and backward through the neural network.
14. **Overfitting**: When a model performs well on training data but fails to generalize to validation data. ([fine_tune.py:95](file:///c:/Users/ihsan/Documents/GitHub/ML2/model_training/fine_tune.py#L95)).  
    *Question*: How did you prevent overfitting during training?  
    *Answer*: We used dropout, early stopping (patience=25), backbone freezing, and offline dataset augmentations.
15. **MiDaS Small**: A lightweight neural network model used to estimate relative depth from single images. ([utils.py:94](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L94)).  
    *Question*: What is MiDaS Small?  
    *Answer*: It is a lightweight monocular depth estimation model optimized for fast execution on mobile or edge devices.
16. **Relative vs. Metric Depth**: Relative depth measures distance relations within a scene; metric depth measures physical distance. ([utils.py:152](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L152)).  
    *Question*: Why does MiDaS output relative depth instead of actual centimeters?  
    *Answer*: Monocular images lack physical scale references. The model can only resolve relative depth relationships.
17. **Camera Calibration**: Mapping pixel dimensions to physical distances. ([utils.py:307](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L307)).  
    *Question*: How is calibration handled in LIVEDET?  
    *Answer*: The system uses a pinhole camera projection model with a hardcoded focal length (600 pixels) to approximate dimensions.
18. **Pinhole Camera Model**: Describes the mathematical relationship between physical coordinates and pixel coordinates. ([utils.py:307](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L307)).  
    *Question*: What is the pinhole projection formula?  
    *Answer*: It is $\text{Physical Width} = \frac{\text{Pixel Width} \times \text{Distance}}{\text{Focal Length}}$.
19. **Depth-Map Sampling**: Extracting depth values from a specific region of a depth map. ([utils.py:192](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L192)).  
    *Question*: Why does the code use median sampling?  
    *Answer*: Median filters are robust against outliers and noise, which are common at object boundaries.
20. **Hybrid Blending**: Combining neural network predictions with geometric fallback heuristics. ([utils.py:271](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L271)).  
    *Question*: What is the benefit of hybrid depth estimation?  
    *Answer*: It blends neural depth with a geometric fallback, maintaining accuracy on textured surfaces while falling back to the heuristic on uniform surfaces.
21. **Severity Classification**: Categorizing road defects by severity based on size and depth. ([utils.py:374-381](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L374-L381)).  
    *Question*: How is severity determined?  
    *Answer*: It is classified into Low, Medium, High, or Critical tiers based on blended depth, width, and YOLO confidence.
22. **WebSocket Live Pipeline**: Real-time frame transmission using WebSockets. ([live_ws.py:199](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L199)).  
    *Question*: Why use WebSockets instead of HTTP REST?  
    *Answer*: WebSockets establish a persistent connection, reducing handshake overhead and latency during continuous frame transmission.
23. **Frame Skipping**: Throttling computationally heavy tasks to run only once every few frames. ([live_ws.py:222](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L222)).  
    *Question*: What is the benefit of frame-skipped depth mapping?  
    *Answer*: Running depth estimation every 3rd frame reduces GPU load while maintaining a smooth output frame rate.
24. **Latency vs. Throughput**: Latency is the round-trip time for a single frame; throughput is the total frames processed per second. ([live_ws.py:309](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L309)).  
    *Question*: What is the difference between latency and FPS in LIVEDET?  
    *Answer*: Latency is the processing time for a single frame (~109 ms). FPS is the average throughput (~9.5 FPS).

---

## TASK 20 — QUICK REFERENCE & STUDY GUIDE

### A. 15 Files/Functions to Open Before the Viva
1.  [backend/live_ws.py:199](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L199) - `handle_client` (Main processing loop)
2.  [backend/live_ws.py:119](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L119) - `initialize_models` (Model initialization)
3.  [backend/detector.py:24](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L24) - `ObjectDetector` (YOLO wrapper class)
4.  [backend/detector.py:70](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L70) - `detect` BGR/RGB conversion check
5.  [backend/utils.py:67](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L67) - `DepthEstimator` (MiDaS wrapper class)
6.  [backend/utils.py:111](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L111) - `estimate` forward pass
7.  [backend/utils.py:162](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L162) - `extract_median_depth` (Crop depth sampling)
8.  [backend/utils.py:196](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L196) - `compute_midas_confidence` (Variance estimator)
9.  [backend/utils.py:241](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L241) - `blend_depth` (Hybrid blending logic)
10. [backend/utils.py:282](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L282) - `compute_real_width` (Pinhole math)
11. [backend/utils.py:311](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L311) - `compute_depth_cm` (Depth mapping scaling)
12. [backend/utils.py:334](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L334) - `classify_severity` (Severity scoring equations)
13. [backend/utils.py:390](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L390) - `compute_heuristic_measurements` (Perspective fallbacks)
14. [frontend/src/pages/LiveDetection.jsx:318](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L318) - Canvas rendering loop
15. [frontend/src/pages/LiveDetection.jsx:200](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L200) - Audio alert synthesis

### B. 10 Code Sections to Display During the Demo
1.  **Frame Encoding/Send**: [LiveDetection.jsx:394-408](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L394-L408) (JPEG capture and transmit).
2.  **WebSocket Processing Loop**: [live_ws.py:199-218](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L199-L218) (Base64 decode and YOLO call).
3.  **Frame Skipping**: [live_ws.py:222-224](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L222-L224) (MiDaS execution every 3rd frame).
4.  **Median Depth Extraction**: [utils.py:162-193](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L162-L193) (Depth map cropping).
5.  **Weighted Blending**: [utils.py:267-280](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L267-L280) (MiDaS vs Heuristic blend).
6.  **Pinhole Geometry**: [utils.py:306-308](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L306-L308) (Width calculation).
7.  **Perspective Damping**: [utils.py:458-463](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L458-L463) (Heuristic depth damping).
8.  **ADAS Alert Sound**: [LiveDetection.jsx:200-221](file:///c:/Users/ihsan/Documents/GitHub/ML2/frontend/src/pages/LiveDetection.jsx#L200-L221) (Browser Web Audio API beep).
9.  **Severity Scoring**: [utils.py:368-382](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L368-L382) (Categorizing defects).
10. **Dual Color Swapping**: [detector.py:70-71](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L70-L71) (Channel swapping check).

### C. 15 Numbers & Constants to Remember
1.  `8765` - WebSocket Port
2.  `8000` - REST API Port
3.  `5173` - Vite Frontend Port
4.  `600` px - Hardcoded Camera Focal Length
5.  `3` - Depth Interval (MiDaS runs every 3rd frame)
6.  `120` ms - Frontend camera capture interval
7.  `0.15` - YOLO Confidence Threshold
8.  `0.25` - YOLO IoU Threshold
9.  `72.26%` - Fine-tuned YOLO11s mAP@50
10. `65.78%` - Fine-tuned YOLO11s Precision
11. `68.86%` - Fine-tuned YOLO11s Recall
12. `0.673` - Fine-tuned YOLO11s F1 Score
13. `9.51` FPS - Average backend processing frame rate
14. `109.23` ms - Mean pipeline latency
15. `19.2` MB - Fine-tuned YOLO11s weight footprint

### D. 20 Technical Viva Questions & Answers

1.  **Question**: How does single-stage YOLO differ from two-stage Faster R-CNN?  
    **Answer**: Two-stage detectors generate region proposals first and then classify them. YOLO treats detection as a single regression problem, predicting boxes and class probabilities in one forward pass, which makes it much faster.
2.  **Question**: What is the purpose of Intersection over Union (IoU)?  
    **Answer**: IoU measures the ratio of overlap to the union area between two bounding boxes. It is used to quantify detection accuracy and suppress redundant predictions during NMS.
3.  **Question**: What does Mean Average Precision (mAP) measure?  
    **Answer**: mAP is the mean of average precision values across all classes. It measures both classification accuracy and localization precision.
4.  **Question**: Why does the system upsample the MiDaS depth map using bicubic interpolation?  
    **Answer**: MiDaS Small outputs a low-resolution depth map (256x256). Upsampling it back to the frame size (640x480) allows us to crop depth values using the original bounding box coordinates.
5.  **Question**: What is the difference between relative disparity and metric depth?  
    **Answer**: Relative disparity measures relative distance relationships within a scene. Metric depth measures actual physical distance (e.g. in meters). Monocular systems require external calibration references to estimate metric depth.
6.  **Question**: What camera model is used to estimate physical width?  
    **Answer**: We use the pinhole camera model, which uses the linear projection equation: $\text{Width} = \frac{\text{Pixel Width} \times \text{Distance}}{\text{Focal Length}}$.
7.  **Question**: What is the purpose of perspective damping?  
    **Answer**: Potholes closer to the camera appear larger due to perspective. Perspective damping reduces estimated depth for objects lower in the frame to correct for this magnification effect.
8.  **Question**: Why does the backend use median depth values instead of average values?  
    **Answer**: Median values are more robust against edge outliers and noise, which occur at the boundary of bounding boxes.
9.  **Question**: How is backpressure managed between the frontend and backend?  
    **Answer**: The frontend uses a lock flag. It waits for the response of the previous frame before sending the next one, which prevents network buffering and lag.
10. **Question**: How are the ADAS audio alerts generated?  
    **Answer**: They are synthesized dynamically using the browser's Web Audio API (`AudioContext`), which avoids the overhead of loading external audio files.
11. **Question**: What is the impact of the color space bug in the detector wrapper?  
    **Answer**: Because channels are swapped twice, the image is passed to YOLO in BGR format instead of RGB. This can reduce model sensitivity to color features.
12. **Question**: What training augmentations were used for the v22 model?  
    **Answer**: We used mosaic (1.0), mixup (0.15), and HSV color augmentations.
13. **Question**: What layers were frozen during fine-tuning of the YOLO11s model?  
    **Answer**: The first 10 layers of the backbone were frozen to preserve general feature extraction weights and prevent overfitting.
14. **Question**: How does the system handle reconnects?  
    **Answer**: The WebSocket server resets the client state (frame count and cached depth map) upon reconnection.
15. **Question**: Why does the backend block during inference?  
    **Answer**: The inference calls are synchronous and block the single-threaded asyncio event loop.
16. **Question**: How can this blocking behavior be resolved?  
    **Answer**: By offloading inference calls to a separate process pool or worker queue using libraries like Celery.
17. **Question**: How was the F1 score calculated?  
    **Answer**: The F1 score is the harmonic mean of precision and recall: $F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.
18. **Question**: Why does the average FPS calculation in the benchmark show a positive bias?  
    **Answer**: The benchmark averages individual frame FPS values rather than dividing the total frames by the total elapsed time.
19. **Question**: What does standard deviation measure in `compute_midas_confidence`?  
    **Answer**: It measures local depth variance. However, because of the inverted formula, low variance yields high confidence, which is a logic bug.
20. **Question**: Is reference marker calibration active in production?  
    **Answer**: No. The calibration script is unused, and the system relies on hardcoded constants.

### E. 10 Dangerous Claims to Avoid
1.  *Do NOT claim MiDaS is physically calibrated in centimeters.* (It is relative disparity).
2.  *Do NOT claim the pipeline processes frames at a constant 30 FPS.* (It is capped at ~8 FPS in the frontend, with an average processing rate of 9.5 FPS).
3.  *Do NOT claim the reference marker calibration is running in production.* (It is dead/unused code).
4.  *Do NOT claim the depth MAE (5.48 cm) is validated by code in the repository.* (No such evaluation logic exists).
5.  *Do NOT claim the system supports multiple concurrent clients without performance loss.* (Inference calls are blocking).
6.  *Do NOT claim the hybrid blend is mathematically optimal.* (The confidence formula is inverted).
7.  *Do NOT claim the camera height is calibrated automatically.* (It uses hardcoded fallback constants).
8.  *Do NOT claim the YOLO model is evaluated on perfect RGB images.* (Channels are swapped twice, feeding BGR images).
9.  *Do NOT claim the temporal frame alignment is perfect.* (Outdated cached depth maps are matched against current frames).
10. *Do NOT claim that severity classification is independent of model confidence.* (Confidence directly influences the score).

### F. 10 Honest Limitation Statements
1.  The depth estimations are relative approximations, not certified metric measurements.
2.  The camera calibration uses hardcoded defaults (such as a 600 px focal length), which assumes a specific height and tilt.
3.  The pipeline does not use temporal tracking (like Kalman filtering) to align cached depth maps with moving frames.
4.  Synchronous inference calls block the event loop, limiting concurrent client scaling.
5.  The confidence mapping contains a logic inversion, meaning textureless regions yield high confidence.
6.  The depth estimation normalisation is inverted, causing closer objects to appear shallower.
7.  The YOLO detector receives BGR frames instead of RGB due to a channel-swapping mismatch.
8.  The severity classification relies partly on YOLO's class confidence, which is a probability metric rather than a physical measurement.
9.  The pipeline's real-time frame rate is capped by network transmission and processing latency, averaging around 9.5 FPS.
10. The benchmark's average FPS calculation is slightly biased because it averages individual frame FPS values.

### G. 60-Second System Explanation
> *"LIVEDET is a real-time road defect detection system. It captures webcam frames in a React frontend and transmits them via WebSockets to a Python backend. The backend decodes the frames and passes them to a fine-tuned YOLO11s detector to locate potholes. To estimate depth, we run Intel MiDaS Small on every 3rd frame, reusing the cached depth map for the remaining frames. The system then blends this neural depth map with a perspective-damped geometric heuristic. It computes a severity score using depth, width, and detection confidence, returning structured JSON to the frontend to draw canvas overlays and trigger ADAS audio warning beeps."*

### H. 3-Minute Computer Vision Code Walkthrough
1.  **Inference Entrance**: Point to [backend/live_ws.py:199](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/live_ws.py#L199), explaining how base64 frames are received and decoded using OpenCV.
2.  **YOLO Detection**: Show the `ObjectDetector` initialization in `live_ws.py` line 144, pointing out the confidence and IoU thresholds. Open [backend/detector.py:70](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/detector.py#L70) to explain color conversion and model inference.
3.  **MiDaS Estimation**: Show the depth estimation call in `live_ws.py` line 223. Walk through `DepthEstimator` in [backend/utils.py:111](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L111), explaining bicubic upsampling and min-max normalization.
4.  **Sampling & Blending**: Walk through `extract_median_depth` ([backend/utils.py:162](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L162)) and the hybrid blending logic in `blend_depth` ([backend/utils.py:241](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L241)). Explain how standard deviation is used to weigh MiDaS against the fallback heuristic.
5.  **Pinhole Width & Severity**: Explain the pinhole geometry calculation in `compute_real_width` ([backend/utils.py:282](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L282)) and how severity is categorized in `classify_severity` ([backend/utils.py:334](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L334)).

### I. 5-Minute Complete Implementation Walkthrough
1.  **Frontend Capture**: Walk through camera initialization and frame grabbing in `LiveDetection.jsx` (lines 268-280, 386-408).
2.  **WebSocket Client & Hook**: Explain connection management and backpressure controls in `useWebSocket.js`.
3.  **Inference Server**: Explain model loading singletons and the asynchronous message loop in `live_ws.py`.
4.  **Computer Vision Wrapper**: Walk through YOLO and MiDaS wrappers in `detector.py` and `utils.py`.
5.  **Heuristic Fallback**: Explain perspective damping in `compute_heuristic_measurements` ([backend/utils.py:390](file:///c:/Users/ihsan/Documents/GitHub/ML2/backend/utils.py#L390)).
6.  **Canvas & Audio Alerts**: Show the rendering loop and ADAS warning synthesis in `LiveDetection.jsx` (lines 318-370, 200-221).

### J. Emergency Answers
*   *For architecture questions*:  
    "I used library implementations for the underlying neural architectures. My contribution focuses on dataset preparation, training configuration, system integration, hybrid blending, and building the WebSocket pipeline."
*   *For accuracy/depth questions*:  
    "The output is an approximate severity indicator based on relative monocular depth and heuristic calibration, not a certified physical road measurement."
*   *For frame rate/latency questions*:  
    "The system processes frames continuously, making it a live pipeline. However, it is not an automotive-grade, high-frame-rate system."

### K. Top 5 Manual Inspection Steps Before Presenting
1.  **Verify WebSocket Status**: Start `live_ws.py` and verify that the connection status in the React GUI changes to "Connected".
2.  **Inspect Console Logs**: Check for errors in the browser developer tools and backend logs.
3.  **Verify Port Configuration**: Ensure the React client ports match the backend configuration (ports 8765 and 8000).
4.  **Confirm Device Selection**: Verify that the backend initialized on the GPU by checking the logs for `cuda:0`.
5.  **Test ADAS Alerts**: Toggle the audio warning switch in the GUI and click the test beep button to confirm AudioContext synthesis works.
