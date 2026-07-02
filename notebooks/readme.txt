PURPOSE & OVERVIEW
------------------
This directory contains interactive Python notebooks used for experimenting with 
model inferences, calibrating webcam devices, evaluating visual predictions, 
benchmarking pipeline speeds, and compiling evaluation matrices without running 
the full application backend.

FILES & SCRIPTS
---------------
- 01_model_validation.ipynb
  Loads trained YOLO models, runs validations on validation datasets, and plots 
  evaluation metrics (Precision, Recall, mAP).

- 02_image_detection.ipynb
  Batch processes local directories of images to run pothole detector predictions 
  and save annotated outputs to disk.

- 03_live_detection.ipynb
  Local webcam stream test notebook that runs real-time camera captures inside 
  an OpenCV popup window for local validation.

- 04_pipeline_benchmark.ipynb
  Performance testing notebook measuring inference FPS, preprocessing bottlenecks, 
  and evaluating end-to-end latency of the YOLO + MiDaS pipeline.

- 05_model_evaluation.ipynb
  In-depth evaluation on the 147-image held-out test split using production 
  confidence thresholds (conf=0.35, iou=0.5), yielding the final thesis metrics.

- 06_model_comparison.ipynb
  Aggregated comparison generating comparative charts, F1-Score curves, and 
  ablation study analyses for the final documentation.

s=============================================================
