================================================================================
LIVEDET — Live Road Defect Detection System (Thesis Prototype)
================================================================================

PURPOSE & OVERVIEW
------------------
This is a real-time computer vision ADAS (Advanced Driver Assistance System) prototype 
designed to detect potholes on roads, estimate their physical width and depth using 
a hybrid monocular camera depth logic (MiDaS + Heuristics), evaluate hazard severity 
levels, and notify drivers using visual UI alerts.

LIST
------------------
- backend/
  Flask REST APIs for static file processing and WebSocket servers for real-time 
  video stream analysis.
  
- frontend/
  React web application dashboard containing live webcam streams, uploaded image 
  tables, and analytics.

- dataset/
  Storage folders containing raw image sets and clean unified YOLO training labels.
  
- models/
  Directory housing base and fine-tuned YOLO object detector weights.

- model_training/
  Python scripts and utilities used to train, augment, and fine-tune YOLO models.

- notebooks/
  Jupyter notebooks for verifying camera setups, running inference batches, 
  and validating metrics.

- scripts/
  Standalone python scripts for dataset administration and formatting tasks.

- 04_TEST_FROM_DEPTH_MEASURED_SET/ & 05_EXTENDED_DEPTH_SUBSET/
  Strictly held-out evaluation datasets for depth validation against Intel 
  RealSense D415 hardware ground truth.

QUICK START
-----------
1. Set up the Python virtual environment (e.g. `venv-gpu`) and install backend dependencies.
2. Set up node packages in the `frontend/` directory (`npm install`).
3. Start the entire system using the root startup script or by running the backend 
   and frontend concurrently.
================================================================================
