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

QUICK START & SYSTEM SETUP
--------------------------

1. Python Backend Dependencies:
   - Ensure you are inside the project root directory.
   - Activate your virtual environment (e.g., `venv-gpu` or `.venv`):
     - On Windows Powershell: `.\venv-gpu\Scripts\Activate.ps1`
     - On Windows CMD: `.\venv-gpu\Scripts\activate.bat`
   - Install dependencies listed in backend/requirements.txt:
     `pip install -r backend/requirements.txt`

2. Node.js & NPM Installation/Updates (Frontend):
   - Check if Node.js and NPM are installed:
     `node -v`
     `npm -v`
   - If not installed, download and install Node.js from https://nodejs.org/ (which includes NPM).
   - To update NPM globally to the latest version, run:
     `npm install -g npm@latest`
   - Go to the frontend directory:
     `cd frontend`
   - Install local node packages:
     `npm install`
   - To update existing node packages inside the directory to their latest compatible versions, run:
     `npm update`

3. Running Components Separately:
   For complete functionality, open separate terminal windows and run the following commands:

   - Term 1: Run the Flask REST Server (for static image and video processing)
     `python backend/app.py`
     - Runs on host: Port 8000 (http://localhost:8000)

   - Term 2: Run the asyncio WebSocket Server (for real-time streaming HUD frame processing)
     `python backend/live_ws.py`
     - Runs on host: Port 8765 (ws://localhost:8765)

   - Term 3: Run the React/Vite Frontend Dev Server
     `cd frontend`
     `npm run dev`
     - Runs on local port: http://localhost:5173 (or as indicated in console)
================================================================================

