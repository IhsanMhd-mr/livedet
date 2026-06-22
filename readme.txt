Live Road Defect Detection System (LIVEDET)

This is a real-time computer vision system designed to detect potholes on roads, estimate their physical width and depth using monocular camera depth logic, evaluate danger severity levels, and notify drivers using visual and audio alerts.

Project Directory Map:

- backend: Flask REST APIs for static file processing and WebSocket servers for real-time video stream analysis.
- frontend: React web application dashboard containing live webcam streams, uploaded image tables, and analytics.
- model_training: Python scripts and dataset utilities used to train and validate YOLO object detectors.
- notebooks: Jupyter notebooks for verifying camera setups, testing image detection batches, and validating metrics.
- scripts: Helper scripts for database administrative task directories and dataset preparations.
- dataset: Storage folders containing raw image sets and clean unified training labels.
- models: Checkpoint directory housing target model weights.
- docs: Detailed research guides and system layout notes.
- report: Thesis research logs and academic summary draft files.

Quick Start:
1. Setup Python virtual environment (.venv) and install dependencies.
2. Setup node packages in frontend.
3. Start the entire system: npm run start
