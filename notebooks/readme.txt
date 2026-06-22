Jupyter Notebooks Guide

Purpose:
This folder contains interactive Python notebooks used to test model inferences, calibrate webcam devices, and evaluate visual predictions.

Files:
- 01_model_validation.ipynb: Loads trained YOLO models, runs validations on validation datasets, and plots evaluation metrics (Precision, Recall, mAP).
- 02_image_detection.ipynb: Batch processes local directories of images to run pothole detector predictions and save outputs to disk.
- 03_live_detection.ipynb: Local webcam stream test notebook that runs real-time camera captures inside an OpenCV popup window.
