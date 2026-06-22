YOLO Model Checkpoints Guide

Purpose:
This folder is used to store pre-trained model weights and target checkpoints for inference.

Files & Subdirectories:
- yolov8_best.pt: Best trained YOLOv8 Small model checkpoint.
- yolo11_best.pt: Best trained YOLO11 Small model checkpoint.
- yolo11n.pt: Pre-trained default weights for YOLO11 Nano model.
- yolo11s.pt: Pre-trained default weights for YOLO11 Small model.
- yolov8n.pt: Pre-trained default weights for YOLOv8 Nano model.

Finetuned Models Subdirectory (finetuned/):
- pothole_detector_yolo11s_v22: Peak-performance YOLO11s checkpoint fine-tuned on the clean merged pothole dataset (mAP@50: 72.26% at Epoch 22). See finetuned/readme.txt for detailed overfitting analysis and v1 vs v2 comparisons.
- yolov8s_fresh: Subdirectory for YOLOv8s fine-tuning experiments.
