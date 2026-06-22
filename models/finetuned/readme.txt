======================================================================
POTHOLE DETECTOR MODEL COMPARISON & OVERFITTING ANALYSIS
======================================================================

Currently Deployed Model (Best Fine-Tuned):
- Folder Name: pothole_detector_yolo11s_v22
- Path: models/finetuned/pothole_detector_yolo11s_v22/weights/best.pt
- Name in .env: YOLOv11s-Pothole-Detector-FineTuned

Original / Baseline Model:
- Folder Name: pothole_detector_yolo11s
- Path: runs/base_models/pothole_detector_yolo11s/weights/best.pt
- Name: YOLOv11s-Pothole-Detector (Original v1)

----------------------------------------------------------------------
PERFORMANCE COMPARISON
----------------------------------------------------------------------
Metric            v1 (Original)      v2 (Fine-Tuned v22)   Delta
----------------------------------------------------------------------
  mAP@50                 61.03%                 72.26%    +11.23%
  mAP@50-95              33.12%                 40.79%     +7.67%
  Precision              69.22%                 65.78%     -3.44%
  Recall                 54.69%                 68.86%    +14.17%
  F1 Score               61.10%                 67.28%     +6.18%
======================================================================

----------------------------------------------------------------------
WHY THE V22 CHECKPOINT WAS SELECTED (OVERFITTING ANALYSIS)
----------------------------------------------------------------------
During fine-tuning of the YOLO11s model on the merged pothole dataset, we monitored performance metrics and losses across epochs to locate the optimal validation point.
The model reaches its peak performance at Epoch 22 before exhibiting signs of overfitting:

1. Peak Performance (Epoch 22):
   - mAP@50 reaches its peak of 72.26%.
   - Validation Box Loss: 1.4833
   - Validation Class Loss: 1.2657

2. Onset of Overfitting (Epochs 23+):
   - By Epoch 25, mAP@50 drops significantly to 65.09%.
   - Validation Class Loss begins to climb (e.g., 1.3033 at Epoch 24, 1.3140 at Epoch 25).
   - Although training losses continue to decrease, validation metrics deteriorate, indicating that checkpoints after Epoch 22 are overfitted.

Therefore, the Epoch 22 checkpoint (saved as best.pt inside pothole_detector_yolo11s_v22) was selected as the final production model. It provides the best trade-off with the highest mAP@50 and a strong balance between Precision and Recall.
