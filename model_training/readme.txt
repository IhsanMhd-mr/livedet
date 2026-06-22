YOLO Pothole Model Training Folder Guide

Core Python Scripts:
- train_unified.py: Base training script to train YOLO models from scratch using a general pothole dataset.
- fine_tune.py: Fine-tuning script to apply transfer learning using annotated road defect datasets.
- auto_tune.py: Automated iterative loop that trains and tunes parameters (learning rate, image resolution) until a target validation score is reached.

Notebooks:
- YOLO_MODEL_COMPARISON_REPORT.ipynb: Metrics comparison reports, charts, and speed evaluations for YOLOv8, YOLOv10, and YOLOv11 variants.
- train.ipynb: Test notebook for checking environments, loading weights, and running single training test runs. Cell 15 output displays comparison of the v1 original model vs the v2 fine-tuned model (peak validation mAP@50 of 72.26% at Epoch 22).

Documentation:
- FOLDER_STRUCTURE.txt: Guide explaining model checkpoints directories inside the runs folder.

Model Deployment:
- The optimal fine-tuned weights (Epoch 22, before overfitting) and overfitting analysis files are copied to "models/finetuned/pothole_detector_yolo11s_v22/".

Directories:
- utils: Helper files containing dataset loaders (dataset_handler.py), physical dimension calculators (depth_estimator.py), and safety classification models (severity_calculator.py).
