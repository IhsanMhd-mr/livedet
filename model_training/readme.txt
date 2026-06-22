YOLO Pothole Model Training Folder Guide

Core Python Scripts:
- train_unified.py: Base training script to train YOLO models from scratch using a general pothole dataset.
- fine_tune.py: Fine-tuning script to apply transfer learning using annotated road defect datasets.
- auto_tune.py: Automated iterative loop that trains and tunes parameters (learning rate, image resolution) until a target validation score is reached.

Notebooks:
- YOLO_MODEL_COMPARISON_REPORT.ipynb: Metrics comparison reports, charts, and speed evaluations for YOLOv8, YOLOv10, and YOLOv11 variants.
- train.ipynb: Test notebook for checking environments, loading weights, and running single training test runs.

Documentation:
- FOLDER_STRUCTURE.txt: Guide explaining model checkpoints directories inside the runs folder.

Directories:
- utils: Helper files containing dataset loaders (dataset_handler.py), physical dimension calculators (depth_estimator.py), and safety classification models (severity_calculator.py).
