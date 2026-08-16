PURPOSE & OVERVIEW
------------------
This folder holds the core training scripts and utilities required to fine-tune 
YOLO models on the LIVEDET dataset. It manages epochs, automated tuning, and 
saving the resulting weights.

FILES & SCRIPTS
---------------
- train_unified.py
  Base script to train YOLO models from scratch using a general pothole dataset.

- fine_tune.py
  Script to apply transfer learning using annotated road defect datasets.

- auto_tune.py
  Automated iterative loop that trains and tunes hyperparameters until a target 
  validation score is reached.

- train.ipynb
  Test notebook for checking environments, loading weights, and running single
  training tests.

- finetune_depthsense.ipynb
  Fine-tuning experiment notebook: continues training the YOLOv11 champion on
  06_DEPTHSENSE_STATIC_DATASET, records every run (win or lose) in
  model_registry.json, and includes a written fine-tuning-vs-from-scratch
  analysis. Promotion of a new champion is a separate, human-approved,
  commented-out cell; models/finetuned/ is never touched.

- model_registry.json
  Persistent record of training/fine-tune runs and the current champion per
  architecture (yolov8, yolov11). Populated by finetune_depthsense.ipynb.

- utils/
  Helper files containing dataset loaders, physical dimension calculators, and 
  severity classification models.
