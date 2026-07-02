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

- utils/
  Helper files containing dataset loaders, physical dimension calculators, and 
  severity classification models.
