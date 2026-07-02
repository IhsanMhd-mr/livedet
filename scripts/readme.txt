PURPOSE & OVERVIEW
------------------
This directory houses all standalone Python administration scripts used for 
dataset preparation, label conversion, offline data augmentation, and testing 
the YOLO engine directly.

FILES & SCRIPTS
---------------
- reorganize_datasets.py
  Administration script to verify source folder paths, check label structures, 
  and copy raw images into clean workspaces.

- prepare_annotated_dataset.py
  Script to format local images, clip bounding boxes, compile annotation indices, 
  and write dataset partition directories.

- live_detect.py
  Standalone OpenCV GUI script to capture local webcam video feed frames directly, 
  run YOLO/depth inference, and display bounding boxes in a desktop window.

- data_cleaning.py
  Clean, validate, and convert Farzad's segmentations dataset to bounding box 
  coordinates under "clean_dataset", performing offline augmentations.