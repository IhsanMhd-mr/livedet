YOLO Pothole Model Training Utilities Guide

This folder contains helper libraries that assist with data handling, depth calculations, and severity evaluations during model training.

1. dataset_handler.py
Purpose:
- Handles folder structure and file validation for custom training datasets in YOLO format.
What it does:
- Splitting: Automates splitting images and labels into train, validation, and test datasets.
- Verification: Parses annotation coordinate files to verify format completeness and check for errors.
- YAML Config: Automatically writes training configuration files (data.yaml) defining data paths, class count, and labels.

2. depth_estimator.py
Purpose:
- Estimates physical pothole dimensions in centimeters from 2D bounding boxes.
What it does:
- Width Estimation: Calculates physical width using standard lane width (default 120cm) and image width proportions:
  width_cm = (bbox_width / image_width) * lane_width_cm
- Depth Estimation: Estimates depth by comparing the bounding box area against total image area:
  - Wide area (>5%): 3-5 cm depth
  - Medium area (1-5%): 7-12 cm depth
  - Small area (<1%): 15-20 cm depth
  - The script adjusts this estimate based on aspect ratio (wide box = shallower, narrow box = deeper) and clamps it to a 2-25 cm range.

3. severity_calculator.py
Purpose:
- Computes overall danger levels and provides specific safety guidelines for drivers.
What it does:
- Severity Score: Calculates a continuous score from 0 to 100 based on weighted metrics:
  - Pothole Depth (50% importance)
  - Pothole Width (30% importance)
  - Detection Confidence (20% importance)
- Classifications: Assigns warning levels:
  - Low: Score under 33 (minor impact)
  - Medium: Score 33 to 66 (moderate impact, repair recommended)
  - High: Score above 66 (critical damage, immediate avoidance)
- Vehicle Recommendations: Provides tailored safety advice (speed limits, actions to take) depending on vehicle type (Standard Sedan vs SUV vs Heavy Truck).
