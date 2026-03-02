# POTTY Model Training Utilities

This package contains utility modules for the POTTY pothole detection system.

## Modules

### 1. depth_estimator.py
Estimates physical dimensions of potholes from bounding box pixels.

```python
from utils.depth_estimator import DepthEstimator

estimator = DepthEstimator(lane_width_cm=120, camera_height_cm=150)
width_cm = estimator.estimate_width(bbox_width_px=100, image_width_px=640)
detections = estimator.process_detections(detections, image_shape=(480, 640, 3))
```

### 2. severity_calculator.py
Calculates severity scores and vehicle-specific recommendations.

```python
from utils.severity_calculator import SeverityCalculator

calc = SeverityCalculator()
score, severity_class, impact = calc.calculate_score(width_cm=80, depth_cm=12, confidence=0.85)
recommendations = calc.get_vehicle_recommendation(severity_class, width_cm, depth_cm)
```

### 3. dataset_handler.py
Manages YOLO dataset organization and format conversion.

```python
from utils.dataset_handler import DatasetHandler

handler = DatasetHandler('raw_images', 'dataset')
handler.create_yolo_structure(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
handler.create_data_yaml(num_classes=1, class_names=['pothole'])
```
