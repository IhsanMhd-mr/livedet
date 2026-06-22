"""
YOLO Object Detector Wrapper
"""

import os
import logging
import cv2
import numpy as np
import torch

# Standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Detector")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("Ultralytics YOLO module loaded")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not found")


class ObjectDetector:
    
    def __init__(self, model_path=None, model_type="yolov11s", device="cpu", confidence_threshold=0.5, iou_threshold=0.45):
        if not YOLO_AVAILABLE:
            raise ImportError("Please install ultralytics using: pip install ultralytics")
            
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_type = model_type
        self.model = None
        self.use_custom_model = False
        self.loaded_model_name = None

        self.load_model_file(model_path)

    def load_model_file(self, model_path):
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model path: {model_path}")
                self.model = YOLO(model_path)
                self.use_custom_model = True
                self.loaded_model_name = f"Custom: {os.path.basename(model_path)}"
            else:
                logger.info(f"Model path not found. Falling back to default: {self.model_type}")
                self.model = YOLO(f"{self.model_type}.pt")
                self.use_custom_model = False
                self.loaded_model_name = f"Pretrained: {self.model_type}"

            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def detect(self, image, confidence_threshold=None, iou_threshold=None):
        try:
            # Handle image path vs image array
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    return {"status": "error", "error": f"Could not read image: {image}"}
            else:
                img = image

            # Convert BGR to RGB for YOLO
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            height, width = img_rgb.shape[:2]
            channels = img_rgb.shape[2] if len(img_rgb.shape) == 3 else 1
            threshold = confidence_threshold or self.confidence_threshold
            iou = iou_threshold or self.iou_threshold

            # Run YOLO model inference
            results = self.model(img_rgb, conf=threshold, iou=iou, device=self.device, verbose=False)

            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Get class name and map it to Pothole
                        cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        if cls_name.lower() in ["plain", "pothole", "pot"]:
                            cls_name = "Pothole"

                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": cls_name
                        })

            return {
                "status": "success",
                "detections": detections,
                "image_shape": [height, width, channels],
                "total_detections": len(detections),
                "model": self.loaded_model_name
            }

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"status": "error", "error": str(e)}

    def annotate_image(self, image, detections):
        annotated = image.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            conf = det.get("confidence", 0.0)
            label = det.get("class_name", "object")
            
            # Draw green rectangle and label text
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label} ({conf:.2f})"
            cv2.putText(annotated, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        return annotated

    @property
    def class_names(self):
        if self.model:
            return self.model.names
        return {}

    @property
    def is_ready(self):
        return self.model is not None
