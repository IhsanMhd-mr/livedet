"""
LIVEDET — YOLO Object Detector
================================
Core detection module wrapping Ultralytics YOLO.
Supports custom-trained and pretrained models.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
    logger.info("[Detector] Ultralytics YOLO loaded successfully")
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("[Detector] Ultralytics YOLO not available")


class ObjectDetector:
    """
    Production-ready YOLO object detector.

    Wraps Ultralytics YOLO for inference, annotation, and model management.
    """

    # Class name mapping to fix any naming issues
    CLASS_NAME_MAP = {
        "plain": "Pot",
        "Plain": "Pot",
        "pothole": "Pot",
        "Pothole": "Pot",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "yolov8s",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to custom-trained .pt file (primary).
            model_type: YOLO variant for fallback (e.g. 'yolov8s').
            device: 'cpu' or 'cuda' or 'cuda:0'.
            confidence_threshold: Default confidence threshold.
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.model: Optional[YOLO] = None
        self.use_custom_model = False
        self.loaded_model_name: Optional[str] = None

        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            )

        self._load_model(model_path)

    # ── Model loading ───────────────────────────────────────────────────
    def _load_model(self, model_path: Optional[str]):
        """Load the YOLO model — custom-trained or pretrained fallback."""
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"[Detector] Loading custom model: {model_path}")
                self.model = YOLO(model_path)
                self.use_custom_model = True
                self.loaded_model_name = f"Custom: {os.path.basename(model_path)}"
                logger.info(f"[Detector] ✓ Custom model loaded: {model_path}")
            else:
                if model_path:
                    logger.warning(f"[Detector] Model not found: {model_path}")
                logger.info(
                    f"[Detector] Falling back to pretrained {self.model_type}"
                )
                self.model = YOLO(f"{self.model_type}.pt")
                self.use_custom_model = False
                self.loaded_model_name = f"Pretrained: {self.model_type}"
                logger.info(f"[Detector] ✓ Pretrained model loaded: {self.model_type}")

            self.model.to(self.device)

        except Exception as e:
            logger.error(f"[Detector] Model initialization failed: {e}")
            raise

    # ── Inference ───────────────────────────────────────────────────────
    def detect(
        self,
        image: Union[str, np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Run YOLO detection on an image.

        Args:
            image: File path or BGR numpy array.
            confidence_threshold: Override default threshold.

        Returns:
            dict with keys:
                - status: 'success' or 'error'
                - detections: list of {bbox, confidence, class_id, class_name}
                - image_shape: [H, W, C]
                - total_detections: int
        """
        try:
            # Load image if path
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    return {"status": "error", "error": f"Cannot load: {image}"}
            else:
                img = image

            # Ensure RGB for YOLO
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            H, W = img_rgb.shape[:2]
            C = img_rgb.shape[2] if len(img_rgb.shape) == 3 else 1
            threshold = confidence_threshold or self.confidence_threshold

            # Run inference
            results = self.model(img_rgb, conf=threshold, device=self.device, verbose=False)

            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        
                        # Apply class name mapping if needed
                        cls_name = self.CLASS_NAME_MAP.get(cls_name, cls_name)

                        detections.append(
                            {
                                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                                "bbox_xyxy": [
                                    float(x1), float(y1), float(x2), float(y2)
                                ],
                                "confidence": conf,
                                "class_id": cls_id,
                                "class_name": cls_name,
                            }
                        )

            return {
                "status": "success",
                "detections": detections,
                "image_shape": [H, W, C],
                "total_detections": len(detections),
                "model": self.loaded_model_name,
            }

        except Exception as e:
            logger.error(f"[Detector] Detection error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    # ── Annotation ──────────────────────────────────────────────────────
    def annotate_image(
        self, image: np.ndarray, detections: List[Dict]
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on an image.

        Args:
            image: BGR numpy array.
            detections: List of detection dicts from detect().

        Returns:
            Annotated image (BGR numpy array).
        """
        annotated = image.copy()

        for det in detections:
            x, y, w, h = det["bbox"]
            conf = det.get("confidence", 0.0)
            label = det.get("class_name", "object")
            color = (0, 255, 0)

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            text = f"{label} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                annotated, (x, y - th - 8), (x + tw + 4, y), color, -1
            )
            cv2.putText(
                annotated, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return annotated

    # ── Properties ──────────────────────────────────────────────────────
    @property
    def class_names(self) -> dict:
        """Return the model's class name mapping."""
        if self.model:
            return self.model.names
        return {}

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None
