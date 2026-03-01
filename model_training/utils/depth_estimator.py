"""
Depth Estimator Module
Estimates pothole physical dimensions (width, depth) from bounding box pixels
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Monocular depth and dimension estimation for potholes.
    Uses heuristic rules based on bounding box area and aspect ratio.
    """
    
    def __init__(self, lane_width_cm=120, camera_height_cm=150):
        """
        Initialize depth estimator.
        
        Args:
            lane_width_cm: Standard road lane width in cm (default: 120 cm)
            camera_height_cm: Camera mounting height from road in cm (default: 150 cm)
        """
        self.lane_width_cm = lane_width_cm
        self.camera_height_cm = camera_height_cm
        logger.info(f"DepthEstimator initialized: lane_width={lane_width_cm}cm, height={camera_height_cm}cm")
    
    def estimate_width(self, bbox_width_px, image_width_px):
        """
        Estimate physical width of pothole in cm.
        
        Formula: width_cm = (bbox_width / image_width) * lane_width
        
        Args:
            bbox_width_px: Bounding box width in pixels
            image_width_px: Image width in pixels
            
        Returns:
            Estimated width in cm
        """
        if image_width_px <= 0:
            return 0
        width_cm = (bbox_width_px / image_width_px) * self.lane_width_cm
        return round(width_cm, 2)
    
    def estimate_depth(self, area_ratio, aspect_ratio):
        """
        Estimate depth using heuristic rules.
        
        Heuristics:
        - Large area (>5%): Shallow depth 3-5 cm (wide pothole)
        - Medium area (1-5%): Medium depth 7-12 cm
        - Small area (<1%): Deep depth 15-20 cm (concentrated)
        
        Args:
            area_ratio: (bbox_area / image_area) as fraction 0-1
            aspect_ratio: bbox_width / bbox_height
            
        Returns:
            Estimated depth in cm
        """
        if area_ratio > 0.05:
            depth_cm = 3 + (area_ratio - 0.05) * 20  # Scale 3-5
        elif area_ratio > 0.01:
            depth_cm = 7 + (area_ratio - 0.01) * 1000  # Scale 7-12
        else:
            depth_cm = 15 + (0.01 - area_ratio) * 500  # Scale 15-20
        
        # Adjust by aspect ratio (wider = shallower)
        if aspect_ratio > 2:
            depth_cm *= 0.8  # Wide pothole -> shallower
        elif aspect_ratio < 0.5:
            depth_cm *= 1.2  # Narrow pothole -> deeper
        
        return round(min(max(depth_cm, 2), 25), 2)  # Clamp 2-25 cm
    
    def process_detections(self, detections, image_shape):
        """
        Process list of detections to add dimension estimates.
        
        Args:
            detections: List of detection dicts with 'bbox' key [x, y, w, h]
            image_shape: Tuple (H, W, C)
            
        Returns:
            Updated detections list with 'width_cm' and 'depth_cm' added
        """
        H, W, C = image_shape
        image_area = H * W
        
        for detection in detections:
            try:
                x, y, w, h = detection['bbox']
                
                # Width estimation
                width_cm = self.estimate_width(w, W)
                detection['width_cm'] = width_cm
                
                # Depth estimation
                bbox_area = w * h
                area_ratio = bbox_area / image_area
                aspect_ratio = w / h if h > 0 else 1
                depth_cm = self.estimate_depth(area_ratio, aspect_ratio)
                detection['depth_cm'] = depth_cm
                
                logger.debug(f"Detection: width={width_cm}cm, depth={depth_cm}cm, area_ratio={area_ratio:.4f}")
            
            except Exception as e:
                logger.error(f"Error processing detection: {e}")
                detection['width_cm'] = 0
                detection['depth_cm'] = 0
        
        return detections


# Configuration for different road types
ROAD_CONFIGS = {
    'highway': {'lane_width_cm': 300, 'camera_height_cm': 180},
    'local_road': {'lane_width_cm': 80, 'camera_height_cm': 120},
    'urban_street': {'lane_width_cm': 120, 'camera_height_cm': 150},
    'parking_lot': {'lane_width_cm': 150, 'camera_height_cm': 150},
    'rural': {'lane_width_cm': 200, 'camera_height_cm': 160},
}
