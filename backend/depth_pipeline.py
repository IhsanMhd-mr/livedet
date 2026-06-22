"""
Unified Depth Estimation Pipeline
=================================
Combines calibration scale factors and the hybrid depth estimation logic
to measure pothole widths and depths.
"""

import cv2
import numpy as np
import logging

from calibration import CameraCalibration, MonocularDepthEstimator
from midas_depth_estimator import HybridDepthEstimator

logger = logging.getLogger("Pipeline")


class UnifiedDepthPipeline:
    """Coordinates calibration and relative depth estimation components"""
    
    def __init__(self, camera_height_cm=100.0, use_midas=True, use_hybrid=True):
        self.camera_height_cm = camera_height_cm
        
        # Load scale calibration
        self.calibration = CameraCalibration()
        logger.info("CameraCalibration initialized")
        
        # Configure hybrid depth models
        if use_hybrid:
            self.depth_estimator = HybridDepthEstimator(
                use_midas=use_midas,
                midas_model_type="small",
                device="cuda:0"
            )
            logger.info("HybridDepthEstimator initialized")
        else:
            self.depth_estimator = MonocularDepthEstimator(
                camera_height_cm=camera_height_cm
            )
            logger.info("MonocularDepthEstimator initialized")
        
        self.use_hybrid = use_hybrid
    
    def measure_pothole(self, image, detection, method="auto"):
        """
        Calculates both width (via calibration) and depth (via hybrid/MiDaS model)
        """
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # 1. Width Calculation (converts pixels to cm using the calibration factor)
            width_px = x2 - x1
            width_cm = self.calibration.pixels_to_cm(width_px)
            
            # Crop pothole and reference road region for heuristic fallback check
            pothole_crop = image[y1:y2, x1:x2]
            
            # Select reference road slice
            ref_x1 = max(0, x1 - 200)
            ref_x2 = x1 - 50
            if ref_x2 > ref_x1:
                road_crop = image[y1:y2, ref_x1:ref_x2]
            else:
                road_crop = image[max(0, y1-100):y1, x1:x2]
            
            # 2. Depth Calculation (chooses model depending on method)
            if method == "auto" and self.use_hybrid:
                depth_result = self.depth_estimator.estimate_depth(image, bbox)
            elif method == "midas" and self.use_hybrid:
                depth_result = self.depth_estimator.midas.estimate_pothole_depth(image, bbox)
            elif method == "heuristic" or not self.use_hybrid:
                # Basic relative comparison
                depth_result = self.depth_estimator.estimate_relative_depth(pothole_crop, road_crop)
            else:
                depth_result = self.depth_estimator.estimate_depth(image, bbox)
            
            # Compile results dictionary
            result = {
                'bbox': bbox,
                'width_cm': width_cm,
                'width_px': width_px,
                'depth_cm': depth_result.get('depth_cm', 'N/A'),
                'confidence': depth_result.get('confidence', 0),
                'depth_method': depth_result.get('method', 'unknown'),
                'calibrated': True,
                'source': depth_result.get('source', 'unknown'),
                'detection': detection
            }
            
            logger.info(f"Measured pothole: {width_cm:.1f}cm Width x {result['depth_cm']}cm Depth")
            return result
            
        except Exception as e:
            logger.error(f"Pothole measurement failed: {e}")
            return {
                'bbox': detection.get('bbox', [0, 0, 0, 0]),
                'width_cm': 'N/A',
                'depth_cm': 'N/A',
                'confidence': 0,
                'error': str(e)
            }
            
    def batch_measure(self, image, detections):
        """Processes multiple pothole detections in a single frame"""
        results = []
        logger.info(f"Processing batch containing {len(detections)} potholes")
        
        for i, detection in enumerate(detections):
            try:
                result = self.measure_pothole(image, detection)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item failed: {e}")
                results.append({
                    'bbox': detection.get('bbox'),
                    'error': str(e)
                })
        return results
        
    def get_pipeline_status(self):
        status = {
            'calibration': self.calibration.get_calibration_status(),
            'depth_estimator': None,
            'pipeline_ready': False
        }
        
        if self.use_hybrid:
            status['depth_estimator'] = self.depth_estimator.get_status()
        else:
            status['depth_estimator'] = {
                'type': 'monocular_heuristic',
                'camera_height_cm': self.camera_height_cm
            }
            
        # Ready if calibration is successful
        status['pipeline_ready'] = (
            status['calibration'].get('calibrated', False) and
            (status['depth_estimator'].get('midas_initialized', False) or
             status['depth_estimator'].get('type') == 'monocular_heuristic')
        )
        return status


def create_detection_response(image, detections, pipeline):
    """Formats raw YOLO results and pipeline measurements for web/API output"""
    try:
        formatted_detections = []
        for det in detections:
            if isinstance(det, dict) and 'bbox' in det:
                formatted_detections.append(det)
            elif hasattr(det, 'xyxy'):  # Handle ultralytics Box object
                x1, y1, x2, y2 = det.xyxy[0]
                formatted_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(det.conf[0]) if hasattr(det, 'conf') else 0.5
                })
                
        measurements = pipeline.batch_measure(image, formatted_detections)
        
        response = {
            'status': 'success',
            'total_potholes': len(measurements),
            'measurements': measurements,
            'pipeline_status': pipeline.get_pipeline_status(),
            'summary': {
                'avg_width_cm': None,
                'avg_depth_cm': None,
                'high_confidence': 0
            }
        }
        
        # Calculate summary indicators
        widths = [m['width_cm'] for m in measurements if isinstance(m.get('width_cm'), (int, float))]
        depths = [m['depth_cm'] for m in measurements if isinstance(m.get('depth_cm'), (int, float))]
        
        if widths:
            response['summary']['avg_width_cm'] = float(np.mean(widths))
        if depths:
            response['summary']['avg_depth_cm'] = float(np.mean(depths))
            
        response['summary']['high_confidence'] = sum(
            1 for m in measurements if m.get('confidence', 0) > 0.7
        )
        
        return response
    except Exception as e:
        logger.error(f"Failed to generate API response dict: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'measurements': []
        }
