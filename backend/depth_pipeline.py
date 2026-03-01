"""
Integrated Depth Estimation Pipeline
Combines calibration, MiDaS depth, and hybrid estimation for production deployment
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import logging

from calibration import CameraCalibration, MonocularDepthEstimator
from midas_depth_estimator import HybridDepthEstimator

logger = logging.getLogger(__name__)


class UnifiedDepthPipeline:
    """
    Complete depth estimation pipeline
    
    Uses all available methods:
    1. Reference marker calibration for width
    2. MiDaS depth map for primary depth
    3. Hybrid fallback with heuristic methods
    4. Confidence-based method selection
    """
    
    def __init__(self, 
                 camera_height_cm: float = 100.0,
                 use_midas: bool = True,
                 use_hybrid: bool = True):
        """
        Initialize unified pipeline
        
        Args:
            camera_height_cm: Camera height above road
            use_midas: Whether to try MiDaS depth
            use_hybrid: Whether to use hybrid estimator
        """
        self.camera_height_cm = camera_height_cm
        
        # Initialize calibration
        self.calibration = CameraCalibration()
        logger.info("[Pipeline] CameraCalibration initialized")
        
        # Initialize depth estimation
        if use_hybrid:
            self.depth_estimator = HybridDepthEstimator(
                use_midas=use_midas,
                midas_model_type="small",
                device="cuda:0"
            )
            logger.info("[Pipeline] HybridDepthEstimator initialized")
        else:
            self.depth_estimator = MonocularDepthEstimator(
                camera_height_cm=camera_height_cm
            )
            logger.info("[Pipeline] MonocularDepthEstimator initialized")
        
        self.use_hybrid = use_hybrid
    
    def measure_pothole(self,
                       image: np.ndarray,
                       detection: Dict[str, Any],
                       method: str = "auto") -> Dict[str, Any]:
        """
        Measure pothole dimensions using unified pipeline
        
        Args:
            image: Full road image
            detection: Detection dict with 'bbox' key
                      bbox format: [x1, y1, x2, y2]
            method: "auto", "midas", "heuristic", or "hybrid"
            
        Returns:
            Dict with width_cm, depth_cm, confidence, methods used
        """
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Width measurement with calibration
            width_px = x2 - x1
            width_cm = self.calibration.pixels_to_cm(width_px)
            
            # Extract crops for depth estimation
            pothole_crop = image[y1:y2, x1:x2]
            
            # Reference region (same row, offset)
            ref_x1 = max(0, x1 - 200)
            ref_x2 = x1 - 50
            if ref_x2 > ref_x1:
                road_crop = image[y1:y2, ref_x1:ref_x2]
            else:
                road_crop = image[max(0, y1-100):y1, x1:x2]
            
            # Depth measurement
            if method == "auto" and self.use_hybrid:
                depth_result = self.depth_estimator.estimate_depth(
                    image, bbox
                )
            elif method == "midas" and self.use_hybrid:
                depth_result = self.depth_estimator.midas.estimate_pothole_depth(
                    image, bbox
                )
            elif method == "heuristic" or not self.use_hybrid:
                depth_result = self.depth_estimator.estimate_relative_depth(
                    pothole_crop, road_crop
                )
            else:
                depth_result = self.depth_estimator.estimate_depth(image, bbox)
            
            # Compile results
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
            
            logger.info(f"[Pipeline] Measured: {width_cm:.1f}cm W × {result['depth_cm']}cm D")
            return result
            
        except Exception as e:
            logger.error(f"[Pipeline] Measurement failed: {e}")
            return {
                'bbox': detection.get('bbox', [0, 0, 0, 0]),
                'width_cm': 'N/A',
                'depth_cm': 'N/A',
                'confidence': 0,
                'error': str(e)
            }
    
    def batch_measure(self,
                     image: np.ndarray,
                     detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Measure multiple potholes in single image
        
        Args:
            image: Road image
            detections: List of detection dicts
            
        Returns:
            List of measurement results
        """
        results = []
        
        logger.info(f"[Pipeline] Processing {len(detections)} detections...")
        
        for i, detection in enumerate(detections):
            try:
                result = self.measure_pothole(image, detection)
                results.append(result)
                logger.debug(f"  [{i+1}/{len(detections)}] Processed")
            except Exception as e:
                logger.error(f"  [{i+1}/{len(detections)}] Failed: {e}")
                results.append({
                    'bbox': detection.get('bbox'),
                    'error': str(e)
                })
        
        logger.info(f"[Pipeline] Completed {len(results)} measurements")
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get complete pipeline status"""
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
        
        # Pipeline is ready if calibration has pixels_per_cm
        status['pipeline_ready'] = (
            status['calibration'].get('calibrated', False) and
            (status['depth_estimator'].get('midas_initialized', False) or
             status['depth_estimator'].get('type') == 'monocular_heuristic')
        )
        
        return status


# Integration for Flask/FastAPI
def create_detection_response(image: np.ndarray,
                             detections: List[Dict],
                             pipeline: UnifiedDepthPipeline) -> Dict[str, Any]:
    """
    Create response for detection endpoint
    
    Args:
        image: Input image
        detections: List of raw detections from YOLO
        pipeline: Initialized UnifiedDepthPipeline
        
    Returns:
        Response dict ready for JSON serialization
    """
    try:
        # Convert YOLO detections to our format if needed
        formatted_detections = []
        for det in detections:
            if isinstance(det, dict) and 'bbox' in det:
                formatted_detections.append(det)
            elif hasattr(det, 'xyxy'):  # ultralytics format
                x1, y1, x2, y2 = det.xyxy[0]
                formatted_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(det.conf[0]) if hasattr(det, 'conf') else 0.5
                })
        
        # Measure all potholes
        measurements = pipeline.batch_measure(image, formatted_detections)
        
        # Build response
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
        
        # Calculate summary stats
        valid_widths = [m['width_cm'] for m in measurements if isinstance(m.get('width_cm'), (int, float))]
        valid_depths = [m['depth_cm'] for m in measurements if isinstance(m.get('depth_cm'), (int, float))]
        
        if valid_widths:
            response['summary']['avg_width_cm'] = float(np.mean(valid_widths))
        
        if valid_depths:
            response['summary']['avg_depth_cm'] = float(np.mean(valid_depths))
        
        response['summary']['high_confidence'] = sum(
            1 for m in measurements if m.get('confidence', 0) > 0.7
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Response creation failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'measurements': []
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("UNIFIED DEPTH PIPELINE - INTEGRATION TEST")
    print("="*80)
    
    # Initialize pipeline
    pipeline = UnifiedDepthPipeline(
        camera_height_cm=100.0,
        use_midas=True,
        use_hybrid=True
    )
    
    # Check status
    print("\nPipeline Status:")
    status = pipeline.get_pipeline_status()
    print(f"  Calibrated: {status['calibration']['calibrated']}")
    print(f"  MiDaS Available: {status['depth_estimator'].get('midas_initialized', 'N/A')}")
    print(f"  Pipeline Ready: {status['pipeline_ready']}")
    
    # Test with synthetic image
    print("\nTesting with synthetic image...")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :] = 150  # Gray road
    image[200:300, 200:300] = 50  # Dark pothole
    
    test_detection = {
        'bbox': [200, 200, 300, 300],
        'confidence': 0.95
    }
    
    result = pipeline.measure_pothole(image, test_detection)
    print(f"\nMeasurement Result:")
    print(f"  Width: {result['width_cm']:.1f} cm")
    print(f"  Depth: {result['depth_cm']} cm")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Method: {result['depth_method']}")
    
    print("\n" + "="*80)
    print("Integration test complete!")
    print("="*80)
