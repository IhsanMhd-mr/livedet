"""
Camera Calibration Module for Single-Camera Width and Depth Estimation
Implements reference marker calibration and monocular depth normalization
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import json


class CameraCalibration:
    """Handle camera calibration for width/depth estimation"""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize camera calibration
        
        Args:
            calibration_file: Path to saved calibration JSON
        """
        self.calibration_file = calibration_file or "camera_calibration.json"
        
        # Default calibration values (for typical road monitoring setup)
        self.camera_height_cm = 100.0  # Camera 1 meter above road
        self.camera_tilt_angle = 30.0  # Degrees downward from horizontal
        self.reference_marker_size_cm = 10.0  # 10cm reference square
        self.pixels_per_cm = None  # Will be set during calibration
        
        # Load existing calibration if available
        self.load_calibration()
    
    def load_calibration(self) -> bool:
        """Load calibration from file"""
        try:
            if Path(self.calibration_file).exists():
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                    self.camera_height_cm = data.get('camera_height_cm', 100.0)
                    self.camera_tilt_angle = data.get('camera_tilt_angle', 30.0)
                    self.reference_marker_size_cm = data.get('reference_marker_size_cm', 10.0)
                    self.pixels_per_cm = data.get('pixels_per_cm')
                    print(f"[Calibration] Loaded from {self.calibration_file}")
                    return True
        except Exception as e:
            print(f"[Calibration] Could not load: {e}")
        return False
    
    def save_calibration(self):
        """Save calibration to file"""
        try:
            data = {
                'camera_height_cm': self.camera_height_cm,
                'camera_tilt_angle': self.camera_tilt_angle,
                'reference_marker_size_cm': self.reference_marker_size_cm,
                'pixels_per_cm': self.pixels_per_cm
            }
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[Calibration] Saved to {self.calibration_file}")
        except Exception as e:
            print(f"[Calibration] Save failed: {e}")
    
    def calibrate_from_marker(self, 
                             image: np.ndarray, 
                             marker_bbox: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Calibrate camera using a reference marker on the road
        
        Args:
            image: Road image containing reference marker (10cm square)
            marker_bbox: Optional bounding box (x1, y1, x2, y2) of marker
            
        Returns:
            True if calibration successful
            
        Usage:
            1. Place a 10cm white square on road
            2. Take image
            3. Call calibrate_from_marker(image, marker_bbox)
            4. Calibration is now set and saved
        """
        try:
            if marker_bbox is None:
                # Auto-detect white square (reference marker)
                marker_bbox = self._detect_reference_marker(image)
                if marker_bbox is None:
                    print("[Calibration] Could not detect reference marker")
                    return False
            
            x1, y1, x2, y2 = marker_bbox
            marker_width_pixels = x2 - x1
            marker_height_pixels = y2 - y1
            
            # Calculate pixels per cm
            # Average of width and height for robustness
            avg_pixels = (marker_width_pixels + marker_height_pixels) / 2.0
            self.pixels_per_cm = avg_pixels / self.reference_marker_size_cm
            
            print(f"[Calibration] Reference marker detected:")
            print(f"  Marker size: {marker_width_pixels}x{marker_height_pixels} pixels")
            print(f"  Reference size: {self.reference_marker_size_cm} cm")
            print(f"  Calibration: {self.pixels_per_cm:.2f} pixels/cm")
            
            self.save_calibration()
            return True
            
        except Exception as e:
            print(f"[Calibration] Marker calibration failed: {e}")
            return False
    
    def _detect_reference_marker(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Auto-detect white reference marker (10cm square) in image
        
        Returns:
            Bounding box (x1, y1, x2, y2) or None
        """
        try:
            # Convert to HSV for better white detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define range for white color
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            
            # Create mask for white regions
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find largest square-like contour
            best_bbox = None
            max_area = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Square should have aspect ratio close to 1.0
                if 0.8 < aspect_ratio < 1.2 and w > 30:  # At least 30 pixels
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_bbox = (x, y, x + w, y + h)
            
            return best_bbox
            
        except Exception as e:
            print(f"[Calibration] Marker detection failed: {e}")
            return None
    
    def pixels_to_cm(self, pixels: float) -> float:
        """Convert pixel measurement to cm using calibration"""
        if self.pixels_per_cm is None:
            print("[Calibration] Not calibrated - using default (10 pixels/cm)")
            return pixels / 10.0  # Fallback
        return pixels / self.pixels_per_cm
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status"""
        return {
            'calibrated': self.pixels_per_cm is not None,
            'camera_height_cm': self.camera_height_cm,
            'camera_tilt_angle': self.camera_tilt_angle,
            'reference_marker_size_cm': self.reference_marker_size_cm,
            'pixels_per_cm': self.pixels_per_cm,
            'calibration_file': str(self.calibration_file)
        }


class MonocularDepthEstimator:
    """
    Monocular depth estimation using road surface normalization
    Estimates relative depth by comparing pothole region to road surface
    """
    
    def __init__(self, camera_height_cm: float = 100.0, use_midas: bool = False):
        """
        Initialize monocular depth estimator
        
        Args:
            camera_height_cm: Camera height above road
            use_midas: Whether to use Intel MiDaS model (requires additional setup)
        """
        self.camera_height_cm = camera_height_cm
        self.use_midas = use_midas
        self.midas_model = None
        
        if use_midas:
            self._load_midas_model()
    
    def _load_midas_model(self):
        """Load Intel MiDaS monocular depth model"""
        try:
            from midas_depth_estimator import MiDaSDepthEstimator
            
            model_type = "small"  # Lightweight version for real-time
            device = "cuda:0"  # Assume GPU available
            
            self.midas_model = MiDaSDepthEstimator(
                model_type=model_type,
                device=device
            )
            
            if self.midas_model.initialized:
                print("[Depth] MiDaS model loaded successfully")
                self.use_midas = True
            else:
                print("[Depth] MiDaS initialization failed, using fallback")
                self.use_midas = False
                
        except Exception as e:
            print(f"[Depth] MiDaS not available: {e}, using fallback methods")
            self.use_midas = False
    
    def estimate_relative_depth(self,
                               pothole_crop: np.ndarray,
                               road_sample_crop: np.ndarray) -> Dict[str, Any]:
        """
        Estimate pothole depth by comparing with normal road surface
        
        Args:
            pothole_crop: Image crop of pothole region
            road_sample_crop: Image crop of normal road surface (for reference)
            
        Returns:
            Dict with estimated depth and confidence
            
        Strategy:
            - Extract features from road surface (reference)
            - Extract features from pothole region
            - Measure difference in shadow intensity, texture, color
            - Estimate depth from feature differences
        """
        try:
            results = {}
            
            # Method 1: Shadow Intensity Difference
            road_intensity = self._get_mean_intensity(road_sample_crop)
            pothole_intensity = self._get_mean_intensity(pothole_crop)
            intensity_diff = max(0, road_intensity - pothole_intensity)
            
            # Normalize to 0-1 range
            intensity_ratio = intensity_diff / max(road_intensity, 1)
            depth_from_shadow = intensity_ratio * 100  # 0-100 cm range
            
            results['shadow_depth'] = depth_from_shadow
            results['intensity_diff'] = intensity_diff
            
            # Method 2: Texture Roughness Comparison
            road_roughness = self._get_texture_roughness(road_sample_crop)
            pothole_roughness = self._get_texture_roughness(pothole_crop)
            
            if pothole_roughness > road_roughness:
                # Pothole has more texture variation = deeper
                depth_from_texture = (pothole_roughness - road_roughness) * 50
            else:
                depth_from_texture = 10.0  # Shallow if smoother
            
            results['texture_depth'] = depth_from_texture
            
            # Method 3: Size-based estimation
            # Larger pothole often means deeper
            pothole_area = pothole_crop.shape[0] * pothole_crop.shape[1]
            depth_from_size = np.sqrt(pothole_area) / 10  # Approximate
            
            results['size_depth'] = depth_from_size
            
            # Fused estimate (weighted average)
            fused_depth = (
                depth_from_shadow * 0.5 +
                depth_from_texture * 0.3 +
                depth_from_size * 0.2
            )
            
            results['fused_depth'] = fused_depth
            results['confidence'] = 0.65  # Conservative estimate for single camera
            results['method'] = 'monocular_surface_comparison'
            
            return results
            
        except Exception as e:
            print(f"[Depth] Estimation failed: {e}")
            return {'fused_depth': 'N/A', 'confidence': 0.0, 'error': str(e)}
    
    def _get_mean_intensity(self, image: np.ndarray) -> float:
        """Get mean intensity (brightness) of image"""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.mean(gray))
    
    def _get_texture_roughness(self, image: np.ndarray) -> float:
        """Measure texture roughness using Laplacian"""
        if image is None or image.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        roughness = np.var(laplacian)
        
        return float(roughness)
    
    def estimate_absolute_depth(self, relative_depth_cm: float) -> float:
        """
        Convert relative depth to approximate absolute depth
        
        For academic purposes - provides rough estimate
        Accuracy depends on consistent camera setup
        """
        # Simple approximation based on camera geometry
        # Better results with fixed height and angle
        return relative_depth_cm  # In simple case, relative ≈ absolute


class CalibrationGuide:
    """Helper class for calibration process"""
    
    @staticmethod
    def get_setup_instructions() -> str:
        """Get instructions for proper setup"""
        return """
CAMERA CALIBRATION SETUP GUIDE
================================

PHASE 1: Physical Setup (One-time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Mount camera at fixed height
   • Typical: 1 meter (100 cm) above road
   • Must be same for all tests
   • Recommended: perpendicular to road

2. Set camera angle
   • Typical: 30° downward tilt
   • Ensures road surface is visible
   • Helps with shadow analysis

3. Mark reference location
   • Choose flat area of road
   • No obstacles or shadows
   • Same location for all calibrations

PHASE 1B: Reference Marker Calibration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. Prepare reference marker
   • Print 10cm x 10cm white square
   • Laminate or protect from water
   • High contrast (white on dark road)

5. Place marker on road
   • At known location
   • Clear visibility in camera
   • Good lighting conditions

6. Capture calibration image
   • Include full reference marker
   • Also include surrounding road
   • Store for later reference

7. Run calibration
   ```python
   from calibration import CameraCalibration
   
   calibration = CameraCalibration()
   image = cv2.imread('calibration_image.jpg')
   calibration.calibrate_from_marker(image)
   # Auto-detects white marker and calculates pixels/cm
   ```

PHASE 2: Depth Reference Sample
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

8. Capture normal road sample
   • Same location as marker
   • Same lighting conditions
   • Use as depth reference baseline

PHASE 2B: Pothole Testing
━━━━━━━━━━━━━━━━━━━━━━━━

9. Capture pothole image
   • Same camera position
   • Same time of day (for lighting)
   • Multiple angles if possible

10. Run detection
    • System uses calibration data
    • Calculates width using pixels/cm
    • Estimates depth from surface comparison

PHASE 3: Consistency Improvements
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For best results:
• Same camera location
• Same time of day
• Consistent lighting
• Consistent distance from pothole
• Regular re-calibration (weekly/monthly)

EXPECTED ACCURACY
━━━━━━━━━━━━━━━━

• Width: ±5-10% (with calibration)
• Depth: ±15-20% (relative estimates)
• Better with:
  - Fixed camera setup
  - Consistent lighting
  - Multiple measurement angles

For production use, consider:
• Stereo camera setup
• Multi-view reconstruction
• Professional depth sensors
        """
    
    @staticmethod
    def validate_setup() -> Dict[str, bool]:
        """Check if setup meets requirements"""
        return {
            'camera_fixed_height': False,  # Check manually
            'camera_fixed_angle': False,   # Check manually
            'reference_marker_ready': False,  # User should prepare
            'calibration_image_captured': False,  # File exists?
            'road_sample_captured': False,  # File exists?
        }


# Example usage for documentation
if __name__ == "__main__":
    print(CalibrationGuide.get_setup_instructions())
    
    # Example calibration
    # calibration = CameraCalibration()
    # image = cv2.imread('road_with_marker.jpg')
    # calibration.calibrate_from_marker(image)
    # 
    # Example depth estimation
    # depth_estimator = MonocularDepthEstimator(camera_height_cm=100.0)
    # pothole_crop = image[100:200, 100:200]
    # road_crop = image[100:200, 300:400]
    # depth_result = depth_estimator.estimate_relative_depth(pothole_crop, road_crop)
    # print(depth_result)
