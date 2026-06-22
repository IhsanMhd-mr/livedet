"""
Camera Calibration & Fallback Depth Estimation Module
======================================================
This module handles:
1. Reference marker detection (finding the 10cm white square on the road)
2. Converting pixel widths to real-world centimeters (width calibration)
3. Fallback depth estimation heuristics (shadow, texture, size)
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path


class CameraCalibration:
    """Manages reference marker scale calculations for size estimation"""
    
    def __init__(self, calibration_file=None):
        self.calibration_file = calibration_file or "camera_calibration.json"
        
        # Default starting values
        self.camera_height_cm = 100.0  # Camera height above ground
        self.camera_tilt_angle = 30.0  # Degrees tilted down
        self.reference_marker_size_cm = 10.0  # 10x10 cm square size
        self.pixels_per_cm = None  # Pixels per centimeter scale factor
        
        self.load_calibration()
    
    def load_calibration(self):
        """Loads camera scale details from saved JSON"""
        try:
            if Path(self.calibration_file).exists():
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                    self.camera_height_cm = data.get('camera_height_cm', 100.0)
                    self.camera_tilt_angle = data.get('camera_tilt_angle', 30.0)
                    self.reference_marker_size_cm = data.get('reference_marker_size_cm', 10.0)
                    self.pixels_per_cm = data.get('pixels_per_cm')
                    print(f"Loaded calibration file: {self.calibration_file}")
                    return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
        return False
    
    def save_calibration(self):
        """Saves current camera scale details to JSON"""
        try:
            data = {
                'camera_height_cm': self.camera_height_cm,
                'camera_tilt_angle': self.camera_tilt_angle,
                'reference_marker_size_cm': self.reference_marker_size_cm,
                'pixels_per_cm': self.pixels_per_cm
            }
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved calibration to: {self.calibration_file}")
        except Exception as e:
            print(f"Error saving calibration: {e}")
            
    def calibrate_from_marker(self, image, marker_bbox=None):
        """
        Calibrate the system scale using a white reference marker in the image
        """
        try:
            if marker_bbox is None:
                # Attempt to auto-detect the square reference marker on the road
                marker_bbox = self._detect_reference_marker(image)
                if marker_bbox is None:
                    print("Could not locate white reference marker automatically")
                    return False
            
            x1, y1, x2, y2 = marker_bbox
            marker_width_px = x2 - x1
            marker_height_px = y2 - y1
            
            # Use average size of detected box for robustness
            avg_pixels = (marker_width_px + marker_height_px) / 2.0
            self.pixels_per_cm = avg_pixels / self.reference_marker_size_cm
            
            print("Calibration complete:")
            print(f"  Detected Marker: {marker_width_px}x{marker_height_px} pixels")
            print(f"  Centimeters Scale: {self.pixels_per_cm:.2f} px/cm")
            
            self.save_calibration()
            return True
        except Exception as e:
            print(f"Calibration run failed: {e}")
            return False
            
    def _detect_reference_marker(self, image):
        """
        Detects a high-contrast white square marker in the image using HSV filtering
        """
        try:
            # Convert BGR image to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Simple color threshold range for white color regions
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Find shape contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
                
            best_bbox = None
            max_area = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check if shape is roughly a square
                if 0.8 < aspect_ratio < 1.2 and w > 20:
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_bbox = (x, y, x + w, y + h)
                        
            return best_bbox
        except Exception as e:
            print(f"Contour detection failed: {e}")
            return None
            
    def pixels_to_cm(self, pixels):
        """Converts pixel measurements to real-world centimeters"""
        if self.pixels_per_cm is None:
            # Default fallback factor if system isn't calibrated
            return pixels / 10.0
        return pixels / self.pixels_per_cm
        
    def get_calibration_status(self):
        return {
            'calibrated': self.pixels_per_cm is not None,
            'camera_height_cm': self.camera_height_cm,
            'camera_tilt_angle': self.camera_tilt_angle,
            'reference_marker_size_cm': self.reference_marker_size_cm,
            'pixels_per_cm': self.pixels_per_cm
        }


class MonocularDepthEstimator:
    """
    Fallback Depth Estimator using custom heuristics.
    Compares the pothole image to a reference road crop.
    """
    
    def __init__(self, camera_height_cm=100.0):
        self.camera_height_cm = camera_height_cm
        
    def estimate_relative_depth(self, pothole_crop, road_sample_crop):
        """
        Estimates depth relative to the road surface.
        Combines three simple heuristics:
        1. Brightness Difference (shadows are cast inside holes)
        2. Surface Roughness (compares variance of textures)
        3. Size Proxy (typically, larger potholes are deeper)
        """
        try:
            results = {}
            
            # Heuristic 1: Shadow/Intensity Check
            road_brightness = self._get_mean_brightness(road_sample_crop)
            pothole_brightness = self._get_mean_brightness(pothole_crop)
            brightness_diff = max(0, road_brightness - pothole_brightness)
            # Scale difference to an approximate depth range (0 to 100 cm)
            depth_shadow = (brightness_diff / max(road_brightness, 1)) * 100
            
            results['shadow_depth'] = depth_shadow
            results['intensity_diff'] = brightness_diff
            
            # Heuristic 2: Texture Check (using Laplacian variance)
            road_texture = self._get_texture_roughness(road_sample_crop)
            pothole_texture = self._get_texture_roughness(pothole_crop)
            
            if pothole_texture > road_texture:
                depth_texture = (pothole_texture - road_texture) * 50
            else:
                depth_texture = 10.0 # Standard shallow default
                
            results['texture_depth'] = depth_texture
            
            # Heuristic 3: Size-based Proxy
            # Calculate pixel area of detected crop
            pot_area = pothole_crop.shape[0] * pothole_crop.shape[1]
            depth_size = np.sqrt(pot_area) / 10.0
            
            results['size_depth'] = depth_size
            
            # Fused Depth Estimation (weighted average)
            fused_depth = (depth_shadow * 0.5) + (depth_texture * 0.3) + (depth_size * 0.2)
            
            results['fused_depth'] = fused_depth
            results['confidence'] = 0.65
            results['method'] = 'monocular_surface_comparison'
            
            return results
        except Exception as e:
            print(f"Heuristic depth calculation error: {e}")
            return {'fused_depth': 'N/A', 'confidence': 0.0, 'error': str(e)}
            
    def _get_mean_brightness(self, image):
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.mean(gray))
        
    def _get_texture_roughness(self, image):
        """Uses Laplacian variance to measure texture roughness/detail"""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Higher variance = sharper detail/rougher texture
        return float(np.var(laplacian))
        
    def estimate_absolute_depth(self, relative_depth_cm):
        # Academic placeholder - in the basic case, relative equates to absolute centimeters
        return relative_depth_cm
