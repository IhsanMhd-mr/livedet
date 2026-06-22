"""
MiDaS Depth Estimator & Hybrid Blending Module
===============================================
This module wraps the Intel MiDaS depth estimator and implements 
the hybrid fallback logic to switch to basic shadow metrics if 
the neural model is slow or unavailable.
"""

import numpy as np
import cv2
import logging
from pathlib import Path

logger = logging.getLogger("MidasDepth")


class MiDaSDepthEstimator:
    """Wrapper class for Intel MiDaS monocular depth model"""
    
    def __init__(self, model_type="small", device="cuda:0", model_cache_dir=None):
        self.model_type = model_type
        self.device = device
        self.model_cache_dir = model_cache_dir or str(Path.home() / '.midas_cache')
        self.model = None
        self.transform = None
        self.initialized = False
        
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        self.initialize_model()
        
    def initialize_model(self):
        """Loads MiDaS model from PyTorch Hub"""
        try:
            import torch
            
            # Ensure timm library is installed
            try:
                import timm
            except ImportError:
                import subprocess
                subprocess.check_call(['pip', 'install', 'timm', '-q'])
                
            logger.info(f"Loading MiDaS {self.model_type} model...")
            
            # Load from hub
            self.model = torch.hub.load("intel-isl/MiDaS", f"MiDaS_{self.model_type}")
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Load corresponding transform functions
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_type == "small":
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.dpt_transform
                
            self.initialized = True
            logger.info("MiDaS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MiDaS: {e}")
            self.initialized = False
            
    def estimate_depth_map(self, image):
        """Generates a relative depth map for the input image"""
        if not self.initialized:
            return None
            
        try:
            import torch
            
            # Preprocess and forward pass
            input_batch = self.transform(image).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0.0 - 1.0 range
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            depth_map = 1.0 - depth_map # Invert so larger value is deeper
            
            return depth_map
        except Exception as e:
            logger.error(f"Depth map computation failed: {e}")
            return None
            
    def estimate_pothole_depth(self, image, pothole_bbox, reference_bbox=None):
        """Measures pothole relative depth level using depth map"""
        if not self.initialized:
            return {'depth_cm': 'N/A', 'confidence': 0.0, 'error': 'Model uninitialized'}
            
        try:
            depth_map = self.estimate_depth_map(image)
            if depth_map is None:
                return {'depth_cm': 'N/A', 'confidence': 0.0, 'error': 'Depth map failed'}
                
            x1, y1, x2, y2 = pothole_bbox
            
            # Get mean depth in pothole region
            pothole_depth_values = depth_map[y1:y2, x1:x2]
            pothole_depth_mean = np.mean(pothole_depth_values)
            pothole_depth_std = np.std(pothole_depth_values)
            
            # Get reference depth from surrounding road
            if reference_bbox is not None:
                rx1, ry1, rx2, ry2 = reference_bbox
                road_depth_mean = np.mean(depth_map[ry1:ry2, rx1:rx2])
            else:
                # Fallback: bottom quarter slice of image represents road surface
                h, w = image.shape[:2]
                road_crop_h = h // 4
                road_depth_mean = np.mean(depth_map[h-road_crop_h:h, :])
                
            # Compute relative difference and scale to cm metric
            depth_diff = (road_depth_mean - pothole_depth_mean)
            relative_depth_cm = depth_diff * 100.0
            
            # Lower deviation of depth = higher confidence
            confidence = max(0.5, 1.0 - (pothole_depth_std * 2.0))
            
            return {
                'depth_cm': relative_depth_cm,
                'confidence': float(confidence),
                'method': f'midas_{self.model_type}'
            }
        except Exception as e:
            logger.error(f"Relative depth calculation failed: {e}")
            return {'depth_cm': 'N/A', 'confidence': 0.0, 'error': str(e)}


class HybridDepthEstimator:
    """
    Combines MiDaS relative depth estimation with a pixel-intensity heuristic fallback
    """
    
    def __init__(self, use_midas=True, midas_model_type="small", device="cuda:0", fallback_weight=0.3):
        self.fallback_weight = fallback_weight
        self.midas = None
        
        if use_midas:
            self.midas = MiDaSDepthEstimator(model_type=midas_model_type, device=device)
            
    def estimate_depth(self, image, pothole_bbox, reference_bbox=None):
        # 1. Try MiDaS model first
        if self.midas and self.midas.initialized:
            midas_res = self.midas.estimate_pothole_depth(image, pothole_bbox, reference_bbox)
            if midas_res.get('confidence', 0) > 0.4 and isinstance(midas_res.get('depth_cm'), (int, float)):
                return {
                    **midas_res,
                    'source': 'midas_primary',
                    'fallback_used': False
                }
                
        # 2. Fallback to heuristic check
        return self._estimate_heuristic(image, pothole_bbox, reference_bbox)
        
    def _estimate_heuristic(self, image, pothole_bbox, reference_bbox=None):
        """Intensity comparison heuristic fallback"""
        try:
            x1, y1, x2, y2 = pothole_bbox
            pothole_crop = image[y1:y2, x1:x2]
            
            # Shadow intensity check inside hole region
            gray = cv2.cvtColor(pothole_crop, cv2.COLOR_BGR2GRAY)
            pot_intensity = np.mean(gray)
            
            if reference_bbox:
                rx1, ry1, rx2, ry2 = reference_bbox
                road_crop = image[ry1:ry2, rx1:rx2]
            else:
                h, w = image.shape[:2]
                road_crop = image[max(0, y2):min(h, y2+100), x1:x2]
                
            if road_crop.size > 0:
                road_gray = cv2.cvtColor(road_crop, cv2.COLOR_BGR2GRAY)
                road_intensity = np.mean(road_gray)
            else:
                road_intensity = 200.0
                
            # Intensity gap maps to depth estimation (up to 50cm max)
            intensity_diff = max(0, road_intensity - pot_intensity)
            depth_cm = (intensity_diff / 255.0) * 50.0
            
            return {
                'depth_cm': depth_cm,
                'confidence': 0.5,
                'method': 'heuristic_intensity',
                'source': 'heuristic_fallback',
                'fallback_used': True
            }
        except Exception as e:
            logger.error(f"Heuristic fallback estimation failed: {e}")
            return {
                'depth_cm': 20.0,
                'confidence': 0.3,
                'method': 'heuristic_default',
                'source': 'heuristic_fallback',
                'fallback_used': True
            }
            
    def get_status(self):
        return {
            'midas_initialized': self.midas.initialized if self.midas else False,
            'fallback_weight': self.fallback_weight
        }
