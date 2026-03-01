"""
Intel MiDaS Monocular Depth Estimation for Pothole Detection
Provides high-quality relative depth maps from single images
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MiDaSDepthEstimator:
    """
    Intel MiDaS monocular depth estimator
    
    Provides better depth estimation than simple heuristics
    by using a trained neural network
    
    Models available:
    - MiDaS_small: Fast, good for real-time (max ~200MB)
    - MiDaS: Balanced quality/speed (max ~400MB)
    - MiDaS_large: High quality but slower (max ~1GB)
    """
    
    def __init__(self, 
                 model_type: str = "small",
                 device: str = "cuda:0",
                 model_cache_dir: Optional[str] = None):
        """
        Initialize MiDaS depth estimator
        
        Args:
            model_type: "small" (default), "normal", or "large"
            device: "cuda:0", "cpu", etc.
            model_cache_dir: Directory to cache downloaded models
        """
        self.model_type = model_type
        self.device = device
        self.model_cache_dir = model_cache_dir or str(Path.home() / '.midas_cache')
        self.model = None
        self.transform = None
        self.initialized = False
        
        # Create cache directory
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Try to initialize
        self._initialize_model()
    
    def _initialize_model(self):
        """Load MiDaS model"""
        try:
            import torch
            
            # Model URLs
            model_urls = {
                "small": "https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small.pt",
                "normal": "https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21.pt",
                "large": "https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_384.pt"
            }
            
            # Model input sizes
            model_sizes = {
                "small": 256,
                "normal": 384,
                "large": 384
            }
            
            logger.info(f"[MiDaS] Loading {self.model_type} model...")
            
            # Try to import timm (required for MiDaS)
            try:
                import timm
            except ImportError:
                logger.warning("[MiDaS] timm not installed, attempting installation...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'timm', '-q'])
                import timm
            
            # Load model using torch hub
            model = torch.hub.load("intel-isl/MiDaS", f"MiDaS_{self.model_type}")
            
            # Set to eval mode
            model.eval()
            model = model.to(self.device)
            
            # Load transforms
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms"
            )
            
            if self.model_type == "small":
                transform = midas_transforms.small_transform
            else:
                transform = midas_transforms.dpt_transform
            
            self.model = model
            self.transform = transform
            self.input_size = model_sizes[self.model_type]
            self.initialized = True
            
            logger.info(f"[MiDaS] Model loaded successfully (device: {self.device})")
            
        except ImportError as e:
            logger.warning(f"[MiDaS] Failed to load MiDaS: {e}")
            logger.warning("[MiDaS] Install requirements: pip install torch timm")
            self.initialized = False
        except Exception as e:
            logger.error(f"[MiDaS] Initialization error: {e}")
            self.initialized = False
    
    def estimate_depth_map(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map for entire image
        
        Args:
            image: BGR image (H, W, 3)
            
        Returns:
            Depth map (H, W) with values 0-1 (inverted, so higher = closer)
            Returns None if model not initialized
        """
        if not self.initialized:
            logger.warning("[MiDaS] Model not initialized, cannot estimate depth")
            return None
        
        try:
            import torch
            
            # Prepare input
            input_batch = self.transform(image).to(self.device)
            
            # Run model
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy
            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0-1 range (inverted: higher = closer/shallower)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            depth_map = 1.0 - depth_map  # Invert so higher = deeper
            
            return depth_map
            
        except Exception as e:
            logger.error(f"[MiDaS] Depth estimation failed: {e}")
            return None
    
    def estimate_pothole_depth(self,
                              image: np.ndarray,
                              pothole_bbox: Tuple[int, int, int, int],
                              reference_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Estimate pothole depth using depth map
        
        Args:
            image: Full road image
            pothole_bbox: Pothole bounding box (x1, y1, x2, y2)
            reference_bbox: Reference road region (optional)
            
        Returns:
            Dict with depth estimate and confidence
        """
        if not self.initialized:
            return {
                'depth_cm': 'N/A',
                'confidence': 0.0,
                'method': 'midas_unavailable',
                'error': 'MiDaS model not initialized'
            }
        
        try:
            # Get depth map
            depth_map = self.estimate_depth_map(image)
            if depth_map is None:
                return {
                    'depth_cm': 'N/A',
                    'confidence': 0.0,
                    'method': 'midas_error',
                    'error': 'Failed to compute depth map'
                }
            
            x1, y1, x2, y2 = pothole_bbox
            
            # Extract depth in pothole region
            pothole_depth_values = depth_map[y1:y2, x1:x2]
            pothole_depth_mean = np.mean(pothole_depth_values)
            pothole_depth_std = np.std(pothole_depth_values)
            
            # Reference depth (road surface)
            if reference_bbox is not None:
                rx1, ry1, rx2, ry2 = reference_bbox
                road_depth_values = depth_map[ry1:ry2, rx1:rx2]
                road_depth_mean = np.mean(road_depth_values)
            else:
                # Use bottom part of image as reference
                h, w = image.shape[:2]
                road_crop_height = h // 4
                road_depth_values = depth_map[h-road_crop_height:h, :]
                road_depth_mean = np.mean(road_depth_values)
            
            # Compute relative depth
            # MiDaS depth is normalized, scale to cm
            depth_difference = (road_depth_mean - pothole_depth_mean)
            relative_depth_cm = depth_difference * 100  # Scale to reasonable cm range
            
            # Confidence based on depth variation
            # Lower variation = higher confidence
            confidence = max(0.5, 1.0 - (pothole_depth_std * 2))
            
            return {
                'depth_cm': relative_depth_cm,
                'depth_mean': float(pothole_depth_mean),
                'road_depth_mean': float(road_depth_mean),
                'depth_std': float(pothole_depth_std),
                'confidence': float(confidence),
                'method': f'midas_{self.model_type}',
                'normalized_depth': float(pothole_depth_mean)
            }
            
        except Exception as e:
            logger.error(f"[MiDaS] Pothole depth estimation failed: {e}")
            return {
                'depth_cm': 'N/A',
                'confidence': 0.0,
                'method': 'midas_error',
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status"""
        return {
            'initialized': self.initialized,
            'model_type': self.model_type,
            'device': self.device,
            'model_cache_dir': self.model_cache_dir,
            'ready': self.initialized
        }


class HybridDepthEstimator:
    """
    Combines MiDaS depth with fallback heuristic methods
    Provides best of both worlds: accuracy when available, fallback when not
    """
    
    def __init__(self, 
                 use_midas: bool = True,
                 midas_model_type: str = "small",
                 device: str = "cuda:0",
                 fallback_weight: float = 0.3):
        """
        Initialize hybrid depth estimator
        
        Args:
            use_midas: Whether to try using MiDaS
            midas_model_type: "small", "normal", or "large"
            device: GPU device or "cpu"
            fallback_weight: Weight of heuristic fallback (0-1)
        """
        self.use_midas = use_midas
        self.fallback_weight = fallback_weight
        
        # Initialize MiDaS
        if use_midas:
            self.midas = MiDaSDepthEstimator(
                model_type=midas_model_type,
                device=device
            )
        else:
            self.midas = None
        
        logger.info(f"[Hybrid] Initialized with MiDaS: {use_midas and self.midas.initialized}")
    
    def estimate_depth(self,
                      image: np.ndarray,
                      pothole_bbox: Tuple[int, int, int, int],
                      reference_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Estimate depth using MiDaS with heuristic fallback
        
        Args:
            image: Full road image
            pothole_bbox: Pothole bounding box
            reference_bbox: Optional reference region
            
        Returns:
            Dict with depth estimate, method used, and confidence
        """
        results = {}
        
        # Try MiDaS first
        if self.midas and self.midas.initialized:
            midas_result = self.midas.estimate_pothole_depth(
                image, pothole_bbox, reference_bbox
            )
            results['midas'] = midas_result
            
            if midas_result.get('confidence', 0) > 0.4:
                # MiDaS worked well
                return {
                    **midas_result,
                    'source': 'midas_primary',
                    'fallback_used': False
                }
        
        # Fallback to heuristic method
        heuristic_result = self._estimate_heuristic(
            image, pothole_bbox, reference_bbox
        )
        results['heuristic'] = heuristic_result
        
        # If MiDaS available but low confidence, blend with heuristic
        if 'midas' in results and results['midas'].get('confidence', 0) > 0.2:
            midas_depth = results['midas'].get('depth_cm', 0)
            heuristic_depth = results['heuristic'].get('depth_cm', 0)
            
            if isinstance(midas_depth, (int, float)) and isinstance(heuristic_depth, (int, float)):
                blended_depth = (
                    midas_depth * (1 - self.fallback_weight) +
                    heuristic_depth * self.fallback_weight
                )
                confidence = (
                    results['midas'].get('confidence', 0) * (1 - self.fallback_weight) +
                    results['heuristic'].get('confidence', 0) * self.fallback_weight
                )
                
                return {
                    'depth_cm': blended_depth,
                    'confidence': confidence,
                    'method': 'hybrid_midas_heuristic',
                    'midas_depth': midas_depth,
                    'heuristic_depth': heuristic_depth,
                    'source': 'blended'
                }
        
        # Use heuristic only
        return {
            **heuristic_result,
            'source': 'heuristic_fallback',
            'midas_available': self.midas is not None
        }
    
    def _estimate_heuristic(self,
                           image: np.ndarray,
                           pothole_bbox: Tuple[int, int, int, int],
                           reference_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Simple heuristic depth estimation (fallback)"""
        try:
            x1, y1, x2, y2 = pothole_bbox
            pothole_crop = image[y1:y2, x1:x2]
            
            # Shadow analysis
            gray = cv2.cvtColor(pothole_crop, cv2.COLOR_BGR2GRAY)
            pot_intensity = np.mean(gray)
            
            # Road reference
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
                road_intensity = 200
            
            # Intensity difference to depth
            intensity_diff = max(0, road_intensity - pot_intensity)
            depth_cm = (intensity_diff / 255.0) * 50  # Scale to 0-50 cm
            
            return {
                'depth_cm': depth_cm,
                'confidence': 0.5,
                'method': 'heuristic_intensity',
                'intensity_difference': intensity_diff
            }
            
        except Exception as e:
            logger.error(f"[Hybrid] Heuristic estimation failed: {e}")
            return {
                'depth_cm': 25.0,  # Default middle value
                'confidence': 0.3,
                'method': 'heuristic_default'
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get estimator status"""
        return {
            'midas_available': self.midas is not None,
            'midas_initialized': self.midas.initialized if self.midas else False,
            'fallback_weight': self.fallback_weight,
            'midas_status': self.midas.get_status() if self.midas else None
        }


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize hybrid estimator (uses MiDaS if available, falls back to heuristic)
    estimator = HybridDepthEstimator(use_midas=True, midas_model_type="small")
    
    print(f"Status: {estimator.get_status()}")
    
    # Example with synthetic image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :] = 150  # Gray road
    image[200:300, 200:300] = 50  # Dark pothole
    
    # Estimate depth
    result = estimator.estimate_depth(image, (200, 200, 300, 300))
    print(f"\nDepth estimation result:")
    print(f"  Depth: {result.get('depth_cm'):.1f} cm")
    print(f"  Confidence: {result.get('confidence'):.2f}")
    print(f"  Method: {result.get('method')}")
    print(f"  Source: {result.get('source')}")
