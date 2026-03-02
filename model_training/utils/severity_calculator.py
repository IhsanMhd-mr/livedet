"""
Severity Calculator Module
Calculates pothole severity scores and provides vehicle recommendations
"""

import logging

logger = logging.getLogger(__name__)


class SeverityCalculator:
    """
    Calculates severity scores for potholes based on dimensions and confidence.
    Provides vehicle-specific safety recommendations.
    """
    
    def __init__(self, 
                 low_threshold=33, 
                 medium_threshold=66,
                 weight_depth=0.5,
                 weight_width=0.3,
                 weight_confidence=0.2):
        """
        Initialize severity calculator with configurable thresholds.
        
        Args:
            low_threshold: Score < this = Low severity (default: 33)
            medium_threshold: Score < this = Medium severity (default: 66)
            weight_depth: Importance weight for depth (default: 0.5)
            weight_width: Importance weight for width (default: 0.3)
            weight_confidence: Importance weight for detection confidence (default: 0.2)
        """
        self.LOW_THRESHOLD = low_threshold
        self.MEDIUM_THRESHOLD = medium_threshold
        
        # Weights must sum to 1.0
        total = weight_depth + weight_width + weight_confidence
        self.WEIGHT_DEPTH = weight_depth / total
        self.WEIGHT_WIDTH = weight_width / total
        self.WEIGHT_CONFIDENCE = weight_confidence / total
        
        # Reference values for normalization
        self.MAX_DEPTH_CM = 25
        self.MAX_WIDTH_CM = 200
        
        logger.info(f"SeverityCalculator initialized: thresholds={low_threshold}/{medium_threshold}, "
                   f"weights=D{self.WEIGHT_DEPTH:.2f}/W{self.WEIGHT_WIDTH:.2f}/C{self.WEIGHT_CONFIDENCE:.2f}")
    
    def calculate_score(self, width_cm, depth_cm, confidence):
        """
        Calculate severity score (0-100).
        
        Formula:
        score = weight_depth * (depth/max_depth)*100 
              + weight_width * (width/max_width)*100 
              + weight_confidence * confidence*100
        
        Args:
            width_cm: Pothole width in cm
            depth_cm: Pothole depth in cm
            confidence: Detection confidence (0-1)
            
        Returns:
            Tuple (score: float, severity_class: str, impact: str)
        """
        # Normalize to 0-100 scale
        depth_score = min(100, (depth_cm / self.MAX_DEPTH_CM) * 100)
        width_score = min(100, (width_cm / self.MAX_WIDTH_CM) * 100)
        conf_score = confidence * 100
        
        # Weighted average
        score = (self.WEIGHT_DEPTH * depth_score +
                self.WEIGHT_WIDTH * width_score +
                self.WEIGHT_CONFIDENCE * conf_score)
        
        # Classify
        if score < self.LOW_THRESHOLD:
            severity_class = 'Low'
            impact = 'Minor damage, no immediate action needed'
        elif score < self.MEDIUM_THRESHOLD:
            severity_class = 'Medium'
            impact = 'Moderate damage, should be repaired soon'
        else:
            severity_class = 'High'
            impact = 'Critical damage, immediate repair needed'
        
        logger.debug(f"Severity: score={score:.1f}, class={severity_class}, "
                    f"w={width_cm}cm, d={depth_cm}cm, conf={confidence:.2f}")
        
        return round(score, 1), severity_class, impact
    
    def get_vehicle_recommendation(self, severity_class, width_cm, depth_cm):
        """
        Get vehicle-specific safety recommendations.
        
        Args:
            severity_class: 'Low', 'Medium', or 'High'
            width_cm: Pothole width
            depth_cm: Pothole depth
            
        Returns:
            Dict with vehicle recommendations
        """
        recommendations = {
            'Low': {
                'car': 'Safe - can pass normally',
                'suv': 'Safe - can pass normally',
                'truck': 'Safe - can pass normally',
                'speed': 'Normal speed (50+ km/h)',
                'action': 'No action required'
            },
            'Medium': {
                'car': 'Reduce speed, avoid if possible',
                'suv': 'Reduce speed, can pass with caution',
                'truck': 'Avoid this route',
                'speed': 'Reduced speed (20-40 km/h)',
                'action': 'Report to authorities for repair'
            },
            'High': {
                'car': 'Avoid completely - risk of damage/accident',
                'suv': 'Extreme caution or avoid - risk to suspension',
                'truck': 'Do not attempt - risk of axle/tire damage',
                'speed': 'Extreme caution (5-15 km/h) or avoid',
                'action': 'Immediate repair needed - report urgently'
            }
        }
        
        rec = recommendations.get(severity_class, recommendations['Low'])
        rec['dimensions'] = f'Width: {width_cm:.1f}cm, Depth: {depth_cm:.1f}cm'
        
        return rec
    
    def bulk_calculate(self, detections):
        """
        Calculate severity for multiple detections.
        
        Args:
            detections: List of detection dicts with 'width_cm', 'depth_cm', 'confidence'
            
        Returns:
            Updated list with severity scores added
        """
        for detection in detections:
            try:
                width = detection.get('width_cm', 0)
                depth = detection.get('depth_cm', 0)
                conf = detection.get('confidence', 0.5)
                
                score, sev_class, impact = self.calculate_score(width, depth, conf)
                
                detection['severity'] = {
                    'score': score,
                    'class': sev_class,
                    'impact': impact
                }
                
                detection['recommendations'] = self.get_vehicle_recommendation(
                    sev_class, width, depth
                )
            
            except Exception as e:
                logger.error(f"Error calculating severity: {e}")
                detection['severity'] = {'score': 0, 'class': 'Unknown', 'impact': 'Error'}
        
        return detections


# Severity thresholds for different standards
SEVERITY_STANDARDS = {
    'conservative': {
        'name': 'Conservative (repair early)',
        'low_threshold': 25,
        'medium_threshold': 50,
        'weight_depth': 0.6,  # Emphasize depth
        'weight_width': 0.2,
        'weight_confidence': 0.2
    },
    'standard': {
        'name': 'Standard (balanced)',
        'low_threshold': 33,
        'medium_threshold': 66,
        'weight_depth': 0.5,
        'weight_width': 0.3,
        'weight_confidence': 0.2
    },
    'aggressive': {
        'name': 'Aggressive (repair only when severe)',
        'low_threshold': 40,
        'medium_threshold': 75,
        'weight_depth': 0.4,
        'weight_width': 0.4,
        'weight_confidence': 0.2
    }
}
