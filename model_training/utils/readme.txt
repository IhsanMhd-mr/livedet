LIVE-DETECTION-ML2 MODEL TRAINING UTILITIES
===========================================

This package contains utility modules for the live-detection-ML2 pothole detection system.

Modules
--------------------------------------------------

=== 1. depthestimator.py ===

Estimates physical dimensions of potholes from bounding box pixels.

from utils.depthestimator import DepthEstimator

estimator = DepthEstimator(lanewidthcm=120, cameraheightcm=150)
widthcm = estimator.estimatewidth(bboxwidthpx=100, imagewidthpx=640)
detections = estimator.processdetections(detections, imageshape=(480, 640, 3))

=== 2. severitycalculator.py ===

Calculates severity scores and vehicle-specific recommendations.

from utils.severitycalculator import SeverityCalculator

calc = SeverityCalculator()
score, severityclass, impact = calc.calculatescore(widthcm=80, depthcm=12, confidence=0.85)
recommendations = calc.getvehiclerecommendation(severityclass, widthcm, depthcm)

=== 3. datasethandler.py ===

Manages YOLO dataset organization and format conversion.

from utils.datasethandler import DatasetHandler

handler = DatasetHandler('rawimages', 'dataset')
handler.createyolostructure(trainratio=0.7, valratio=0.15, testratio=0.15)
handler.createdatayaml(numclasses=1, class_names=['pothole'])
