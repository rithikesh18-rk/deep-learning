import os

# Re-export detector components from utils.detector for full compatibility
from utils.detector import HelmetDetector, get_detector, _detector_instance

if __name__ == "__main__":
    detector = HelmetDetector()
    print("HelmetDetector initialized. Loaded:", detector.model_loaded)
    print("Class Names:", detector.class_names)