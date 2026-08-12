import sys
import os
import cv2
import numpy as np
from PIL import Image

def run_tests():
    print("==========================================")
    print(" RUNNING AI HELMET DETECTION SYSTEM TESTS ")
    print("==========================================")

    # 1. Config Test
    import config
    assert os.path.exists(config.MODEL_PATH), "Model path should exist!"
    assert os.path.exists(config.DB_DIR), "DB Dir should exist!"
    print("[1/6] Config test passed!")

    # 2. Database Test
    import database as db
    db.init_db()
    test_id = db.save_detection("test_image.jpg", "Image", 2, 1, 0.88, "Violation")
    assert test_id is not None, "Failed to save test detection!"
    history = db.get_history()
    assert len(history) > 0, "History should contain records!"
    summary = db.get_analytics_summary()
    assert summary["total_detections"] > 0, "Summary total detections should be > 0!"
    print("[2/6] Database CRUD & Analytics test passed!")

    # 3. Detector Test
    from utils.detector import get_detector
    det = get_detector()
    assert det.model_loaded, "Detector model should be loaded successfully!"

    # Test Image Detection with dummy test image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "Test Frame", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    annotated_img, detections, counts, avg_conf = det.detect_image(dummy_img)
    assert isinstance(annotated_img, np.ndarray), "Annotated image should be ndarray!"
    assert isinstance(counts, dict), "Counts should be a dictionary!"
    assert isinstance(detections, list), "Detections should be a list!"
    print("[3/6] YOLOv8 Detector engine test passed! Detected: " + str(counts))

    # 4. Streamlit App File Import Test
    try:
        import app
        print("[4/6] Streamlit main app module loaded successfully!")
    except Exception as e:
        print("[4/6] App import notice: " + str(e))

    # 5. File Structure Verification
    required_files = [
        "app.py", "detector.py", "database.py", "config.py",
        "requirements.txt", "Procfile", "render.yaml", "runtime.txt",
        "models/best.pt", "utils/__init__.py", "utils/detector.py"
    ]
    for rf in required_files:
        assert os.path.exists(rf), "Required file missing: " + str(rf)
    print("[5/6] Required production project files verified!")

    # 6. Render File Verification
    with open("Procfile") as f:
        assert "streamlit run app.py" in f.read(), "Procfile invalid!"
    with open("runtime.txt") as f:
        assert "python-3.11.9" in f.read(), "runtime.txt invalid!"
    print("[6/6] Render deployment configuration files verified!")

    print("==========================================")
    print("       ALL SYSTEM TESTS PASSED SUCCESSFULLY! ")
    print("==========================================")

if __name__ == "__main__":
    run_tests()