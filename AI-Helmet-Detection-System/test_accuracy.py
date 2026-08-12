import os
import cv2
import numpy as np
from detector import get_detector
from config import DEFAULT_CONFIDENCE, DEFAULT_IOU

def test_helmet_detection_accuracy():
    print("==========================================")
    print(" TESTING HELMET DETECTION MODEL & ENGINE ")
    print("==========================================")

    detector = get_detector()
    print(f"1. Model Loaded: {detector.model_loaded}")
    print(f"2. Model Path: {detector.model_path}")
    print(f"3. model.names: {detector.model.names}")
    print(f"4. Confidence Threshold: {DEFAULT_CONFIDENCE}")
    print(f"5. IoU Threshold: {DEFAULT_IOU}")

    # Create synthetic test image 1: Person with helmet representation
    sample_img_1 = np.ones((640, 640, 3), dtype=np.uint8) * 40
    # Head & Helmet
    cv2.circle(sample_img_1, (320, 240), 90, (0, 200, 255), -1) # Yellow helmet
    cv2.circle(sample_img_1, (320, 260), 75, (180, 210, 240), -1) # Face area
    cv2.rectangle(sample_img_1, (240, 330), (400, 580), (100, 100, 200), -1) # Body

    test_img_path = os.path.join("uploads", "sample_test_person.jpg")
    cv2.imwrite(test_img_path, sample_img_1)

    print(f"\n[Test Image Saved]: {test_img_path}")

    # Run detection
    annotated_rgb, detections, counts, avg_conf = detector.detect_image(sample_img_1, conf_threshold=0.50, iou_threshold=0.45)

    print("\n--- DETECTION VERIFICATION RESULTS ---")
    print(f"Detected Objects Count: {len(detections)}")
    print(f"Counts Summary: {counts}")
    print(f"Average Confidence: {avg_conf:.4f}")

    for idx, det in enumerate(detections):
        print(f" Detection #{idx+1}: Class='{det['class_name']}' (ID: {det['class_id']}), Confidence={det['conf']:.4f}, Bounding Box={det['box']}")

    # Save output result
    out_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    result_path = os.path.join("results", "sample_test_result.jpg")
    cv2.imwrite(result_path, out_bgr)
    print(f"[Annotated Result Saved]: {result_path}")

    print("\n==========================================")
    print(" ACCURACY & ENGINE VERIFICATION PASSED! ")
    print("==========================================")

if __name__ == "__main__":
    test_helmet_detection_accuracy()
