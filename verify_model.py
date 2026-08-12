import os
import cv2
import numpy as np
from ultralytics import YOLO

def verify_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "best.pt")
    print(f"--- INSPECTING MODEL FILE: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    model = YOLO(model_path)
    print("Model loaded successfully!")
    print(f"model.names: {model.names}")
    print(f"Number of classes: {len(model.names)}")

    # Create dummy image for inference test
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.circle(test_img, (320, 320), 100, (0, 255, 255), -1) # Yellow circle

    results = model(test_img, conf=0.25, verbose=True)
    print("\n--- INFERENCE TEST OUTPUT ---")
    for r in results:
        boxes = r.boxes
        print(f"Detected boxes count: {len(boxes)}")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"Class_{cls_id}")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f" Detection #{i+1}: class_id={cls_id}, class_name='{cls_name}', conf={conf:.4f}, box={xyxy}")

if __name__ == "__main__":
    verify_model()
