import os
import sys
from ultralytics import YOLO

def initialize_model():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    best_pt_path = os.path.join(model_dir, "best.pt")

    if not os.path.exists(best_pt_path):
        print(f"Creating default model weights file at: {best_pt_path}")
        model = YOLO("yolov8n.pt")
        model.save(best_pt_path)
        print("Model initialized and saved to models/best.pt successfully!")
    else:
        print(f"Model already exists at {best_pt_path}")

if __name__ == "__main__":
    initialize_model()
