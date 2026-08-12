import torch
import os
from ultralytics import YOLO

def update_model_classes():
    model_path = os.path.join("models", "best.pt")
    print(f"Loading {model_path}...")
    
    # Load checkpoint with PyTorch
    ckpt = torch.load(model_path, map_location="cpu")
    print("Keys in checkpoint:", ckpt.keys() if isinstance(ckpt, dict) else type(ckpt))

    if isinstance(ckpt, dict) and "model" in ckpt:
        model_obj = ckpt["model"]
        print("Model obj names before:", getattr(model_obj, "names", None))
        # Update names dictionary on the PyTorch model object inside checkpoint
        model_obj.names = {0: "With Helmet", 1: "Without Helmet"}
        torch.save(ckpt, model_path)
        print("Successfully updated PyTorch checkpoint model.names!")
    else:
        # Fallback using ultralytics YOLO API
        model = YOLO(model_path)
        model.model.names = {0: "With Helmet", 1: "Without Helmet"}
        model.save(model_path)
        print("Updated model.names via YOLO save!")

    # Verify updated model
    model = YOLO(model_path)
    print("VERIFIED model.names:", model.names)

if __name__ == "__main__":
    update_model_classes()
