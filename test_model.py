import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

MODEL_PATH = os.path.join("models", "best.pt")
TEST_IMAGE_DIR = "test_images"
RESULT_DIR = os.path.join("results", "model_test")

os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 70)
print(" YOLO MODEL VALIDATION TEST ")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    exit(1)

model = YOLO(MODEL_PATH)
print(f"\nModel Path: {MODEL_PATH}")
print(f"Model Task: {model.model.task if hasattr(model, 'model') else 'unknown'}")
print(f"Model Input Size: {model.model.args.get('imgsz', 640) if hasattr(model, 'model') and hasattr(model.model, 'args') else 'unknown'}")
print(f"Number of Classes: {len(model.names)}")
print(f"Class Mapping (model.names): {model.names}")

image_files = []
if os.path.exists(TEST_IMAGE_DIR):
    for f in os.listdir(TEST_IMAGE_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(os.path.join(TEST_IMAGE_DIR, f))

image_files.sort()

if not image_files:
    print(f"\nNo test images found in '{TEST_IMAGE_DIR}/'.")
    print("Please place 5-10 sample JPG/JPEG/PNG images in the 'test_images/' folder and re-run this script.")
    exit(0)

print(f"\nFound {len(image_files)} test image(s). Running inference...")
print("-" * 70)

total_detections = 0
total_with_helmet = 0
total_without_helmet = 0
confidence_sum = 0.0
confidence_count = 0

for img_path in image_files:
    filename = os.path.basename(img_path)
    print(f"\nFilename: {filename}")

    try:
        pil_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
    except Exception as e:
        print(f"  ERROR opening image: {e}")
        continue

    results = model(pil_img, conf=0.50, iou=0.45, verbose=False)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"Class_{cls_id}")
            conf = float(box.conf[0])
            xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "box": xyxy
            })

    if detections:
        for det in detections:
            print(f"  Detected class: {det['class_name']} (ID: {det['class_id']})")
            print(f"  Confidence: {det['confidence']:.4f}")
            print(f"  Bounding box: {det['box']}")
            total_detections += 1
            confidence_sum += det['confidence']
            confidence_count += 1

            name_lower = det['class_name'].lower()
            if "with" in name_lower and "helmet" in name_lower and "without" not in name_lower:
                total_with_helmet += 1
            elif "without" in name_lower or "no helmet" in name_lower:
                total_without_helmet += 1
    else:
        print("  No detections.")

    annotated = results[0].plot()
    out_path = os.path.join(RESULT_DIR, f"annotated_{filename}")
    cv2.imwrite(out_path, annotated)
    print(f"  Saved annotated image: {out_path}")

avg_confidence = (confidence_sum / confidence_count) if confidence_count > 0 else 0.0

print("\n" + "=" * 70)
print(" SUMMARY ")
print("=" * 70)
print(f"Total images tested: {len(image_files)}")
print(f"Total detections: {total_detections}")
print(f"With Helmet detections: {total_with_helmet}")
print(f"Without Helmet detections: {total_without_helmet}")
print(f"Average confidence: {avg_confidence:.4f}")

print("\n" + "=" * 70)
print(" MODEL CLASS MAPPING VERIFICATION ")
print("=" * 70)
print(f"Actual mapping has {len(model.names)} classes.")
expected_mapping = {0: "With Helmet", 1: "Without Helmet"}
print(f"Expected mapping: {expected_mapping}")
if model.names == expected_mapping:
    print("RESULT: Model mapping matches expected helmet classes.")
else:
    print("RESULT: Model mapping DOES NOT match expected helmet classes.")
    print("The model is detecting COCO classes, not helmet/no-helmet.")