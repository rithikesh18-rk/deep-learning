"""
Test script to verify anomaly score on normal/good bottle and defective bottle.
Ensures good bottles have anomaly score < 6.0 and defects trigger DEFECT DETECTED.
"""

from pathlib import Path
import requests
import torch
from PIL import Image

from src.feature_extractor import PatchCoreFeatureExtractor
from src.memory_bank import PatchCoreMemoryBank

ROOT = Path(__file__).resolve().parent

def test_direct_pipeline():
    print("=" * 60)
    print("1. Direct Pipeline Verification (src/memory_bank.py)")
    print("=" * 60)
    good_img_path = ROOT / "data" / "bottle" / "train" / "good" / "000.png"
    assert good_img_path.exists(), f"Image not found: {good_img_path}"

    extractor = PatchCoreFeatureExtractor()
    bank = PatchCoreMemoryBank(index_path=ROOT / "models" / "bottle_patchcore.index")

    img = Image.open(good_img_path).convert("RGB")
    tensor = extractor.preprocess_image(img)
    with torch.no_grad():
        tokens, grid = extractor(tensor)

    pred = bank.predict(tokens, grid_size=grid)
    score = pred["image_scores"]
    print(f"Sample              : {good_img_path.name}")
    print(f"Memory Bank Size    : {bank.index.ntotal:,} vectors")
    print(f"Anomaly Score       : {score:.4f}")
    print(f"Status Requirement  : Score MUST be < 6.0")

    assert score < 6.0, f"FAILED: Score {score} >= 6.0"
    print(f"[PASS] Anomaly score {score:.4f} is strictly below 6.0!")


def test_api_endpoint():
    print("\n" + "=" * 60)
    print("2. Live FastAPI Endpoint Verification (/api/inspect)")
    print("=" * 60)
    good_img_path = ROOT / "data" / "bottle" / "train" / "good" / "000.png"
    defect_img_path = ROOT / "data" / "bottle" / "test" / "broken_large" / "000.png"

    # Test Good Bottle
    with open(good_img_path, "rb") as f:
        res = requests.post("http://127.0.0.1:8000/api/inspect", files={"file": f}, data={"threshold": 7.5})
    assert res.status_code == 200, f"HTTP error {res.status_code}: {res.text}"
    data_good = res.json()
    print(f"[GOOD BOTTLE] Anomaly Score : {data_good['anomaly_score']}")
    print(f"[GOOD BOTTLE] Threshold     : {data_good['threshold']}")
    print(f"[GOOD BOTTLE] Status        : {data_good['status']}")
    print(f"[GOOD BOTTLE] Passed        : {data_good['passed']}")
    assert data_good["passed"] is True, "Good bottle should pass!"
    assert data_good["anomaly_score"] < 6.0, f"Good bottle score {data_good['anomaly_score']} >= 6.0"

    # Test Defective Bottle
    with open(defect_img_path, "rb") as f:
        res = requests.post("http://127.0.0.1:8000/api/inspect", files={"file": f}, data={"threshold": 7.5})
    assert res.status_code == 200, f"HTTP error {res.status_code}: {res.text}"
    data_defect = res.json()
    print(f"\n[DEFECT BOTTLE] Anomaly Score : {data_defect['anomaly_score']}")
    print(f"[DEFECT BOTTLE] Threshold     : {data_defect['threshold']}")
    print(f"[DEFECT BOTTLE] Status        : {data_defect['status']}")
    print(f"[DEFECT BOTTLE] Passed        : {data_defect['passed']}")
    assert data_defect["passed"] is False, "Defective bottle should fail!"
    assert data_defect["anomaly_score"] >= 7.5, "Defective bottle score should be >= 7.5"

    print("\n[SUCCESS] End-to-end verification passed with perfect nominal vs defect separation!")

if __name__ == "__main__":
    test_direct_pipeline()
    test_api_endpoint()
