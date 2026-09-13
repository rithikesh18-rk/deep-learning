"""
Calibration script for DefectGuard:
1. Evaluates all images in data/bottle/test/good/ (unseen nominal bottles)
2. Evaluates all images in data/bottle/test/broken_large/ (defective bottles)
3. Computes Top-K mean scores
4. Calculates optimal decision threshold (midpoint between max good and min defect)
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image

from src.feature_extractor import PatchCoreFeatureExtractor
from src.memory_bank import PatchCoreMemoryBank

ROOT = Path(__file__).resolve().parent
TEST_GOOD_DIR = ROOT / "data" / "bottle" / "test" / "good"
TEST_DEFECT_DIR = ROOT / "data" / "bottle" / "test" / "broken_large"
INDEX_PATH = ROOT / "models" / "bottle_patchcore.index"

def main():
    print("=" * 65)
    print("DefectGuard - Threshold Calibration & Evaluation")
    print("=" * 65)

    extractor = PatchCoreFeatureExtractor()
    bank = PatchCoreMemoryBank(index_path=INDEX_PATH)
    print(f"[INIT] Loaded Memory Bank with {bank.index.ntotal:,} vectors.")

    good_images = sorted(list(TEST_GOOD_DIR.glob("*.png")))
    defect_images = sorted(list(TEST_DEFECT_DIR.glob("*.png")))

    print(f"[EVAL] Found {len(good_images)} unseen test/good images.")
    print(f"[EVAL] Found {len(defect_images)} test/broken_large images.\n")

    # 1. Evaluate test/good
    print("--- Evaluating data/bottle/test/good/ (Unseen Nominal) ---")
    good_scores = []
    with torch.no_grad():
        for p in good_images:
            img = Image.open(p).convert("RGB")
            tensor = extractor.preprocess_image(img)
            tokens, grid = extractor(tensor)
            pred = bank.predict(tokens, grid_size=grid)
            score = float(pred["image_scores"])
            good_scores.append(score)
            print(f"  {p.name}: Top-5 Mean Score = {score:.3f}")

    good_scores = np.array(good_scores)
    print(f"\n[GOOD STATS] Min: {good_scores.min():.3f} | Mean: {good_scores.mean():.3f} | Max: {good_scores.max():.3f}")

    # 2. Evaluate test/broken_large
    print("\n--- Evaluating data/bottle/test/broken_large/ (Defective) ---")
    defect_scores = []
    with torch.no_grad():
        for p in defect_images:
            img = Image.open(p).convert("RGB")
            tensor = extractor.preprocess_image(img)
            tokens, grid = extractor(tensor)
            pred = bank.predict(tokens, grid_size=grid)
            score = float(pred["image_scores"])
            defect_scores.append(score)
            print(f"  {p.name}: Top-5 Mean Score = {score:.3f}")

    defect_scores = np.array(defect_scores)
    print(f"\n[DEFECT STATS] Min: {defect_scores.min():.3f} | Mean: {defect_scores.mean():.3f} | Max: {defect_scores.max():.3f}")

    # 3. Compute optimal decision threshold
    max_good = float(good_scores.max())
    min_defect = float(defect_scores.min())
    midpoint_thresh = round((max_good + min_defect) / 2.0, 1)

    print("\n" + "=" * 65)
    print("CALIBRATION SUMMARY & RECOMMENDED THRESHOLD")
    print("=" * 65)
    print(f"Max Good Score (Upper Bound Nominal) : {max_good:.3f}")
    print(f"Min Defect Score (Lower Bound Defect) : {min_defect:.3f}")
    print(f"Separation Margin (Delta)             : {min_defect - max_good:.3f}")
    print(f"Recommended Decision Threshold        : {midpoint_thresh}")
    print("=" * 65)

    # Save to a json file for reference
    import json
    calib_data = {
        "max_good": max_good,
        "min_defect": min_defect,
        "delta": min_defect - max_good,
        "calibrated_threshold": midpoint_thresh,
        "test_good_scores": good_scores.tolist(),
        "test_defect_scores": defect_scores.tolist(),
    }
    with open(ROOT / "models" / "threshold_calibration.json", "w") as f:
        json.dump(calib_data, f, indent=2)

if __name__ == "__main__":
    main()
