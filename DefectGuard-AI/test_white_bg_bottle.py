"""
Test script for verifying background isolation and patch token masking
on white-background bottle images in DefectGuard (PatchCore).
"""

import sys
import io
from pathlib import Path
import numpy as np
import cv2
import torch
import requests
from PIL import Image

from src.feature_extractor import PatchCoreFeatureExtractor, isolate_foreground, generate_patch_mask
from src.memory_bank import PatchCoreMemoryBank

ROOT = Path(__file__).resolve().parent
GOOD_SAMPLE = ROOT / "data" / "bottle" / "test" / "good" / "000.png"
DEFECT_SAMPLE = ROOT / "data" / "bottle" / "test" / "broken_large" / "000.png"
INDEX_PATH = ROOT / "models" / "bottle_patchcore.index"


def create_noisy_white_bg_bottle(base_img_path: Path) -> Image.Image:
    """
    Creates a bottle image with a noisy/textured off-white background
    simulating an uncurated image downloaded from the internet.
    """
    img_bgr = cv2.imread(str(base_img_path))
    h, w, _ = img_bgr.shape

    # Segment original bottle
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bottle_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Generate synthetic noisy/textured background (gradient + gaussian noise + high frequency texture)
    y, x = np.mgrid[0:h, 0:w]
    gradient = (235 + 20 * np.sin(x / 50.0) * np.cos(y / 50.0)).astype(np.float32)
    noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
    synthetic_bg = np.clip(gradient + noise, 0, 255).astype(np.uint8)
    synthetic_bg_bgr = cv2.cvtColor(synthetic_bg, cv2.COLOR_GRAY2BGR)

    # Composite bottle on noisy background
    composite_bgr = synthetic_bg_bgr.copy()
    composite_bgr[bottle_mask > 0] = img_bgr[bottle_mask > 0]

    return Image.fromarray(cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB))


def test_isolation_and_masking():
    print("=" * 65)
    print("1. Background Isolation & Patch Masking Verification")
    print("=" * 65)

    assert GOOD_SAMPLE.exists(), f"Sample image not found: {GOOD_SAMPLE}"
    assert INDEX_PATH.exists(), f"FAISS index not found: {INDEX_PATH}"

    # Create synthetic internet image with noisy background
    noisy_img_pil = create_noisy_white_bg_bottle(GOOD_SAMPLE)
    noisy_arr = np.array(noisy_img_pil)

    # 1. Test isolate_foreground
    isolated_pil, binary_mask = isolate_foreground(noisy_img_pil)
    isolated_arr = np.array(isolated_pil)

    # Check background pixels are forced to pure black [0, 0, 0]
    bg_pixels = isolated_arr[binary_mask == 0]
    max_bg_val = bg_pixels.max()
    print(f"[TEST 1] Background pixels forced to pure black: {max_bg_val == 0} (Max BG value = {max_bg_val})")
    assert max_bg_val == 0, f"Expected background to be [0, 0, 0], got max value {max_bg_val}"

    # Check bottle foreground is preserved
    fg_pixels = isolated_arr[binary_mask == 255]
    print(f"[TEST 1] Foreground pixels preserved: {fg_pixels.mean():.1f} avg intensity")
    assert fg_pixels.mean() > 10.0, "Foreground bottle appears empty or wiped."

    # 2. Test patch mask generation
    patch_mask = generate_patch_mask(binary_mask, grid_size=(28, 28))
    fg_patches = int(np.sum(patch_mask))
    total_patches = int(patch_mask.size)
    fg_pct = (fg_patches / total_patches) * 100
    print(f"[TEST 2] Patch mask shape: {patch_mask.shape}, FG patches: {fg_patches}/{total_patches} ({fg_pct:.1f}%)")
    assert 200 <= fg_patches <= 650, f"Unexpected foreground patch count: {fg_patches}"

    # 3. Compare Anomaly Scores: Unmasked vs Masked
    extractor = PatchCoreFeatureExtractor()
    bank = PatchCoreMemoryBank(index_path=INDEX_PATH)

    # Unmasked inference on noisy image (background noise directly pollutes patch tokens)
    t_unmasked = extractor.preprocess_image(noisy_img_pil)
    with torch.no_grad():
        tokens_unmasked, grid = extractor(t_unmasked)
    pred_unmasked = bank.predict(tokens_unmasked, grid_size=grid, patch_mask=None)
    score_unmasked = pred_unmasked["image_scores"]

    # Masked inference on isolated image
    t_isolated = extractor.preprocess_image(isolated_pil)
    with torch.no_grad():
        tokens_isolated, _ = extractor(t_isolated)
    pred_masked = bank.predict(tokens_isolated, grid_size=grid, patch_mask=patch_mask, full_mask=binary_mask)
    score_masked = pred_masked["image_scores"]

    print(f"\n[EVALUATION]")
    print(f"  Unmasked Score (with BG noise) : {score_unmasked:.4f}")
    print(f"  Masked Score (with Isolation)  : {score_masked:.4f}")

    # Check that anomaly map has 0.0 in background
    anomaly_map = pred_masked["anomaly_maps"]
    mask_224 = cv2.resize(binary_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    bg_anomaly_max = float(anomaly_map[mask_224 == 0].max())
    print(f"  Max Anomaly in Background Area : {bg_anomaly_max:.4f}")
    assert bg_anomaly_max == 0.0, f"Expected 0.0 anomaly in background, got {bg_anomaly_max}"

    print("\n[PASS] Background isolation and token masking verified successfully!")


def test_api_with_white_background():
    print("\n" + "=" * 65)
    print("2. Live API Endpoint Verification (/api/inspect on port 8001)")
    print("=" * 65)

    noisy_img_pil = create_noisy_white_bg_bottle(GOOD_SAMPLE)
    buf = io.BytesIO()
    noisy_img_pil.save(buf, format="PNG")
    buf.seek(0)

    url = "http://127.0.0.1:8001/api/inspect"
    response = requests.post(url, files={"file": ("noisy_white_bottle.png", buf, "image/png")})
    assert response.status_code == 200, f"API failed with {response.status_code}: {response.text}"

    data = response.json()
    print(f"Filename           : {data.get('filename')}")
    print(f"Anomaly Score      : {data.get('anomaly_score')}")
    print(f"Threshold          : {data.get('threshold')}")
    print(f"Status             : {data.get('status')}")
    print(f"Passed             : {data.get('passed')}")
    print(f"Foreground Patches : {data.get('foreground_patches')} / {data.get('total_patches')}")
    print(f"Inference Latency  : {data.get('inference_time_ms')} ms")

    assert "heatmap_base64" in data and len(data["heatmap_base64"]) > 100
    assert "overlay_base64" in data and len(data["overlay_base64"]) > 100
    assert data.get("foreground_patches") < data.get("total_patches"), "Should only compute foreground patches"
    print("\n[PASS] Live FastAPI endpoint correctly handled white-background bottle with masking!")


if __name__ == "__main__":
    test_isolation_and_masking()
    test_api_with_white_background()
