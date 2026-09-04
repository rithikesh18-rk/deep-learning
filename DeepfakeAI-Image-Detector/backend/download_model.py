"""Model Downloader for Render Build & Deployment.
Ensures genuine trained weights for deepfake_detector_improved.pth exist on disk.
If the model file is missing, empty, or a Git LFS pointer, it downloads the genuine 113MB checkpoint from GitHub media storage.
"""
import os
import sys
import urllib.request
from pathlib import Path

MODEL_FILENAME = "deepfake_detector_improved.pth"
MIN_FILE_SIZE = 50 * 1024 * 1024  # genuine file is ~113.9 MB, pointer is ~130 bytes
LFS_MEDIA_URL = (
    "https://media.githubusercontent.com/media/rithikesh18-rk/deep-learning/main/"
    "DeepfakeAI-Image-Detector/backend/models/deepfake_detector_improved.pth"
)

def ensure_model(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / MODEL_FILENAME

    if model_path.exists():
        size = model_path.stat().st_size
        if size >= MIN_FILE_SIZE:
            print(f"[MODEL CHECK] Valid trained model exists at {model_path} ({size} bytes).")
            return model_path
        print(f"[MODEL CHECK] Existing file at {model_path} is too small ({size} bytes, likely Git LFS pointer). Re-downloading...")
    else:
        print(f"[MODEL CHECK] Checkpoint not found at {model_path}. Downloading...")

    temp_path = target_dir / f"{MODEL_FILENAME}.tmp"
    req = urllib.request.Request(LFS_MEDIA_URL, headers={"User-Agent": "Mozilla/5.0"})
    print(f"[MODEL DOWNLOAD] Streaming genuine checkpoint from {LFS_MEDIA_URL}...")
    with urllib.request.urlopen(req, timeout=180) as resp, open(temp_path, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (10 * 1024 * 1024) < 1024 * 1024:
                print(f"[MODEL DOWNLOAD] Downloaded {total // (1024 * 1024)} MB...")

    temp_path.replace(model_path)
    final_size = model_path.stat().st_size
    print(f"[MODEL DOWNLOAD COMPLETE] Saved {model_path} ({final_size} bytes).")
    return model_path

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    models_dir = base / "models"
    ensure_model(models_dir)
