"""
DefectGuard - Dataset Setup Script
Downloads the MVTec AD 'bottle' subcategory dataset and extracts/organizes it into:
  data/bottle/train/good/
  data/bottle/test/good/
  data/bottle/test/broken_large/
  data/bottle/test/broken_small/
  data/bottle/test/contamination/
  data/bottle/ground_truth/ (optional annotations)
"""

import os
import sys
import time
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Optional imports if installed
try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
BOTTLE_DIR = DATA_DIR / "bottle"

# Mirror candidates
HF_REPO = "foersben/mvtec-ad"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_REPO}"
HF_RAW_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# Potential direct tar.xz mirrors
TAR_MIRRORS = [
    # Direct archive mirrors if available
    "https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/bottle.tar.xz",
]


def download_file(url: str, dest_path: Path, session=None, retries: int = 3) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            if session and requests:
                resp = session.get(url, stream=True, timeout=30)
                if resp.status_code == 200:
                    with open(dest_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                    return True
                elif resp.status_code == 404:
                    return False
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "DefectGuard/0.1"})
                with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                    out_file.write(response.read())
                return True
        except Exception as e:
            if attempt == retries:
                print(f"[WARN] Failed to download {url}: {e}", file=sys.stderr)
                return False
            time.sleep(1.0 * attempt)
    return False


def try_download_archive() -> bool:
    """Attempt to download and extract a pre-packaged tar.xz archive."""
    for mirror_url in TAR_MIRRORS:
        print(f"[INFO] Checking archive mirror: {mirror_url}")
        archive_path = DATA_DIR / "bottle.tar.xz"
        try:
            req = urllib.request.Request(mirror_url, headers={"User-Agent": "DefectGuard/0.1"}, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as res:
                if res.status == 200:
                    print(f"[INFO] Downloading archive from {mirror_url}...")
                    if download_file(mirror_url, archive_path):
                        print("[INFO] Extracting archive...")
                        with tarfile.open(archive_path, "r:*") as tar:
                            tar.extractall(path=DATA_DIR)
                        if archive_path.exists():
                            archive_path.unlink()
                        return True
        except Exception:
            continue
    return False


def download_from_huggingface() -> bool:
    """Download individual bottle images from Hugging Face dataset mirror."""
    import json

    print(f"[INFO] Fetching file catalog from Hugging Face ({HF_REPO})...")
    req = urllib.request.Request(HF_API_URL, headers={"User-Agent": "DefectGuard/0.1"})
    with urllib.request.urlopen(req, timeout=20) as res:
        metadata = json.loads(res.read().decode("utf-8"))

    siblings = metadata.get("siblings", [])
    bottle_files = [s["rfilename"] for s in siblings if s["rfilename"].startswith("bottle/")]

    if not bottle_files:
        print("[ERROR] No bottle files found in repository metadata.", file=sys.stderr)
        return False

    print(f"[INFO] Discovered {len(bottle_files)} files for 'bottle' dataset.")

    session = requests.Session() if requests else None
    success_count = 0
    total_files = len(bottle_files)

    def _task(rel_path: str):
        # target: rel_path is e.g. "bottle/train/good/000.png"
        target_path = DATA_DIR / rel_path
        # Skip if file already exists with non-zero size
        if target_path.exists() and target_path.stat().st_size > 0:
            return True
        url = f"{HF_RAW_BASE}/{rel_path}"
        return download_file(url, target_path, session=session)

    print("[INFO] Downloading dataset files concurrently...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_task, p): p for p in bottle_files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if fut.result():
                success_count += 1
            if done % 50 == 0 or done == total_files:
                print(f"[INFO] Progress: {done}/{total_files} files processed ({success_count} downloaded/verified).")

    return success_count == total_files


def verify_dataset() -> bool:
    """Verify presence and validity of train and test images."""
    print("\n--- Verifying Dataset Extraction ---")
    train_good = BOTTLE_DIR / "train" / "good"
    test_dir = BOTTLE_DIR / "test"

    if not train_good.exists() or not test_dir.exists():
        print(f"[ERROR] Required folders missing. Train: {train_good.exists()}, Test: {test_dir.exists()}")
        return False

    train_files = list(train_good.glob("*.png"))
    print(f"[OK] Training 'good' samples: {len(train_files)} images found (expected 209).")

    test_subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
    total_test = 0
    for subdir in sorted(test_subdirs):
        imgs = list(subdir.glob("*.png"))
        total_test += len(imgs)
        print(f"     - test/{subdir.name}: {len(imgs)} images")
    print(f"[OK] Total test samples: {total_test} images found across {len(test_subdirs)} categories (expected 83).")

    # Optional image verification
    sample_imgs = train_files[:3] + list(test_dir.rglob("*.png"))[:3]
    if Image:
        for s in sample_imgs:
            try:
                with Image.open(s) as img:
                    img.verify()
            except Exception as err:
                print(f"[ERROR] Corrupt image detected at {s}: {err}")
                return False
        print("[OK] Image integrity check passed on sample images.")

    if len(train_files) >= 200 and total_test >= 80:
        print("\n[SUCCESS] MVTec AD 'bottle' dataset successfully set up and verified!\n")
        return True
    else:
        print(f"[ERROR] Incomplete dataset: train={len(train_files)}, test={total_test}")
        return False


def main():
    BOTTLE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("DefectGuard - MVTec 'bottle' Dataset Setup")
    print("=" * 60)

    # 1. Try direct archive if available
    archive_success = try_download_archive()

    # 2. If not archived or missing, download from reliable Hugging Face repository
    if not archive_success:
        print("[INFO] Using reliable Hugging Face mirror for direct file sync...")
        download_from_huggingface()

    # 3. Verify
    if not verify_dataset():
        sys.exit(1)


if __name__ == "__main__":
    main()
