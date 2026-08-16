"""
Script to download ashishpadala/indian-food-images dataset from Kaggle,
extract and map 6 target classes into dataset/raw/<class_name>/,
and clean up temporary dataset folders.
"""

import sys
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict
from tqdm import tqdm

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.prepare_dataset import initialize_directories, is_valid_image

KAGGLE_DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/ashishpadala/indian-food-images"
TEMP_DIR = config.DATASET_DIR / "temp_kaggle"
ZIP_PATH = TEMP_DIR / "indian_food_images.zip"
EXTRACT_DIR = TEMP_DIR / "extracted"

# Exact folder mapping rules requested:
# Dosa -> dosa
# Idly -> idli
# Biryani -> biryani
# Roti -> chapati
# Puri -> poori
# Sambar -> sambar
CLASS_MAPPING = {
    "Dosa": "dosa",
    "Idly": "idli",
    "Biryani": "biryani",
    "Roti": "chapati",
    "Puri": "poori",
    "Sambar": "sambar"
}


class DownloadProgressBar(tqdm):
    """Progress bar hook for urllib download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_kaggle_zip(url: str, output_path: Path):
    """Downloads Kaggle dataset zip archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset (ashishpadala/indian-food-images)...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Kaggle Zip Download") as t:
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            meta = response.info()
            total_size = meta.get("Content-Length")
            if total_size:
                t.total = int(total_size)

            block_size = 8192
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                t.update(len(buffer))

    print("Download complete!")


def clean_obsolete_folders():
    """Removes unused class subfolders from dataset/raw, dataset/train, and dataset/validation."""
    valid_set = set(config.FOOD_CLASSES)
    for parent in [config.RAW_DIR, config.TRAIN_DIR, config.VAL_DIR]:
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if child.is_dir() and child.name not in valid_set:
                shutil.rmtree(child, ignore_errors=True)


def process_kaggle_dataset():
    """
    Downloads, extracts, maps, and populates the 6 target classes into dataset/raw/.
    """
    print("=" * 70)
    print(f"{'KAGGLE DATASET POPULATION (6 TARGET CLASSES)':^70}")
    print("=" * 70)

    # 1. Clean obsolete class folders and ensure directory structure
    clean_obsolete_folders()
    initialize_directories()

    # 2. Download zip if missing
    if not ZIP_PATH.exists():
        download_kaggle_zip(KAGGLE_DATASET_URL, ZIP_PATH)
    else:
        print(f"Found existing zip file at: {ZIP_PATH}")

    # 3. Perform selective extraction directly into dataset/raw/<class_name>/
    print("\nSelectively extracting target class folders from zip archive...")
    copied_stats = {target_class: 0 for target_class in CLASS_MAPPING.values()}

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting matching classes"):
            if member.is_dir():
                continue

            member_path = Path(member.filename)
            parts_lower = [p.lower() for p in member_path.parts]

            for source_folder_name, target_class in CLASS_MAPPING.items():
                if source_folder_name.lower() in parts_lower:
                    target_dir = config.RAW_DIR / target_class
                    target_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = target_dir / member_path.name

                    if not dest_file.exists() or dest_file.stat().st_size == 0:
                        with zip_ref.open(member) as src_file, open(dest_file, "wb") as out_file:
                            out_file.write(src_file.read())

                    if is_valid_image(dest_file):
                        copied_stats[target_class] += 1
                    else:
                        if dest_file.exists():
                            dest_file.unlink()

    for source_folder_name, target_class in CLASS_MAPPING.items():
        count = copied_stats[target_class]
        print(f"  [OK] {source_folder_name:<10} -> dataset/raw/{target_class:<8} ({count} valid images extracted)")

    # 4. Clean up temporary zip files
    print("\nCleaning up temporary files...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("Cleanup complete!")

    print("\n" + "=" * 70)
    print("Raw dataset population successful!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    process_kaggle_dataset()
