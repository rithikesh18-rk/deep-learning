"""
Dataset Preparation Pipeline for 20 Food Classes.

Features:
- Initializes folder structures for raw, train, and validation datasets.
- Validates image file integrity using PIL (filters corrupt/non-image files).
- Splits raw dataset images into train (80%) and validation (20%) sets.
- Generates detailed status reports on dataset readiness.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from tqdm import tqdm

# Add parent directory to path when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config

# Valid image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_valid_image(file_path: Path) -> bool:
    """
    Validates whether a given file path is a readable, uncorrupted image.

    Args:
        file_path (Path): Path to the image file.

    Returns:
        bool: True if valid image, False otherwise.
    """
    if not file_path.is_file() or file_path.name.startswith("."):
        return False

    if file_path.suffix.lower() not in VALID_EXTENSIONS:
        return False

    if file_path.stat().st_size == 0:
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image header and stream integrity
        return True
    except Exception:
        return False


def initialize_directories(classes: List[str] = config.FOOD_CLASSES) -> None:
    """
    Creates subdirectories for each food class in raw, train, and validation folders.
    Adds .gitkeep files to empty folders to keep Git directory tracking intact.
    """
    print("Initializing dataset folder structure...")
    for root_dir in [config.RAW_DIR, config.TRAIN_DIR, config.VAL_DIR]:
        root_dir.mkdir(parents=True, exist_ok=True)
        for food_class in classes:
            class_dir = root_dir / food_class
            class_dir.mkdir(parents=True, exist_ok=True)
            gitkeep = class_dir / ".gitkeep"
            if not gitkeep.exists() and len(list(class_dir.glob("*"))) == 0:
                gitkeep.touch()
    print("Folder structure initialization complete.")


def scan_and_validate_class(class_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Scans a class folder and separates valid images from invalid/corrupted files.

    Returns:
        Tuple[List[Path], List[Path]]: (valid_image_paths, invalid_file_paths)
    """
    if not class_dir.exists():
        return [], []

    valid_images = []
    invalid_files = []

    for file_path in class_dir.iterdir():
        if file_path.name == ".gitkeep":
            continue

        if is_valid_image(file_path):
            valid_images.append(file_path)
        else:
            invalid_files.append(file_path)

    return valid_images, invalid_files


def split_dataset(
    val_ratio: float = 0.2,
    seed: int = 42,
    move_files: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Splits images from dataset/raw/<class>/ into dataset/train/<class>/ and dataset/validation/<class>/.

    Args:
        val_ratio (float): Ratio of images to assign to validation (default 0.2 for 80/20 split).
        seed (int): Random seed for reproducible splitting.
        move_files (bool): If True, move files instead of copying.

    Returns:
        Dict[str, Dict[str, int]]: Split count statistics per class.
    """
    random.seed(seed)
    initialize_directories()

    stats = {}
    print(f"\nSplitting raw dataset into Train ({int((1-val_ratio)*100)}%) and Validation ({int(val_ratio*100)}%)...")

    for food_class in config.FOOD_CLASSES:
        raw_class_dir = config.RAW_DIR / food_class
        train_class_dir = config.TRAIN_DIR / food_class
        val_class_dir = config.VAL_DIR / food_class

        valid_images, invalid_files = scan_and_validate_class(raw_class_dir)

        if invalid_files:
            print(f"  [WARNING] {food_class}: Found {len(invalid_files)} invalid or corrupt files in raw folder (ignored).")

        if not valid_images:
            stats[food_class] = {"raw": 0, "train": 0, "val": 0}
            continue

        # Shuffle deterministically
        random.shuffle(valid_images)

        num_val = max(1, int(len(valid_images) * val_ratio)) if len(valid_images) > 1 else 0
        val_images = valid_images[:num_val]
        train_images = valid_images[num_val:]

        # Copy/move into target folders
        for img_path in train_images:
            target = train_class_dir / img_path.name
            if move_files:
                shutil.move(str(img_path), str(target))
            else:
                shutil.copy2(str(img_path), str(target))

        for img_path in val_images:
            target = val_class_dir / img_path.name
            if move_files:
                shutil.move(str(img_path), str(target))
            else:
                shutil.copy2(str(img_path), str(target))

        stats[food_class] = {
            "raw": len(valid_images),
            "train": len(train_images),
            "val": len(val_images)
        }

    print("Dataset splitting operation finished.")
    return stats


def check_dataset_status(classes: List[str] = config.FOOD_CLASSES) -> Dict[str, Dict[str, int]]:
    """
    Scans dataset directories and returns comprehensive statistics for raw, train, and val sets.
    """
    initialize_directories(classes)

    status_data = {}
    total_raw = 0
    total_train = 0
    total_val = 0

    print("\n" + "=" * 70)
    print(f"{'AI FOOD DATASET STATUS REPORT (20 CLASSES)':^70}")
    print("=" * 70)
    print(f"{'Class Name':<18} | {'Raw Valid':<10} | {'Train':<10} | {'Val':<10} | {'Status':<12}")
    print("-" * 70)

    for food_class in classes:
        raw_valid, raw_invalid = scan_and_validate_class(config.RAW_DIR / food_class)
        train_valid, _ = scan_and_validate_class(config.TRAIN_DIR / food_class)
        val_valid, _ = scan_and_validate_class(config.VAL_DIR / food_class)

        n_raw = len(raw_valid)
        n_train = len(train_valid)
        n_val = len(val_valid)

        total_raw += n_raw
        total_train += n_train
        total_val += n_val

        # Readiness status evaluation
        if n_train > 0 and n_val > 0:
            readiness = "READY"
        elif n_raw > 0:
            readiness = "NEEDS SPLIT"
        else:
            readiness = "EMPTY"

        status_data[food_class] = {
            "raw": n_raw,
            "train": n_train,
            "val": n_val,
            "status": readiness
        }

        print(f"{food_class:<18} | {n_raw:<10} | {n_train:<10} | {n_val:<10} | {readiness:<12}")

    print("-" * 70)
    print(f"{'TOTALS':<18} | {total_raw:<10} | {total_train:<10} | {total_val:<10} |")
    print("=" * 70 + "\n")

    # Detailed recommendations
    missing_classes = [c for c, d in status_data.items() if d["train"] == 0 and d["val"] == 0 and d["raw"] == 0]
    unsplit_classes = [c for c, d in status_data.items() if d["raw"] > 0 and d["train"] == 0]

    print("[SUMMARY & ACTIONABLE ITEMS]")
    if total_train > 0 and total_val > 0:
        print(f"  [OK] Training set ready with {total_train} images across active classes.")
        print(f"  [OK] Validation set ready with {total_val} images across active classes.")
    else:
        print("  [X] No images available in `dataset/train/` or `dataset/validation/`.")

    if unsplit_classes:
        print(f"\n  [!] Found raw images in {len(unsplit_classes)} classes that are not yet split.")
        print("      Run `py -3 src/prepare_dataset.py --split` to partition them 80/20 into train/validation.")

    if missing_classes:
        print(f"\n  [!] Missing images for {len(missing_classes)} out of 20 target food classes:")
        for idx, cls in enumerate(missing_classes, 1):
            print(f"      {idx:2d}. {cls}")
        print("\n--> ACTION REQUIRED: Please manually place image files (.jpg, .png, .webp) into `dataset/raw/<class_name>/`.")

    return status_data


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation and Splitting Script")
    parser.add_argument("--init-folders", action="store_true", help="Initialize class directories")
    parser.add_argument("--split", action="store_true", help="Split dataset/raw images into train and validation")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2 for 80/20 split)")
    parser.add_argument("--move", action="store_true", help="Move files during split instead of copying")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting reproducibility")

    args = parser.parse_args()

    if args.init_folders:
        initialize_directories()
        check_dataset_status()
    elif args.split:
        split_dataset(val_ratio=args.val_ratio, seed=args.seed, move_files=args.move)
        check_dataset_status()
    else:
        check_dataset_status()


if __name__ == "__main__":
    main()
