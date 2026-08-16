"""
Script to download official Food-101 dataset, inspect class names,
and extract exact semantic matches into dataset/raw/<class_name>/
"""

import sys
import os
import urllib.request
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.prepare_dataset import initialize_directories, check_dataset_status, is_valid_image

# Official Food-101 dataset URL
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
TEMP_DIR = config.DATASET_DIR / "temp_food101"
TAR_PATH = TEMP_DIR / "food-101.tar.gz"
EXTRACT_DIR = TEMP_DIR / "extracted"

# Official 101 classes in Food-101 dataset
ALL_FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]


def get_exact_semantic_matches() -> Dict[str, str]:
    """
    Finds exact semantic matches between target 20 classes and Food-101 classes.
    Adheres strictly to zero false-positive mappings (e.g. no chicken_wings -> chicken).

    Returns:
        Dict[str, str]: {target_class: food101_class}
    """
    matches = {}
    for target in config.FOOD_CLASSES:
        if target in ALL_FOOD101_CLASSES:
            matches[target] = target
        elif target == "burger" and "hamburger" in ALL_FOOD101_CLASSES:
            matches[target] = "hamburger"
    return matches


class DownloadProgressBar(tqdm):
    """Progress bar hook for urllib download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    """Downloads a file with progress reporting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset archive from:\n  {url}")
    print(f"Saving to:\n  {output_path}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Food-101 Download") as t:
        urllib.request.urlretrieve(url, filename=str(output_path), reporthook=t.update_to)
    print("Download complete!")


def process_food101_dataset():
    """
    Downloads Food-101 dataset archive if missing, selectively extracts exact matching
    classes into dataset/raw/<class_name>/, cleans up temporary files, and prints report.
    """
    print("=" * 70)
    print(f"{'FOOD-101 DATASET EXTRACTION & POPULATION':^70}")
    print("=" * 70)

    # 1. Analyze semantic matches
    matches = get_exact_semantic_matches()
    food101_to_target = {v: k for k, v in matches.items()}
    missing_classes = [c for c in config.FOOD_CLASSES if c not in matches]

    print(f"\nTarget Classes Count          : {len(config.FOOD_CLASSES)}")
    print(f"Total Food-101 Classes        : {len(ALL_FOOD101_CLASSES)}")
    print(f"Exact Semantic Matches Found  : {len(matches)} -> {list(matches.keys())}")
    print(f"Missing Target Classes (16)   : {missing_classes}")

    # Initialize raw, train, val directory structures
    initialize_directories()

    # 2. Check if archive or raw images already exist
    copied_counts = {target_cls: 0 for target_cls in matches.keys()}

    # Check if target images are already present in raw/
    already_populated = True
    for target_cls in matches.keys():
        valid_imgs, _ = scan_and_validate_class_count(config.RAW_DIR / target_cls)
        copied_counts[target_cls] = valid_imgs
        if valid_imgs == 0:
            already_populated = False

    if already_populated:
        print("\nAll matching Food-101 classes are already populated in dataset/raw/!")
        print_summary_report(ALL_FOOD101_CLASSES, matches, copied_counts, missing_classes)
        return

    # 3. Download tar.gz if not present
    if not TAR_PATH.exists():
        download_file(FOOD101_URL, TAR_PATH)
    else:
        print(f"\nFound existing Food-101 archive at: {TAR_PATH}")

    # 4. Perform selective extraction directly into dataset/raw/<class_name>/
    print("\nSelectively extracting matching food categories from archive...")
    extracted_counts = {target_cls: 0 for target_cls in matches.keys()}

    with tarfile.open(TAR_PATH, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="Extracting matching classes"):
            if not member.isfile() or not member.name.startswith("food-101/images/"):
                continue

            parts = member.name.split("/")
            if len(parts) == 4:
                food101_cls = parts[2]
                img_filename = parts[3]

                if food101_cls in food101_to_target:
                    target_cls = food101_to_target[food101_cls]
                    target_dir = config.RAW_DIR / target_cls
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_file = target_dir / img_filename

                    if not target_file.exists():
                        fileobj = tar.extractfile(member)
                        if fileobj:
                            with open(target_file, "wb") as f:
                                f.write(fileobj.read())
                            extracted_counts[target_cls] += 1

    # Cleanup temporary tarball to free disk space
    if TAR_PATH.exists():
        print(f"\nCleaning up temporary archive file ({TAR_PATH})...")
        try:
            os.remove(TAR_PATH)
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print("Cleanup complete!")
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    # Verify extracted counts
    for target_cls in matches.keys():
        valid_imgs, _ = scan_and_validate_class_count(config.RAW_DIR / target_cls)
        copied_counts[target_cls] = valid_imgs

    print_summary_report(ALL_FOOD101_CLASSES, matches, copied_counts, missing_classes)


def scan_and_validate_class_count(class_dir: Path) -> Tuple[int, int]:
    """Helper to return (valid_count, total_count)."""
    if not class_dir.exists():
        return 0, 0
    valid_count = 0
    total_count = 0
    for p in class_dir.iterdir():
        if p.name == ".gitkeep":
            continue
        total_count += 1
        if is_valid_image(p):
            valid_count += 1
    return valid_count, total_count


def print_summary_report(
    all_classes: List[str],
    matches: Dict[str, str],
    copied_counts: Dict[str, int],
    missing_classes: List[str]
):
    """Prints the required final report."""
    print("\n" + "=" * 70)
    print(f"{'FOOD-101 EXTRACTION SUMMARY REPORT':^70}")
    print("=" * 70)
    print(f"Total Food-101 Classes Inspected : {len(all_classes)}")
    print(f"Exact Matches Copied             : {len(matches)} classes")
    print(f"Classes Still Missing            : {len(missing_classes)} classes")
    print("-" * 70)

    print("\n[MATCHED & COPIED CLASSES]")
    for target_cls, food101_cls in matches.items():
        count = copied_counts.get(target_cls, 0)
        print(f"  - {target_cls:<15} (from Food-101 '{food101_cls}') : {count} valid images")

    print("\n[MISSING CLASSES NOT IN FOOD-101]")
    for idx, cls in enumerate(missing_classes, 1):
        print(f"  {idx:2d}. {cls:<15} (Requires manual image addition)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Food-101 Dataset Download & Inspection Tool")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect class matches without downloading")

    args = parser.parse_args()

    if args.inspect_only:
        matches = get_exact_semantic_matches()
        missing = [c for c in config.FOOD_CLASSES if c not in matches]
        copied_counts = {k: 0 for k in matches.keys()}
        print_summary_report(ALL_FOOD101_CLASSES, matches, copied_counts, missing)
    else:
        process_food101_dataset()
