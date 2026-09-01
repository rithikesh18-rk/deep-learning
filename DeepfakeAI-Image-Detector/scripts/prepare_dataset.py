"""CIFAKE Dataset Ingestion & Partitioning Script.

Extracts a balanced 10,000-image subset (5,000 Authentic, 5,000 AI-Generated)
from dragonintelligence/CIFAKE-image-dataset and partitions with zero data leakage:
  - train (70%): 7,000 images (3,500 Real, 3,500 AI)
  - val   (15%): 1,500 images (750 Real, 750 AI)
  - test  (15%): 1,500 images (750 Real, 750 AI)
"""

import os
import io
import hashlib
import random
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

random.seed(42)

DATASET_ROOT = Path("dataset")

def prepare_cifake_dataset(target_per_class: int = 5000):
    print("Downloading CIFAKE dataset parquet file from Hugging Face...")
    parquet_path = hf_hub_download(
        repo_id="dragonintelligence/CIFAKE-image-dataset",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset"
    )
    print(f"Downloaded parquet to: {parquet_path}")

    table = pq.read_table(parquet_path)
    print(f"Total available samples in parquet: {len(table)}")

    # Separate by class
    real_images = []
    fake_images = []

    seen_hashes = set()

    for idx in range(len(table)):
        label = table["label"][idx].as_py()
        img_bytes = table["image"][idx].as_py()["bytes"]
        
        # Deduplication check
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)

        if label == 0 and len(real_images) < target_per_class:
            real_images.append((img_bytes, 0))
        elif label == 1 and len(fake_images) < target_per_class:
            fake_images.append((img_bytes, 1))

        if len(real_images) >= target_per_class and len(fake_images) >= target_per_class:
            break

    print(f"Extracted {len(real_images)} Authentic (Real) and {len(fake_images)} AI-Generated (Fake) images.")

    # Shuffle each class
    random.shuffle(real_images)
    random.shuffle(fake_images)

    # 70% Train, 15% Val, 15% Test splits
    n_train = int(target_per_class * 0.70)  # 3500
    n_val = int(target_per_class * 0.15)    # 750
    n_test = target_per_class - n_train - n_val  # 750

    splits = {
        "train": (real_images[:n_train], fake_images[:n_train]),
        "val": (real_images[n_train:n_train+n_val], fake_images[n_train:n_train+n_val]),
        "test": (real_images[n_train+n_val:], fake_images[n_train+n_val:])
    }

    for split_name, (reals, fakes) in splits.items():
        real_dir = DATASET_ROOT / split_name / "authentic"
        fake_dir = DATASET_ROOT / split_name / "ai_generated"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        for i, (img_bytes, _) in enumerate(reals):
            img_path = real_dir / f"real_{i:05d}.png"
            with open(img_path, "wb") as f:
                f.write(img_bytes)

        for i, (img_bytes, _) in enumerate(fakes):
            img_path = fake_dir / f"fake_{i:05d}.png"
            with open(img_path, "wb") as f:
                f.write(img_bytes)

        print(f"Split '{split_name}': {len(reals)} authentic, {len(fakes)} ai_generated images saved.")

    print("\nDataset structure successfully prepared under ./dataset/")

if __name__ == "__main__":
    prepare_cifake_dataset(target_per_class=5000)
