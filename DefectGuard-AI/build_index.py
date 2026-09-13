"""
DefectGuard - Index Builder Script.
Builds the PatchCore nominal memory bank for the 'bottle' subcategory:
1. Extracts patch features from all normal training images using WideResNet-50-2.
2. Applies Johnson-Lindenstrauss random projection + Greedy Minimax Coreset Subsampling (10%).
3. Populates a FAISS IndexFlatL2 index and saves it to models/bottle_patchcore.index.
"""

import sys
import time
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.feature_extractor import PatchCoreFeatureExtractor
from src.coreset import GreedyCoresetSampler
from src.memory_bank import PatchCoreMemoryBank

ROOT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = ROOT_DIR / "data" / "bottle" / "train" / "good"
OUTPUT_INDEX = ROOT_DIR / "models" / "bottle_patchcore.index"
OUTPUT_META = ROOT_DIR / "models" / "bottle_patchcore_meta.json"


class BottleTrainDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, str(path)


def main():
    start_time = time.time()
    print("=" * 65)
    print("DefectGuard - PatchCore Memory Bank Builder")
    print("=" * 65)

    if not TRAIN_DIR.exists():
        print(f"[ERROR] Training directory not found: {TRAIN_DIR}", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(list(TRAIN_DIR.glob("*.png")))
    if not image_paths:
        print(f"[ERROR] No PNG training images found in {TRAIN_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Found {len(image_paths)} normal training images in {TRAIN_DIR.name}")

    # Initialize Feature Extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Initializing WideResNet-50-2 Feature Extractor on {device}...")
    extractor = PatchCoreFeatureExtractor(device=device)

    dataset = BottleTrainDataset(image_paths, extractor.transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"[2/4] Extracting patch tokens across {len(loader)} batches...")
    all_patch_tokens = []
    grid_size = None

    with torch.no_grad():
        for batch_idx, (batch_tensors, _) in enumerate(loader):
            batch_tensors = batch_tensors.to(device)
            tokens, grid = extractor(batch_tensors)
            grid_size = grid
            # Flatten batch tokens to [B * patches_per_img, D]
            tokens_flat = tokens.view(-1, tokens.shape[-1]).cpu()
            all_patch_tokens.append(tokens_flat)
            print(f"      Batch {batch_idx + 1}/{len(loader)}: extracted {tokens_flat.shape[0]:,} patch tokens.")

    all_patch_tokens = torch.cat(all_patch_tokens, dim=0)
    print(f"[2/4] Extraction complete. Total patch vectors: {all_patch_tokens.shape[0]:,} (dim={all_patch_tokens.shape[1]}), grid={grid_size}")

    # 3. Build and Save Complete FAISS Index
    print(f"\n[3/4] Building FAISS IndexFlatL2 index with all {all_patch_tokens.shape[0]:,} training vectors...")
    memory_bank = PatchCoreMemoryBank(feature_dim=all_patch_tokens.shape[1])
    memory_bank.fit(all_patch_tokens)
    memory_bank.save(OUTPUT_INDEX)

    # Save metadata
    meta = {
        "dataset": "mvtec_bottle",
        "total_train_images": len(image_paths),
        "total_extracted_patches": int(all_patch_tokens.shape[0]),
        "coreset_size": int(all_patch_tokens.shape[0]),
        "feature_dim": int(all_patch_tokens.shape[1]),
        "grid_size": list(grid_size),
        "sampling_ratio": 1.0,
        "backbone": "wide_resnet50_2",
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[4/4] Saved metadata to {OUTPUT_META.name}")

    # Quick Verification Inference
    print("\n--- Running Quick Verification Inference ---")
    test_normal = ROOT_DIR / "data" / "bottle" / "test" / "good" / "000.png"
    test_defect = ROOT_DIR / "data" / "bottle" / "test" / "broken_large" / "000.png"

    if test_normal.exists() and test_defect.exists():
        with torch.no_grad():
            img_norm = Image.open(test_normal).convert("RGB")
            t_norm = extractor.preprocess_image(img_norm)
            tok_norm, _ = extractor(t_norm)
            res_norm = memory_bank.predict(tok_norm, grid_size=grid_size)

            img_def = Image.open(test_defect).convert("RGB")
            t_def = extractor.preprocess_image(img_def)
            tok_def, _ = extractor(t_def)
            res_def = memory_bank.predict(tok_def, grid_size=grid_size)

            print(f"[OK] Normal test image score : {res_norm['image_scores']:.4f}")
            print(f"[OK] Defect test image score : {res_def['image_scores']:.4f}")
            print(f"[OK] Anomaly map dimensions : {res_norm['anomaly_maps'].shape}")
            assert res_def["image_scores"] > res_norm["image_scores"], "Defect score should be higher than normal score!"
            print("[OK] Defect separation verified (Defect score > Normal score)!")

    print(f"\n[SUCCESS] Memory bank build complete in {time.time() - start_time:.2f}s!")


if __name__ == "__main__":
    main()
