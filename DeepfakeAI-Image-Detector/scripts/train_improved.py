"""Improved Dual-Stream Forensic Detector Training Pipeline.

Features:
- Aspect-ratio preserving letterbox preprocessing (matching inference).
- Multi-directory dataset ingestion (baseline CIFAKE + optional high-resolution dataset).
- Differential learning rates with 2-stage transfer learning (head warmup -> fine-tuning).
- Forensic-preserving data augmentations (JPEG compression, subtle jitter, horizontal flip).
- Cosine Annealing learning rate scheduling with early stopping.
- Checkpoint isolation (preserves baseline deepfake_detector_best.pth).
"""

import os
import sys
import io
import time
import random
import argparse
from pathlib import Path

# Add project root and backend directory to sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from model import DualStreamForensicNet
from frequency_utils import letterbox_image, letterbox_gray, RGB_NORMALIZE

# Multi-threading optimization
torch.set_num_threads(min(16, os.cpu_count() or 4))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HANN_2D = np.outer(np.hanning(224), np.hanning(224)).astype(np.float32)


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_fft_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Extracts aspect-ratio letterboxed 2D-FFT log-magnitude spectrum tensor."""
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    gray_letterboxed = letterbox_gray(gray, target_size=(224, 224))
    norm_gray = gray_letterboxed - np.mean(gray_letterboxed)
    windowed = norm_gray * HANN_2D

    f_shift = np.fft.fftshift(np.fft.fft2(windowed))
    mag = 20.0 * np.log(np.abs(f_shift) + 1e-8)

    m_min, m_max = np.min(mag), np.max(mag)
    if m_max - m_min > 1e-8:
        norm_mag = (mag - m_min) / (m_max - m_min)
    else:
        norm_mag = np.zeros_like(mag, dtype=np.float32)

    return torch.from_numpy(norm_mag).float().unsqueeze(0)


class ForensicAugmentation:
    """Forensic-preserving spatial augmentation."""
    def __init__(self, is_train: bool = True):
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        if self.is_train:
            # Horizontal Flip (50% probability)
            if random.random() > 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Mild color jitter (40% probability)
            if random.random() > 0.6:
                pil_img = self.color_jitter(pil_img)

            # Simulated JPEG compression artifact (30% probability)
            if random.random() > 0.7:
                buffer = io.BytesIO()
                quality = random.randint(75, 95)
                pil_img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                pil_img = Image.open(buffer).convert("RGB")

        # Aspect-ratio preserving letterbox to 224x224
        letterboxed = letterbox_image(pil_img, target_size=(224, 224))
        tensor = self.to_tensor(letterboxed)
        return RGB_NORMALIZE(tensor)


class MultiSourceForensicDataset(Dataset):
    """Loads binary dataset samples from multiple directory roots supporting all standard formats."""
    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, root_dirs: list[Path], is_train: bool = False, max_samples: int = None):
        self.samples = []
        self.transform = ForensicAugmentation(is_train=is_train)

        real_samples = []
        fake_samples = []

        for root in root_dirs:
            if not root.exists():
                continue
            real_dir = root / "authentic"
            fake_dir = root / "ai_generated"

            if real_dir.exists():
                for ext in self.SUPPORTED_EXTS:
                    for p in real_dir.glob(f"*{ext}"):
                        real_samples.append((p, 0))
            if fake_dir.exists():
                for ext in self.SUPPORTED_EXTS:
                    for p in fake_dir.glob(f"*{ext}"):
                        fake_samples.append((p, 1))

        # Sort for determinism
        real_samples = sorted(real_samples, key=lambda x: str(x[0]))
        fake_samples = sorted(fake_samples, key=lambda x: str(x[0]))

        if max_samples is not None and max_samples > 0:
            half = max_samples // 2
            real_samples = real_samples[:half]
            fake_samples = fake_samples[:half]

        self.samples = real_samples + fake_samples
        print(f"Dataset {[str(r) for r in root_dirs]}: Loaded {len(self.samples)} images ({len(real_samples)} Authentic, {len(fake_samples)} AI).", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            pil_img = img.convert("RGB")
            freq_tensor = compute_fft_tensor(pil_img)
            rgb_tensor = self.transform(pil_img)

        return rgb_tensor, freq_tensor, label


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for rgb_tensors, freq_tensors, labels in dataloader:
            rgb_tensors = rgb_tensors.to(DEVICE)
            freq_tensors = freq_tensors.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(rgb_tensors, freq_tensors)
            loss = criterion(logits, labels)
            total_loss += loss.item() * rgb_tensors.size(0)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds) if n_samples > 0 else 0.0
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        roc_auc = 0.5
    cm = confusion_matrix(all_targets, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "targets": all_targets,
        "preds": all_preds,
        "probs": all_probs
    }


def train_improved_model(args):
    set_seed(args.seed)

    print("=" * 80, flush=True)
    print("IMPROVED DUAL-STREAM FORENSIC DETECTOR TRAINING PIPELINE", flush=True)
    print(f"Device: {DEVICE} (Threads: {torch.get_num_threads()}) | Seed: {args.seed}", flush=True)
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Unfreeze Epoch: {args.unfreeze_epoch}", flush=True)
    print(f"Improved Checkpoint Target: {args.save_path}", flush=True)
    print("=" * 80, flush=True)

    # Ingestion Roots
    train_roots = [Path("dataset/train")]
    if args.extra_dataset_dir and Path(args.extra_dataset_dir).exists():
        train_roots.append(Path(args.extra_dataset_dir))

    val_roots = [Path("dataset/val")]
    test_roots = [Path("dataset/test")]

    train_ds = MultiSourceForensicDataset(train_roots, is_train=True, max_samples=args.max_train_samples)
    val_ds = MultiSourceForensicDataset(val_roots, is_train=False, max_samples=args.max_val_samples)
    test_ds = MultiSourceForensicDataset(test_roots, is_train=False, max_samples=args.max_test_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize Model
    model = DualStreamForensicNet(pretrained=True, num_classes=2).to(DEVICE)

    # Differential Param Groups
    head_params = list(model.classifier.parameters()) + list(model.frequency_backbone.parameters())
    backbone_late_params = list(model.spatial_backbone.stages[2].parameters()) + list(model.spatial_backbone.stages[3].parameters())
    backbone_early_params = list(model.spatial_backbone.stem.parameters()) + list(model.spatial_backbone.stages[0].parameters()) + list(model.spatial_backbone.stages[1].parameters())

    # Initial Freeze: Freeze early and late backbone during head warmup
    for p in backbone_early_params:
        p.requires_grad = False
    for p in backbone_late_params:
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr_head, "weight_decay": 1e-4},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_val_roc_auc = 0.0
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Stage 2 Transition: Unfreeze late backbone stages after warmup
        if epoch == args.unfreeze_epoch:
            print(f"\n[Stage 2] Unfreezing ConvNeXt stages 2 & 3 for controlled fine-tuning (lr={args.lr_backbone})...", flush=True)
            for p in backbone_late_params:
                p.requires_grad = True

            optimizer = torch.optim.AdamW([
                {"params": head_params, "lr": args.lr_head * 0.5, "weight_decay": 1e-4},
                {"params": backbone_late_params, "lr": args.lr_backbone, "weight_decay": 1e-4},
            ])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1, eta_min=1e-6)

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.time()
        for batch_idx, (rgb, freq, labels) in enumerate(train_loader):
            rgb = rgb.to(DEVICE)
            freq = freq.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(rgb, freq)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * rgb.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = correct / total * 100.0
                print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Batch [{batch_idx+1:03d}/{len(train_loader):03d}] | Loss: {loss.item():.4f} | Running Acc: {batch_acc:.2f}%", flush=True)

        scheduler.step()
        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # Check for NaN / Inf loss
        if np.isnan(epoch_train_loss) or np.isinf(epoch_train_loss):
            print(f"\n[CRITICAL ERROR] Detected NaN or Inf loss during training at Epoch {epoch}! Stopping safely.", flush=True)
            break

        # Validation Evaluation
        val_metrics = evaluate(model, val_loader, criterion)
        epoch_time = time.time() - epoch_start
        current_lrs = [f"{param_group['lr']:.2e}" for param_group in optimizer.param_groups]

        print(f"\n--- [EPOCH {epoch:02d}/{args.epochs:02d} EVALUATION ({epoch_time:.1f}s)] ---", flush=True)
        print(f"Learning Rate(s): {', '.join(current_lrs)}", flush=True)
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%", flush=True)
        print(f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']*100:.2f}%", flush=True)
        print(f"Val Prec:   {val_metrics['precision']:.4f} | Val Rec:   {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}", flush=True)
        print(f"Val Confusion Matrix:\n{val_metrics['confusion_matrix']}", flush=True)
        print("-" * 60, flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"]
        })

        # Save Improved Best Model Checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_roc_auc = val_metrics["roc_auc"]
            patience_counter = 0

            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"--> [NEW BEST] Saving improved checkpoint (Val F1: {best_val_f1:.4f}, ROC-AUC: {best_val_roc_auc:.4f}) to {save_path}...", flush=True)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "state_dict": model.state_dict(),
                "best_val_f1": best_val_f1,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_roc_auc": val_metrics["roc_auc"],
                "history": history,
                "model_architecture": "DualStreamForensicNet (ConvNeXt-Tiny + 2D-FFT)",
                "classes": ["Authentic", "AI-Generated"],
                "letterbox_preprocessing": True
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] No improvement in validation F1 for {args.patience} consecutive epochs. Stopping.", flush=True)
                break

    total_time = time.time() - start_time
    print(f"\nTraining pipeline finished in {total_time/60:.2f} minutes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Improved Dual-Stream Forensic Detector")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr-head", type=float, default=5e-4, help="Learning rate for head/FFT (default: 5e-4)")
    parser.add_argument("--lr-backbone", type=float, default=5e-5, help="Learning rate for backbone (default: 5e-5)")
    parser.add_argument("--unfreeze-epoch", type=int, default=3, help="Epoch to unfreeze ConvNeXt late stages (default: 3)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--extra-dataset-dir", type=str, default=None, help="Optional extra dataset root")
    parser.add_argument("--save-path", type=str, default="backend/models/deepfake_detector_improved.pth", help="Path to save improved checkpoint")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max train samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max val samples")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--log-interval", type=int, default=15, help="Log interval")
    args = parser.parse_args()

    train_improved_model(args)
