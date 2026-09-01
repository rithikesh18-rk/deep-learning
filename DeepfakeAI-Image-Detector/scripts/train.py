"""Dual-Stream Forensic Neural Network Training, Checkpointing & Resumption Pipeline.

Features:
- Periodic checkpointing after EVERY epoch to 'backend/models/training_checkpoint_latest.pth'
- Best model tracking to 'backend/models/deepfake_detector_best.pth'
- Full state preservation: model weights, optimizer, scheduler, epoch, best metric, history
- Seamless resumption via '--resume' flag
- Real-time flushed logging
- Comprehensive validation suite (Loss, Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add backend directory to sys.path
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

# Maximize multi-core CPU utilization
torch.set_num_threads(min(16, os.cpu_count() or 4))

# Configuration Defaults
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 5
LR_CLASSIFIER = 5e-4
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4

SAVE_DIR = Path("backend/models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT_PATH = SAVE_DIR / "training_checkpoint_latest.pth"
BEST_CHECKPOINT_PATH = SAVE_DIR / "deepfake_detector_best.pth"

# 2D Hann Window for Frequency Stream Preprocessing
HANN_2D = np.outer(np.hanning(224), np.hanning(224)).astype(np.float32)

# Transforms
TRAIN_RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VAL_RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def compute_fft_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Extracts normalized 1-channel windowed 2D-FFT log-magnitude spectrum tensor."""
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    gray_224 = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    norm_gray = gray_224 - np.mean(gray_224)
    windowed = norm_gray * HANN_2D

    f_shift = np.fft.fftshift(np.fft.fft2(windowed))
    mag = 20.0 * np.log(np.abs(f_shift) + 1e-8)

    # Min-Max Normalization to [0, 1]
    m_min, m_max = np.min(mag), np.max(mag)
    if m_max - m_min > 1e-8:
        norm_mag = (mag - m_min) / (m_max - m_min)
    else:
        norm_mag = np.zeros_like(mag, dtype=np.float32)

    return torch.from_numpy(norm_mag).float().unsqueeze(0)  # (1, 224, 224)


class ForensicDualStreamDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None, max_samples=None):
        self.samples = []
        self.transform = transform

        real_dir = root_dir / "authentic"
        fake_dir = root_dir / "ai_generated"

        real_samples = []
        fake_samples = []

        if real_dir.exists():
            for p in sorted(real_dir.glob("*.png")):
                real_samples.append((p, 0))
        if fake_dir.exists():
            for p in sorted(fake_dir.glob("*.png")):
                fake_samples.append((p, 1))

        if max_samples is not None and max_samples > 0:
            half = max_samples // 2
            real_samples = real_samples[:half]
            fake_samples = fake_samples[:half]

        self.samples = real_samples + fake_samples
        print(f"Dataset '{root_dir}': Loaded {len(self.samples)} images ({len(real_samples)} real, {len(fake_samples)} ai).", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            pil_img = img.convert("RGB")
            freq_tensor = compute_fft_tensor(pil_img)
            rgb_tensor = self.transform(pil_img) if self.transform else VAL_RGB_TRANSFORM(pil_img)

        return rgb_tensor, freq_tensor, label


def evaluate_model(model, dataloader, criterion):
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

            probs = torch.softmax(logits, dim=-1)[:, 1]  # Prob of AI-generated (class 1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples
    acc = accuracy_score(all_targets, all_preds)
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


def train_pipeline(args):
    print("=" * 80, flush=True)
    print("DUAL-STREAM FORENSIC DETECTOR TRAINING PIPELINE", flush=True)
    print(f"Device: {DEVICE} (Threads: {torch.get_num_threads()}) | Batch Size: {args.batch_size} | Total Epochs: {args.epochs}", flush=True)
    print(f"Latest Checkpoint Path: {LATEST_CHECKPOINT_PATH}", flush=True)
    print(f"Best Model Path:        {BEST_CHECKPOINT_PATH}", flush=True)
    print("=" * 80, flush=True)

    # 1. Datasets & Loaders
    train_ds = ForensicDualStreamDataset(Path("dataset/train"), transform=TRAIN_RGB_TRANSFORM, max_samples=args.max_train_samples)
    val_ds = ForensicDualStreamDataset(Path("dataset/val"), transform=VAL_RGB_TRANSFORM, max_samples=args.max_val_samples)
    test_ds = ForensicDualStreamDataset(Path("dataset/test"), transform=VAL_RGB_TRANSFORM, max_samples=args.max_test_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model Initialization
    print("\nInitializing DualStreamForensicNet with ConvNeXt-Tiny backbone...", flush=True)
    model = DualStreamForensicNet(pretrained=True, num_classes=2).to(DEVICE)

    # Freeze early stem & stages 0, 1 of ConvNeXt for fast transfer learning & strong generalization
    for param in model.spatial_backbone.stem.parameters():
        param.requires_grad = False
    for param in model.spatial_backbone.stages[0].parameters():
        param.requires_grad = False
    for param in model.spatial_backbone.stages[1].parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR_CLASSIFIER, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_f1 = 0.0
    history = []

    # 3. Resume from Checkpoint if requested or if latest checkpoint exists
    checkpoint_to_resume = None
    if args.resume:
        if isinstance(args.resume, str) and Path(args.resume).is_file():
            checkpoint_to_resume = Path(args.resume)
        elif LATEST_CHECKPOINT_PATH.is_file():
            checkpoint_to_resume = LATEST_CHECKPOINT_PATH

    if checkpoint_to_resume and checkpoint_to_resume.is_file():
        print(f"\n--> Resuming training from checkpoint: {checkpoint_to_resume}", flush=True)
        ckpt = torch.load(checkpoint_to_resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        history = ckpt.get("history", [])
        print(f"Resumed successfully! Next Epoch to run: {start_epoch} (Best Val F1 so far: {best_val_f1:.4f})", flush=True)

    if start_epoch > args.epochs:
        print(f"Training already completed up to epoch {args.epochs}. Running final test evaluation...", flush=True)
    else:
        # 4. Training Loop
        start_time = time.time()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            epoch_start = time.time()
            for batch_idx, (rgb_tensors, freq_tensors, labels) in enumerate(train_loader):
                rgb_tensors = rgb_tensors.to(DEVICE)
                freq_tensors = freq_tensors.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                logits = model(rgb_tensors, freq_tensors)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * rgb_tensors.size(0)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(train_loader):
                    batch_acc = correct / total * 100.0
                    print(f"Epoch [{epoch}/{args.epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Train Loss: {loss.item():.4f} | Running Acc: {batch_acc:.2f}%", flush=True)

            scheduler.step()
            epoch_train_loss = train_loss / total
            epoch_train_acc = correct / total

            # Validation Phase
            val_metrics = evaluate_model(model, val_loader, criterion)
            epoch_time = time.time() - epoch_start

            print(f"\n--- [EPOCH {epoch}/{args.epochs} RESULTS ({epoch_time:.1f}s)] ---", flush=True)
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
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"]
            })

            # Checkpoint 1: Always Save Latest Resumable Checkpoint After EVERY Epoch
            print(f"--> Saving latest resumable checkpoint to: {LATEST_CHECKPOINT_PATH}...", flush=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1": max(best_val_f1, val_metrics["f1"]),
                "val_metrics": val_metrics,
                "history": history,
                "model_architecture": "DualStreamForensicNet (ConvNeXt-Tiny + 2D-FFT)",
                "classes": ["Authentic", "AI-Generated"]
            }, LATEST_CHECKPOINT_PATH)

            # Checkpoint 2: Save Best Model Separately When F1 Improves
            if val_metrics["f1"] > best_val_f1 or not BEST_CHECKPOINT_PATH.is_file():
                best_val_f1 = max(best_val_f1, val_metrics["f1"])
                print(f"--> Validation F1 updated ({best_val_f1:.4f})! Saving best model to {BEST_CHECKPOINT_PATH}...", flush=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "state_dict": model.state_dict(),  # Alias for backend loader compatibility
                    "best_val_f1": best_val_f1,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "model_architecture": "DualStreamForensicNet (ConvNeXt-Tiny + 2D-FFT)",
                    "classes": ["Authentic", "AI-Generated"]
                }, BEST_CHECKPOINT_PATH)

        total_training_time = time.time() - start_time
        print(f"\nTraining session completed in {total_training_time/60:.2f} minutes.", flush=True)

    # 5. Final Evaluation on Held-out Test Set
    print("\n" + "=" * 80, flush=True)
    print("FINAL EVALUATION ON UNSEEN HELD-OUT TEST SET", flush=True)
    print("=" * 80, flush=True)

    eval_checkpoint = BEST_CHECKPOINT_PATH if BEST_CHECKPOINT_PATH.is_file() else LATEST_CHECKPOINT_PATH
    if eval_checkpoint.is_file():
        print(f"Loading checkpoint for evaluation: {eval_checkpoint}", flush=True)
        ckpt = torch.load(eval_checkpoint, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        model.load_state_dict(state_dict)

        test_metrics = evaluate_model(model, test_loader, criterion)
        print(f"Test Loss:             {test_metrics['loss']:.4f}", flush=True)
        print(f"Test Accuracy:         {test_metrics['accuracy']*100:.2f}%", flush=True)
        print(f"Test Precision:        {test_metrics['precision']*100:.2f}%", flush=True)
        print(f"Test Recall:           {test_metrics['recall']*100:.2f}%", flush=True)
        print(f"Test F1-Score:         {test_metrics['f1']*100:.2f}%", flush=True)
        print(f"Test ROC-AUC:          {test_metrics['roc_auc']:.4f}", flush=True)
        print("\nTest Confusion Matrix [TN, FP / FN, TP]:", flush=True)
        print(test_metrics["confusion_matrix"], flush=True)
        print("=" * 80, flush=True)
    else:
        print(f"No checkpoint file found at {eval_checkpoint} to evaluate.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dual-Stream Forensic Detector")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Total number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for dataloaders")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional max training samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional max validation samples")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional max test samples")
    parser.add_argument("--log-interval", type=int, default=10, help="Print progress every N batches")
    args = parser.parse_args()

    train_pipeline(args)
