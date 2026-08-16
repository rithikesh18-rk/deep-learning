"""
Training and Fine-Tuning script for AI Food Calorie & Nutrition Estimation model using PyTorch.

Supports:
- Pretrained transfer learning & controlled backbone fine-tuning.
- AdamW optimizer with weight decay.
- Early stopping (patience = 4).
- Learning rate scheduling and per-epoch LR logging.
- Per-class accuracy breakdown and confusion matrix evaluation.
"""

import argparse
import sys
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.dataset import get_data_loaders
from src.model import create_model


def check_class_balance(train_dataset: torch.utils.data.Dataset, class_names: list):
    """Prints the training dataset class counts to check for class imbalance."""
    counts = Counter(train_dataset.targets)
    print("\n[DATASET CLASS BALANCE CHECK]")
    print("-" * 50)
    for idx, name in enumerate(class_names):
        print(f"  Class {idx}: {name:<15} | Count: {counts[idx]} images")
    print("-" * 50)


def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """Executes one training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device):
    """Executes one validation epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate_final_model(model: nn.Module, loader: torch.utils.data.DataLoader, class_names: list, device: torch.device):
    """
    Evaluates the model on the validation dataset and computes:
    - Overall validation accuracy
    - Per-class accuracy
    - 6x6 Confusion matrix
    """
    model.eval()
    num_classes = len(class_names)

    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for label, pred in zip(labels.view(-1), preds.view(-1)):
                l = label.item()
                p = pred.item()
                if l == p:
                    class_correct[l] += 1
                class_total[l] += 1
                confusion_matrix[l][p] += 1
                total_samples += 1

    overall_acc = (sum(class_correct) / max(1, total_samples)) * 100.0

    print("\n" + "=" * 70)
    print(f"{'EVALUATION REPORT':^70}")
    print("=" * 70)
    print(f"Overall Validation Accuracy: {overall_acc:.2f}%\n")

    print("[PER-CLASS ACCURACY BREAKDOWN]")
    print("-" * 70)
    print(f"{'Class Name':<15} | {'Correct / Total':<20} | {'Accuracy':<12}")
    print("-" * 70)
    for i, name in enumerate(class_names):
        correct_c = class_correct[i]
        total_c = class_total[i]
        acc_c = (correct_c / total_c * 100.0) if total_c > 0 else 0.0
        print(f"{name:<15} | {correct_c:3d} / {total_c:3d}               | {acc_c:6.2f}%")
    print("-" * 70)

    print("\n[CONFUSION MATRIX] (Rows = True Class, Columns = Predicted Class)")
    print("-" * 70)

    header = f"{'True \\ Pred':<12} | " + " | ".join([f"{name[:6]:>6}" for name in class_names])
    print(header)
    print("-" * len(header))

    for i, true_name in enumerate(class_names):
        row_str = f"{true_name[:12]:<12} | " + " | ".join([f"{confusion_matrix[i][j]:6d}" for j in range(num_classes)])
        print(row_str)

    print("=" * 70 + "\n")
    return overall_acc


def train(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
    max_epochs: int = 20,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = 1e-4,
    model_name: str = config.MODEL_NAME,
    save_path: Path = config.MODELS_DIR / "food_classifier_finetuned.pth",
    fine_tune: bool = True,
    early_stopping_patience: int = 4
):
    """
    Main training and fine-tuning function.
    """
    print("=" * 70)
    print(f"{'AI FOOD CLASSIFIER - FINE-TUNING EXPERIMENT':^70}")
    print("=" * 70)

    device = config.DEVICE
    print(f"Device               : {device}")
    print(f"Architecture         : {model_name}")
    print(f"Fine-Tuning Mode     : {fine_tune} (Unfreezing top feature blocks)")
    print(f"Initial Learning Rate: {learning_rate}")
    print(f"Early Stop Patience  : {early_stopping_patience} epochs")
    print(f"Checkpoint Save Path : {save_path}")

    # 1. Load datasets
    try:
        train_loader, val_loader, class_names = get_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            image_size=config.IMAGE_SIZE,
            num_workers=config.NUM_WORKERS
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] Dataset loading failed: {e}")
        sys.exit(1)

    print(f"\nDetected {len(class_names)} Food Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}")

    # 2. Check class balance
    check_class_balance(train_dataset=train_loader.dataset, class_names=class_names)

    # 3. Create model
    try:
        model = create_model(
            num_classes=len(class_names),
            model_name=model_name,
            freeze_backbone=not fine_tune,
            fine_tune=fine_tune,
            unfreeze_last_n_blocks=2
        ).to(device)
    except Exception as e:
        print(f"\n[ERROR] Model creation failed: {e}")
        sys.exit(1)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter tensors: {len(trainable_params)}")

    # 4. Loss function, AdamW optimizer, and LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_val_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    actual_epochs_trained = 0

    print("\nStarting Training & Fine-Tuning Phase...")
    print("-" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12} | {'Learning Rate':<12}")
    print("-" * 75)

    for epoch in range(1, max_epochs + 1):
        actual_epochs_trained = epoch
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(
            f"[{epoch:02d}/{max_epochs:02d}]   | "
            f"{train_loss:<12.4f} | {train_acc:6.2f}%     | "
            f"{val_loss:<12.4f} | {val_acc:6.2f}%     | {current_lr:.6f}"
        )

        # Checkpoint best model
        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "model_name": model_name,
                "image_size": config.IMAGE_SIZE,
                "num_classes": len(class_names),
                "epoch": epoch,
                "val_acc": val_acc
            }
            torch.save(checkpoint, save_path)
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print(f"\n--> EARLY STOPPING TRIGGERED at Epoch {epoch}! No val acc improvement for {early_stopping_patience} consecutive epochs.")
                break

    print("-" * 75)
    print(f"Training Complete! Total Epochs Trained: {actual_epochs_trained}")
    print(f"Best Validation Accuracy : {best_val_acc:.2f}% (Achieved at Epoch {best_epoch})")
    print(f"Saved Checkpoint Path    : {save_path}")

    # Load best saved fine-tuned model for evaluation
    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluate_final_model(model, val_loader, class_names, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / Fine-Tune Food Classifier Model")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME, help="Model architecture")
    parser.add_argument("--save-path", type=str, default=str(config.MODELS_DIR / "food_classifier_finetuned.pth"), help="Save path for fine-tuned model")
    parser.add_argument("--fine-tune", action="store_true", default=True, help="Enable fine-tuning of top backbone blocks")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience in epochs")

    args = parser.parse_args()

    train(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model,
        save_path=Path(args.save_path),
        fine_tune=args.fine_tune,
        early_stopping_patience=args.patience
    )
