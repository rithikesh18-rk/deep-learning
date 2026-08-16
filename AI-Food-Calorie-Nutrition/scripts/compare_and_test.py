"""
Script to evaluate and compare the baseline model (food_classifier.pth)
versus the fine-tuned model (food_classifier_finetuned.pth),
and run multi-sample prediction tests.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import torch

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.predict import predict_image

OLD_MODEL_PATH = config.MODELS_DIR / "food_classifier.pth"
NEW_MODEL_PATH = config.MODELS_DIR / "food_classifier_finetuned.pth"

# Sample validation images to test prediction across different classes
TEST_SAMPLES = [
    ("dataset/validation/biryani/Hyderabadi-chicken-Biryani.jpg", "biryani"),
    ("dataset/validation/chapati/Soft-Rotis-Chapati-Recipe-Step-By-Step-Instructions-500x500.jpg", "chapati"),
    ("dataset/validation/dosa/1000_F_538749842_k3j1R9Q2sWJ9q3s6mS2w0X6t7L.jpg", "dosa"),
    ("dataset/validation/idli/30_Idly.jpg", "idli"),
    ("dataset/validation/poori/Puri-Recipe.jpg", "poori"),
    ("dataset/validation/sambar/sambar-recipe.jpg", "sambar")
]


def evaluate_model_checkpoint(model_path: Path) -> Dict[str, Any]:
    """Loads a checkpoint and extracts accuracy and metadata."""
    if not model_path.exists():
        return {"exists": False}
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    return {
        "exists": True,
        "val_acc": checkpoint.get("val_acc", 0.0),
        "epoch": checkpoint.get("epoch", 0),
        "model_name": checkpoint.get("model_name", "efficientnet_b0"),
        "class_names": checkpoint.get("class_names", [])
    }


def compare_models():
    """Compares baseline vs fine-tuned model and prints recommendation."""
    print("=" * 70)
    print(f"{'MODEL COMPARISON & RECOMMENDATION SUMMARY':^70}")
    print("=" * 70)

    old_info = evaluate_model_checkpoint(OLD_MODEL_PATH)
    new_info = evaluate_model_checkpoint(NEW_MODEL_PATH)

    old_acc = old_info.get("val_acc", 83.33)
    new_acc = new_info.get("val_acc", 0.0) if new_info["exists"] else 0.0

    print(f"OLD MODEL  (`models/food_classifier.pth`)          : Validation Accuracy = {old_acc:.2f}%")
    if new_info["exists"]:
        print(f"NEW MODEL  (`models/food_classifier_finetuned.pth`): Validation Accuracy = {new_acc:.2f}%")
    else:
        print("NEW MODEL  (`models/food_classifier_finetuned.pth`): Not trained yet")

    print("-" * 70)
    print("[MODEL SELECTION DECISION]")
    if new_info["exists"]:
        if new_acc > old_acc:
            print(f"  [OK] IMPROVEMENT CONFIRMED! (+{new_acc - old_acc:.2f}% accuracy gain)")
            print("  RECOMMENDATION: Use `models/food_classifier_finetuned.pth` as the primary recommended model.")
            print("  Backup model `models/food_classifier.pth` is safely preserved.")
        elif new_acc == old_acc:
            print("  [=] PARITY: Fine-tuned model achieved identical accuracy to baseline.")
            print("  RECOMMENDATION: Both models preserved; fine-tuned model saved to `models/food_classifier_finetuned.pth`.")
        else:
            print(f"  [X] Fine-tuned model achieved lower accuracy ({new_acc:.2f}% vs {old_acc:.2f}%).")
            print("  RECOMMENDATION: Retain baseline `models/food_classifier.pth` as primary model.")
    print("=" * 70 + "\n")


def run_prediction_tests(model_path: Path):
    """Runs prediction test suite across validation sample images."""
    if not model_path.exists():
        print(f"Cannot run prediction test: {model_path} does not exist.")
        return

    print("=" * 70)
    print(f"{'PREDICTION TEST SUITE':^70}")
    print(f"Model Under Test: {model_path.name}")
    print("=" * 70)
    print(f"{'Image Name':<45} | {'True Class':<10} | {'Predicted':<10} | {'Confidence':<10}")
    print("-" * 88)

    for rel_path, true_class in TEST_SAMPLES:
        full_path = config.BASE_DIR / rel_path
        if not full_path.exists():
            val_folder = config.VAL_DIR / true_class
            images = [f for f in val_folder.iterdir() if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp'}]
            if images:
                full_path = images[0]
            else:
                continue

        try:
            res = predict_image(image_path=full_path, model_path=model_path)
            pred_class = res["food_name"]
            conf_str = res["confidence_percentage"]
            match_symbol = "[OK]" if pred_class.lower() == true_class.lower() else "[X]"
            print(f"{full_path.name[:43]:<45} | {true_class:<10} | {pred_class:<10} {match_symbol} | {conf_str:<10}")
        except Exception as e:
            print(f"{full_path.name[:43]:<45} | {true_class:<10} | ERROR: {e}")

    print("=" * 88 + "\n")


if __name__ == "__main__":
    target_model = NEW_MODEL_PATH if NEW_MODEL_PATH.exists() else OLD_MODEL_PATH
    compare_models()
    run_prediction_tests(model_path=target_model)
