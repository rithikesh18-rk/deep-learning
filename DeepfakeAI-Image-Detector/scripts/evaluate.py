"""Comprehensive Checkpoint Evaluation & Multi-Dataset Benchmark Script.

Evaluates single or multiple checkpoints on:
1. Full held-out test dataset (dataset/test/)
2. External high-resolution test suite (test_images/)
3. Computes Accuracy, Precision, Recall, F1, ROC-AUC, False-Negative and False-Positive rates.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root and backend directory to sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from model import DualStreamForensicNet
from frequency_utils import (
    extract_fft_spectrum,
    preprocess_rgb_image,
    letterbox_image,
    letterbox_gray,
    RGB_NORMALIZE
)
from scripts.train_improved import MultiSourceForensicDataset, compute_fft_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(checkpoint_path: str) -> DualStreamForensicNet:
    model = DualStreamForensicNet(pretrained=False, num_classes=2).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt.state_dict()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate_dataset(model: DualStreamForensicNet, dataset_root: Path):
    ds = MultiSourceForensicDataset([dataset_root], is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

    all_targets = []
    all_probs_ai = []
    all_preds = []

    with torch.no_grad():
        for rgb, freq, targets in loader:
            rgb = rgb.to(DEVICE)
            freq = freq.to(DEVICE)

            logits = model(rgb, freq)
            probs = torch.softmax(logits, dim=-1)

            all_targets.extend(targets.numpy().tolist())
            all_probs_ai.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    all_targets = np.array(all_targets)
    all_probs_ai = np.array(all_probs_ai)
    all_preds = np.array(all_preds)

    total = len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(all_targets, all_probs_ai)
    except Exception:
        roc_auc = 0.5
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "total": total,
        "authentic_count": int(np.sum(all_targets == 0)),
        "ai_count": int(np.sum(all_targets == 1)),
        "accuracy": acc * 100.0,
        "precision": prec * 100.0,
        "recall": rec * 100.0,
        "f1": f1 * 100.0,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fnr": (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0,
        "fpr": (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0,
        "auth_probs": all_probs_ai[all_targets == 0],
        "ai_probs": all_probs_ai[all_targets == 1],
    }


def evaluate_test_images_dir(model: DualStreamForensicNet, images_dir: Path):
    results = []
    supported_exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in supported_exts])

    for f in files:
        with open(f, "rb") as fp:
            contents = fp.read()

        rgb_tensor = preprocess_rgb_image(contents).to(DEVICE)
        freq_tensor, _ = extract_fft_spectrum(contents)
        freq_tensor = freq_tensor.to(DEVICE)

        with torch.no_grad():
            logits = model(rgb_tensor, freq_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        prob_ai = float(probs[1] * 100.0)
        verdict = "AI-GENERATED" if prob_ai >= 50.0 else "AUTHENTIC SENSOR CAPTURE"

        results.append({
            "filename": f.name,
            "prob_ai": prob_ai,
            "verdict": verdict,
            "logits": [float(logits[0, 0].item()), float(logits[0, 1].item())]
        })
    return results


def print_metrics_table(label: str, metrics: dict):
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS: {label}")
    print(f"{'='*70}")
    print(f"Total Samples:             {metrics['total']}")
    print(f"Authentic Samples (0):     {metrics['authentic_count']}")
    print(f"AI-Generated Samples (1):  {metrics['ai_count']}")
    print(f"Accuracy:                  {metrics['accuracy']:.2f}%")
    print(f"Precision:                 {metrics['precision']:.2f}%")
    print(f"Recall (AI Class):         {metrics['recall']:.2f}%")
    print(f"F1-Score:                  {metrics['f1']:.2f}%")
    print(f"ROC-AUC:                   {metrics['roc_auc']:.4f}")
    print(f"False Negatives (AI -> Real): {metrics['fn']} / {metrics['ai_count']} ({metrics['fnr']:.2f}%)")
    print(f"False Positives (Real -> AI): {metrics['fp']} / {metrics['authentic_count']} ({metrics['fpr']:.2f}%)")
    print("\nConfusion Matrix:")
    print("                          Predicted Authentic | Predicted AI")
    print(f"Actual Authentic (0)     |       {metrics['tn']:5d}       |    {metrics['fp']:5d}")
    print(f"Actual AI-Generated (1)  |       {metrics['fn']:5d}       |    {metrics['tp']:5d}")
    print(f"{'='*70}\n")


def compare_models(baseline_path: str, improved_path: str):
    print("\n" + "#" * 80)
    print("SIDE-BY-SIDE MODEL BENCHMARK COMPARISON")
    print(f"Baseline: {baseline_path}")
    print(f"Improved: {improved_path}")
    print("#" * 80)

    base_model = load_model_from_checkpoint(baseline_path)
    imp_model = load_model_from_checkpoint(improved_path)

    base_res = evaluate_dataset(base_model, Path("dataset/test"))
    imp_res = evaluate_dataset(imp_model, Path("dataset/test"))

    print("\n" + "=" * 65)
    print(f"{'Metric':<30} | {'Baseline':<14} | {'Improved':<14}")
    print("-" * 65)
    print(f"{'Accuracy':<30} | {base_res['accuracy']:>13.2f}% | {imp_res['accuracy']:>13.2f}%")
    print(f"{'Precision (AI)':<30} | {base_res['precision']:>13.2f}% | {imp_res['precision']:>13.2f}%")
    print(f"{'Recall (AI)':<30} | {base_res['recall']:>13.2f}% | {imp_res['recall']:>13.2f}%")
    print(f"{'F1-Score':<30} | {base_res['f1']:>13.2f}% | {imp_res['f1']:>13.2f}%")
    print(f"{'ROC-AUC':<30} | {base_res['roc_auc']:>14.4f} | {imp_res['roc_auc']:>14.4f}")
    print(f"{'AI False Negative Rate (FNR)':<30} | {base_res['fnr']:>13.2f}% | {imp_res['fnr']:>13.2f}%")
    print(f"{'Authentic False Positive Rate (FPR)':<30} | {base_res['fpr']:>13.2f}% | {imp_res['fpr']:>13.2f}%")
    print("=" * 65)

    print("\n[External Test Images Evaluation]")
    print(f"{'Filename':<32} | {'Baseline AI%':<14} | {'Improved AI%':<14}")
    print("-" * 65)
    base_ext = evaluate_test_images_dir(base_model, Path("test_images"))
    imp_ext = evaluate_test_images_dir(imp_model, Path("test_images"))

    for b, i in zip(base_ext, imp_ext):
        print(f"{b['filename']:<32} | {b['prob_ai']:>13.2f}% | {i['prob_ai']:>13.2f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Models")
    parser.add_argument("--checkpoint", type=str, default="backend/models/deepfake_detector_best.pth", help="Checkpoint to evaluate")
    parser.add_argument("--compare", type=str, default=None, help="Second checkpoint to compare against")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Test dataset directory")
    args = parser.parse_args()

    if args.compare:
        compare_models(args.checkpoint, args.compare)
    else:
        model = load_model_from_checkpoint(args.checkpoint)
        res = evaluate_dataset(model, Path(args.test_dir))
        print_metrics_table(args.checkpoint, res)

        print("\nExternal Sample Image Analysis:")
        ext_res = evaluate_test_images_dir(model, Path("test_images"))
        for r in ext_res:
            print(f"  {r['filename']:<32} -> AI Prob: {r['prob_ai']:6.2f}% | Verdict: {r['verdict']}")
