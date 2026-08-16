"""
Inference script to predict food category and confidence score from an input image.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Union
import torch
import torch.nn.functional as F

# Environment variables to constrain CPU threading on shared hosting (Linux/Render)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Safe PyTorch CPU Thread settings
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Add parent directory to path when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.dataset import load_single_image
from src.model import create_model
from src.nutrition import get_nutrition_info

# Setup Logger
logger = logging.getLogger("AI_Food_Predictor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(ch)

import threading

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def get_loaded_model(model_path: Union[str, Path] = config.MODEL_PATH) -> Dict[str, Any]:
    """
    Thread-safe lazy loading and memory caching of model instance.

    Args:
        model_path (Union[str, Path]): Path to trained checkpoint.

    Returns:
        Dict[str, Any]: Cached dictionary with 'model', 'class_names', 'image_size', 'device'.
    """
    model_path = Path(model_path).resolve()
    key = str(model_path)

    if key not in _MODEL_CACHE:
        with _MODEL_LOCK:
            if key not in _MODEL_CACHE:
                logger.info(f"Stage 3a: Loading model checkpoint from resolved path: '{model_path}'")
                if not model_path.exists():
                    logger.error(f"Checkpoint file missing at: '{model_path}'")
                    raise FileNotFoundError(f"Trained model checkpoint not found at path: '{model_path}'")

                device = torch.device("cpu")

                logger.info("Stage 3b: Loading torch state_dict onto CPU...")
                checkpoint = torch.load(model_path, map_location=device)
                class_names = checkpoint["class_names"]
                model_name = checkpoint.get("model_name", config.MODEL_NAME)
                image_size = checkpoint.get("image_size", config.IMAGE_SIZE)

                logger.info(f"Stage 3c: Reconstructing '{model_name}' (num_classes={len(class_names)}) with pretrained=False...")
                model = create_model(
                    num_classes=len(class_names),
                    model_name=model_name,
                    freeze_backbone=False,
                    pretrained=False
                ).to(device)

                logger.info("Stage 3d: Applying model state_dict...")
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                _MODEL_CACHE[key] = {
                    "model": model,
                    "class_names": class_names,
                    "image_size": image_size,
                    "device": device
                }
                logger.info("Stage 3e: Model cached in memory successfully.")

    return _MODEL_CACHE[key]


def predict_image(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = config.MODEL_PATH
) -> Dict[str, Any]:
    """
    Predicts the food category, confidence score, and returns nutrition info for a given image.

    Args:
        image_path (Union[str, Path]): Path to the target image file.
        model_path (Union[str, Path]): Path to the trained model checkpoint (.pth).

    Returns:
        Dict[str, Any]: Dictionary containing food_name, confidence, serving_size, calories, protein, carbohydrates, fat.
    """
    image_path = Path(image_path).resolve()
    logger.info(f"Stage 2: Starting predict_image() for image: '{image_path.name}'")

    if not image_path.exists():
        logger.error(f"Input image not found: '{image_path}'")
        raise FileNotFoundError(f"Input image not found at path: '{image_path}'")

    model_data = get_loaded_model(model_path)
    model = model_data["model"]
    class_names = model_data["class_names"]
    image_size = model_data["image_size"]
    device = model_data["device"]

    logger.info("Stage 4: Preprocessing input image tensor via PIL transform...")
    image_tensor = load_single_image(image_path=image_path, image_size=image_size).to(device)

    logger.info("Stage 5: Executing PyTorch inference (torch.no_grad)...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)

    predicted_idx = predicted_idx_tensor.item()
    confidence = confidence_tensor.item()
    raw_class_name = class_names[predicted_idx]
    logger.info(f"Stage 5 Result: Predicted Class='{raw_class_name}', Confidence={confidence * 100.0:.2f}%")

    logger.info(f"Stage 6: Retrieving local nutrition info for '{raw_class_name}'...")
    nutrition = get_nutrition_info(raw_class_name)

    return {
        "food_name": nutrition["food_name"],
        "predicted_class": raw_class_name,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100.0:.2f}%",
        "serving_size": nutrition["serving_size"],
        "calories": nutrition["calories"],
        "protein": nutrition["protein"],
        "carbohydrates": nutrition["carbohydrates"],
        "fat": nutrition["fat"]
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Food Category & Nutrition from Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input food image")
    parser.add_argument("--model-path", type=str, default=str(config.MODEL_PATH), help="Path to saved model checkpoint (.pth)")

    args = parser.parse_args()

    try:
        result = predict_image(image_path=args.image, model_path=args.model_path)

        print("\n" + "=" * 55)
        print(" AI Food Calorie & Nutrition Classifier - Result ")
        print("=" * 55)
        print(f" Image Path       : {args.image}")
        print(f" Predicted Food   : {result['food_name']}")
        print(f" Confidence Score : {result['confidence_percentage']} ({result['confidence']:.4f})")
        print("-" * 55)
        print(" NUTRITIONAL INFORMATION (per serving)")
        print(f" Serving Size     : {result['serving_size']}")
        print(f" Calories         : {result['calories']}")
        print(f" Protein          : {result['protein']}")
        print(f" Carbohydrates    : {result['carbohydrates']}")
        print(f" Fat              : {result['fat']}")
        print("=" * 55 + "\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An error occurred during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
