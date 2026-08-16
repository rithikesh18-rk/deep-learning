"""
Inference script to predict food category and confidence score from an input image.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Union
import torch
import torch.nn.functional as F

# Add parent directory to path when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.dataset import load_single_image
from src.model import create_model
from src.nutrition import get_nutrition_info


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
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found at path: '{image_path}'")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at path: '{model_path}'. "
            "Please run 'python src/train.py' first after preparing your dataset."
        )

    device = config.DEVICE

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", config.MODEL_NAME)
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)

    # Reconstruct model architecture and load trained weights
    model = create_model(
        num_classes=len(class_names),
        model_name=model_name,
        freeze_backbone=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Preprocess single input image
    image_tensor = load_single_image(image_path=image_path, image_size=image_size).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence_tensor, predicted_idx_tensor = torch.max(probabilities, dim=1)

    predicted_idx = predicted_idx_tensor.item()
    confidence = confidence_tensor.item()
    raw_class_name = class_names[predicted_idx]

    # Retrieve local nutrition data
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
