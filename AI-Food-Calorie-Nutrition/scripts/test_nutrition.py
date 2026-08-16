"""
Test script to verify Phase 4: Nutrition Database Integration across all 6 food classes.
Tests direct utility lookup and end-to-end image inference using the fine-tuned model checkpoint.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.nutrition import get_nutrition_info
from src.predict import predict_image

MODEL_PATH = config.MODELS_DIR / "food_classifier_finetuned.pth"

# Validation images for each class
SAMPLE_IMAGES = {
    "biryani": "dataset/validation/biryani/Hyderabadi-chicken-Biryani.jpg",
    "chapati": "dataset/validation/chapati/Soft-Rotis-Chapati-Recipe-Step-By-Step-Instructions-500x500.jpg",
    "dosa": "dataset/validation/dosa/1000_F_538749842_k3j1R9Q2sWJ9q3s6mS2w0X6t7L.jpg",
    "idli": "dataset/validation/idli/30_Idly.jpg",
    "poori": "dataset/validation/poori/Puri-Recipe.jpg",
    "sambar": "dataset/validation/sambar/sambar-recipe.jpg"
}


def test_direct_nutrition_lookup():
    """Verifies that all 6 classes return exact nutrition database entries."""
    print("=" * 105)
    print("1. DIRECT NUTRITION UTILITY UNIT TEST (All 6 Classes)")
    print("=" * 105)
    
    classes = ["dosa", "idli", "biryani", "chapati", "poori", "sambar"]
    all_passed = True

    header_class = "Class Key"
    header_name = "Food Name"
    header_serving = "Serving Size"
    header_cal = "Calories"
    header_prot = "Protein"
    header_carbs = "Carbs"
    header_fat = "Fat"

    print(f"{header_class:<10} | {header_name:<10} | {header_serving:<40} | {header_cal:<10} | {header_prot:<8} | {header_carbs:<8} | {header_fat:<6}")
    print("-" * 105)

    for c in classes:
        info = get_nutrition_info(c)
        if info["calories"] == "N/A":
            all_passed = False
        fname = info["food_name"]
        ssize = info["serving_size"]
        cal = info["calories"]
        prot = info["protein"]
        carbs = info["carbohydrates"]
        fat = info["fat"]
        print(f"{c:<10} | {fname:<10} | {ssize:<40} | {cal:<10} | {prot:<8} | {carbs:<8} | {fat:<6}")

    print("-" * 105)
    if all_passed:
        print("[SUCCESS] All 6 food classes returned valid local nutrition data.\n")
    else:
        print("[FAIL] Some food classes failed nutrition lookup.\n")


def test_end_to_end_predictions():
    """Runs prediction pipeline on sample validation images and prints combined results."""
    print("=" * 105)
    print(f"2. END-TO-END PREDICTION & NUTRITION TEST USING `{MODEL_PATH.name}`")
    print("=" * 105)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Fine-tuned model checkpoint not found at: {MODEL_PATH}")
        return

    header_target = "Target Class"
    header_pred = "Predicted"
    header_conf = "Conf."
    header_serving = "Serving Size"
    header_cal = "Calories"
    header_prot = "Protein"
    header_carbs = "Carbs"
    header_fat = "Fat"

    print(f"{header_target:<12} | {header_pred:<10} | {header_conf:<8} | {header_serving:<40} | {header_cal:<9} | {header_prot:<7} | {header_carbs:<7} | {header_fat:<6}")
    print("-" * 115)

    for food_class, rel_path in SAMPLE_IMAGES.items():
        img_path = config.BASE_DIR / rel_path
        if not img_path.exists():
            val_folder = config.VAL_DIR / food_class
            images = [f for f in val_folder.iterdir() if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp'}]
            if images:
                img_path = images[0]
            else:
                print(f"{food_class:<12} | [IMAGE NOT FOUND]")
                continue

        try:
            res = predict_image(image_path=img_path, model_path=MODEL_PATH)
            fname = res["food_name"]
            conf = res["confidence_percentage"]
            ssize = res["serving_size"]
            cal = res["calories"]
            prot = res["protein"]
            carbs = res["carbohydrates"]
            fat = res["fat"]
            print(f"{food_class:<12} | {fname:<10} | {conf:<8} | {ssize:<40} | {cal:<9} | {prot:<7} | {carbs:<7} | {fat:<6}")
        except Exception as e:
            print(f"{food_class:<12} | ERROR: {e}")

    print("=" * 115 + "\n")


if __name__ == "__main__":
    test_direct_nutrition_lookup()
    test_end_to_end_predictions()
