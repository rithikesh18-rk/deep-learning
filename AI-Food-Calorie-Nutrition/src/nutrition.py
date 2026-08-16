"""
Nutrition database module for 6 target food classes:
dosa, idli, biryani, chapati, poori, sambar.

Provides local approximate nutritional values per standard serving size.
"""

from typing import Dict, Any

# Local Nutrition Database for Supported Foods
NUTRITION_DATABASE: Dict[str, Dict[str, Any]] = {
    "dosa": {
        "food_name": "Dosa",
        "serving_size": "1 plate (1 medium dosa / ~150g)",
        "calories": "250 kcal",
        "protein": "6.0 g",
        "carbohydrates": "45.0 g",
        "fat": "6.0 g"
    },
    "idli": {
        "food_name": "Idli",
        "serving_size": "1 plate (2 pieces / ~100g)",
        "calories": "130 kcal",
        "protein": "4.0 g",
        "carbohydrates": "28.0 g",
        "fat": "0.5 g"
    },
    "biryani": {
        "food_name": "Biryani",
        "serving_size": "1 plate (~300g)",
        "calories": "450 kcal",
        "protein": "18.0 g",
        "carbohydrates": "55.0 g",
        "fat": "16.0 g"
    },
    "chapati": {
        "food_name": "Chapati",
        "serving_size": "1 plate (2 chapatis / ~80g)",
        "calories": "140 kcal",
        "protein": "4.0 g",
        "carbohydrates": "24.0 g",
        "fat": "3.0 g"
    },
    "poori": {
        "food_name": "Poori",
        "serving_size": "1 plate (2 pooris with bhaji / ~120g)",
        "calories": "300 kcal",
        "protein": "5.0 g",
        "carbohydrates": "35.0 g",
        "fat": "16.0 g"
    },
    "sambar": {
        "food_name": "Sambar",
        "serving_size": "1 bowl (~200ml)",
        "calories": "110 kcal",
        "protein": "5.0 g",
        "carbohydrates": "18.0 g",
        "fat": "3.0 g"
    }
}


def get_nutrition_info(class_name: str) -> Dict[str, Any]:
    """
    Lookup nutrition data for a predicted food class name.

    Args:
        class_name (str): Raw or formatted predicted food class name.

    Returns:
        Dict[str, Any]: Dictionary containing food_name, serving_size, calories, protein, carbohydrates, and fat.
    """
    key = class_name.strip().lower()
    
    if key in NUTRITION_DATABASE:
        return NUTRITION_DATABASE[key].copy()
    
    # Fallback for unknown classes
    return {
        "food_name": class_name.capitalize(),
        "serving_size": "Unknown / Standard Serving",
        "calories": "N/A",
        "protein": "N/A",
        "carbohydrates": "N/A",
        "fat": "N/A"
    }


def get_all_nutrition_data() -> Dict[str, Dict[str, Any]]:
    """
    Returns complete local nutrition database dictionary.
    """
    return NUTRITION_DATABASE.copy()
