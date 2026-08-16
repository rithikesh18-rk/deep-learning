"""
Flask Web Application for AI Food Calorie & Nutrition Estimation.

Serves a clean local web demo interface using the fine-tuned model checkpoint
and local nutrition database.
"""

import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image

import src.config as config
from src.predict import predict_image, get_loaded_model

app = Flask(__name__)
app.secret_key = "ai_food_calorie_nutrition_demo_secret_key"

# Configuration
UPLOAD_FOLDER = config.BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PRIMARY_MODEL_PATH = config.MODELS_DIR / "food_classifier_finetuned.pth"
if not PRIMARY_MODEL_PATH.exists():
    PRIMARY_MODEL_PATH = config.MODELS_DIR / "food_classifier.pth"

# Warm-up model in memory on app startup to prevent request latency and worker crashes
try:
    print(f"Pre-loading model into memory from: {PRIMARY_MODEL_PATH}")
    get_loaded_model(PRIMARY_MODEL_PATH)
    print("Model pre-loaded successfully.")
except Exception as _e:
    print(f"Warning: Model pre-loading failed: {_e}")


def is_allowed_file(filename: str) -> bool:
    """Checks if uploaded filename has an allowed extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    """Renders home page with file upload form."""
    error = request.args.get("error", None)
    return render_template("index.html", error=error)


@app.route("/predict", methods=["POST"])
def predict():
    """Handles food image upload, runs inference, and renders prediction result."""
    if "image" not in request.files:
        return redirect(url_for("index", error="No image file provided in request."))

    file = request.files["image"]

    if file.filename == "":
        return redirect(url_for("index", error="No image file selected. Please select a food image."))

    if not is_allowed_file(file.filename):
        return redirect(
            url_for(
                "index",
                error=f"Unsupported file format '{Path(file.filename).suffix}'. Supported formats: JPG, JPEG, PNG, WEBP."
            )
        )

    try:
        # Generate safe unique filename
        ext = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        saved_path = UPLOAD_FOLDER / unique_filename
        file.save(saved_path)

        # Validate that the file is indeed a readable valid image
        try:
            with Image.open(saved_path) as img:
                img.verify()
        except Exception:
            # Delete invalid file
            if saved_path.exists():
                saved_path.unlink()
            return redirect(url_for("index", error="The uploaded file appears to be corrupted or not a valid image."))

        # Perform inference and nutrition lookup
        result = predict_image(image_path=saved_path, model_path=PRIMARY_MODEL_PATH)
        image_url = url_for("static", filename=f"uploads/{unique_filename}")

        return render_template("result.html", result=result, image_url=image_url)

    except FileNotFoundError as e:
        return redirect(url_for("index", error=f"Model error: {e}"))
    except Exception as e:
        return redirect(url_for("index", error=f"An error occurred during image analysis: {str(e)}"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting AI Food Calorie & Nutrition Web Application...")
    print(f"Primary Model Checkpoint : {PRIMARY_MODEL_PATH}")
    print(f"Server URL               : http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
