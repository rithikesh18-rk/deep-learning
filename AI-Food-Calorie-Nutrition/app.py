"""
Flask Web Application for AI Food Calorie & Nutrition Estimation.

Serves a clean local web demo interface using the fine-tuned model checkpoint
and local nutrition database.
"""

import os
import sys
import uuid
import logging
from pathlib import Path

# Environment variables to constrain CPU threading on shared web hosts (Linux/Render)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

import src.config as config
from src.predict import predict_image

# Setup Logger for Flask Web Service
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AI_Food_WebApp")

app = Flask(__name__)
app.secret_key = "ai_food_calorie_nutrition_demo_secret_key"

# Configuration & Absolute Directory Paths
UPLOAD_FOLDER = (config.BASE_DIR / "static" / "uploads").resolve()
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PRIMARY_MODEL_PATH = (config.MODELS_DIR / "food_classifier_finetuned.pth").resolve()
if not PRIMARY_MODEL_PATH.exists():
    PRIMARY_MODEL_PATH = (config.MODELS_DIR / "food_classifier.pth").resolve()

logger.info(f"App initialized. Base Dir: '{config.BASE_DIR}', Model Path: '{PRIMARY_MODEL_PATH}'")


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
    logger.info("Stage 1: Received POST /predict request")

    if "image" not in request.files:
        logger.warning("Rejecting request: No 'image' field in request.files")
        return redirect(url_for("index", error="No image file provided in request."))

    file = request.files["image"]

    if file.filename == "":
        logger.warning("Rejecting request: Empty filename selected")
        return redirect(url_for("index", error="No image file selected. Please select a food image."))

    if not is_allowed_file(file.filename):
        ext = Path(file.filename).suffix.lower()
        logger.warning(f"Rejecting request: Unsupported file extension '{ext}'")
        return redirect(
            url_for(
                "index",
                error=f"Unsupported file format '{ext}'. Supported formats: JPG, JPEG, PNG, WEBP."
            )
        )

    saved_path = None
    try:
        # Generate safe unique filename
        ext = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        saved_path = (UPLOAD_FOLDER / unique_filename).resolve()
        
        logger.info(f"Stage 1b: Saving uploaded image to path: '{saved_path}'")
        file.save(saved_path)

        # Validate that the file is indeed a readable valid image
        logger.info("Stage 1c: Validating image readability via PIL...")
        try:
            with Image.open(saved_path) as img:
                img.convert("RGB")
        except Exception as img_err:
            logger.error(f"Image validation failed for '{saved_path.name}': {img_err}")
            if saved_path.exists():
                saved_path.unlink()
            return redirect(url_for("index", error="The uploaded file appears to be corrupted or not a valid image."))

        # Perform inference and nutrition lookup
        logger.info(f"Stage 1d: Invoking predict_image(saved_path='{saved_path.name}', model_path='{PRIMARY_MODEL_PATH.name}')")
        result = predict_image(image_path=saved_path, model_path=PRIMARY_MODEL_PATH)

        image_url = url_for("static", filename=f"uploads/{unique_filename}")
        logger.info(f"Stage 7: Successfully rendering result page for food='{result['food_name']}', confidence='{result['confidence_percentage']}'")

        return render_template("result.html", result=result, image_url=image_url)

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in /predict handler: {e}", exc_info=True)
        if saved_path and saved_path.exists():
            saved_path.unlink()
        return redirect(url_for("index", error=f"Model checkpoint file error: {e}"))
    except Exception as e:
        logger.error(f"Unhandled Exception in /predict handler: {e}", exc_info=True)
        if saved_path and saved_path.exists():
            saved_path.unlink()
        return redirect(url_for("index", error=f"An error occurred during image analysis: {str(e)}"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting AI Food Calorie & Nutrition Web Application...")
    print(f"Primary Model Checkpoint : {PRIMARY_MODEL_PATH}")
    print(f"Server URL               : http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
