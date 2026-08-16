"""
Script to generate clean, high-resolution demo UI screenshots for docs/screenshots/
matching the Flask web application design system.
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs" / "screenshots"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Validation Image Sources
VAL_DIR = BASE_DIR / "dataset" / "validation"
BIRYAN_IMG = VAL_DIR / "biryani" / "Hyderabadi-chicken-Biryani.jpg"
DOSA_IMG = VAL_DIR / "dosa" / "1000_F_538749842_k3j1R9Q2sWJ9q3s6mS2w0X6t7L.jpg"
IDLI_IMG = VAL_DIR / "idli" / "30_Idly.jpg"


def get_font(size=16, bold=False):
    """Try loading Arial/Segoe UI, falling back to default."""
    font_names = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    if bold:
        font_names = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] + font_names
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_homepage_screenshot():
    width, height = 960, 680
    img = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(img)

    font_title = get_font(28, bold=True)
    font_subtitle = get_font(14)
    font_badge = get_font(13, bold=True)
    font_btn = get_font(16, bold=True)
    font_pill = get_font(13, bold=True)

    # Header Badge
    draw.rounded_rectangle([320, 40, 640, 70], radius=15, fill="#064e3b", outline="#10b981", width=1)
    draw.text((480, 55), "Deep Learning & Computer Vision Demo", fill="#10b981", font=font_badge, anchor="mm")

    # Title & Subtitle
    draw.text((480, 100), "AI Food Calorie & Nutrition Estimation", fill="#f8fafc", font=font_title, anchor="mm")
    draw.text((480, 135), "Upload a food image to classify the dish using Fine-Tuned EfficientNet-B0", fill="#94a3b8", font=font_subtitle, anchor="mm")

    # Main Card
    draw.rounded_rectangle([130, 170, 830, 600], radius=16, fill="#1e293b", outline="#334155", width=1)

    # Dropzone Area
    draw.rounded_rectangle([170, 200, 790, 420], radius=12, fill="#0f172a", outline="#475569", width=2)
    
    # Camera Icon graphic
    draw.rounded_rectangle([455, 245, 505, 285], radius=6, fill="#06b6d4")
    draw.ellipse([470, 255, 490, 275], fill="#0f172a")

    draw.text((480, 320), "Choose a food image or drag & drop here", fill="#f8fafc", font=get_font(18, bold=True), anchor="mm")
    draw.text((480, 355), "Supports JPG, JPEG, PNG, WEBP (Max 16MB)", fill="#94a3b8", font=font_subtitle, anchor="mm")

    # Primary CTA Button
    draw.rounded_rectangle([170, 445, 790, 500], radius=10, fill="#10b981")
    draw.text((480, 472), "Analyze Food & Estimate Nutrition", fill="#ffffff", font=font_btn, anchor="mm")

    # Supported Pills
    draw.text((480, 530), "SUPPORTED FOOD CLASSES (6 DISHES)", fill="#94a3b8", font=get_font(12, bold=True), anchor="mm")
    
    pills = ["Biryani", "Dosa", "Idli", "Chapati", "Poori", "Sambar"]
    rx = 210
    for p in pills:
        rw = draw.textlength(p, font=font_pill) + 26
        draw.rounded_rectangle([rx, 550, rx + rw, 580], radius=15, fill="#334155")
        draw.text((rx + rw/2, 565), p, fill="#e2e8f0", font=font_pill, anchor="mm")
        rx += rw + 14

    out_path = DOCS_DIR / "homepage.png"
    img.save(out_path, quality=95)
    print(f"Generated: {out_path}")


def create_prediction_screenshot(food_name, conf_str, conf_val, serving, cal, prot, carbs, fat, source_img_path, filename):
    width, height = 960, 680
    img = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(img)

    font_title = get_font(26, bold=True)
    font_subtitle = get_font(14)
    font_badge = get_font(18, bold=True)
    font_btn = get_font(15, bold=True)

    # Header
    draw.rounded_rectangle([340, 30, 620, 58], radius=14, fill="#064e3b", outline="#10b981", width=1)
    draw.text((480, 44), "Classification & Nutrition Analysis", fill="#10b981", font=get_font(13, bold=True), anchor="mm")
    draw.text((480, 82), "Prediction Result", fill="#f8fafc", font=font_title, anchor="mm")
    draw.text((480, 112), "Analyzed via Fine-Tuned EfficientNet-B0 Classifier", fill="#94a3b8", font=font_subtitle, anchor="mm")

    # Main Card
    draw.rounded_rectangle([60, 140, 900, 630], radius=16, fill="#1e293b", outline="#334155", width=1)

    # Left: Uploaded Food Image Preview
    left_x1, left_y1, left_x2, left_y2 = 90, 170, 430, 470
    draw.rounded_rectangle([left_x1, left_y1, left_x2, left_y2], radius=12, fill="#0f172a", outline="#334155", width=1)
    
    if source_img_path.exists():
        with Image.open(source_img_path) as src_im:
            src_im = src_im.convert("RGB")
            src_im.thumbnail((left_x2 - left_x1 - 8, left_y2 - left_y1 - 8))
            w_i, h_i = src_im.size
            pos_x = left_x1 + (left_x2 - left_x1 - w_i) // 2
            pos_y = left_y1 + (left_y2 - left_y1 - h_i) // 2
            img.paste(src_im, (pos_x, pos_y))

    # Prediction Badge below image
    draw.rounded_rectangle([170, 485, 350, 520], radius=16, fill="#06b6d4")
    draw.text((260, 502), food_name, fill="#ffffff", font=font_badge, anchor="mm")

    # Confidence Bar
    draw.text((260, 535), f"Model Confidence: {conf_str}", fill="#94a3b8", font=get_font(13, bold=True), anchor="mm")
    draw.rounded_rectangle([110, 550, 410, 560], radius=5, fill="#334155")
    fill_w = int(300 * conf_val)
    draw.rounded_rectangle([110, 550, 110 + fill_w, 560], radius=5, fill="#10b981")

    # Right: Nutrition Details Column
    right_x = 460
    draw.text((right_x, 170), "Nutritional Information", fill="#f8fafc", font=get_font(20, bold=True))
    draw.text((right_x, 200), f"Serving Size: {serving}", fill="#94a3b8", font=get_font(13, bold=True))

    # Highlight Card: Calories
    draw.rounded_rectangle([right_x, 230, right_x + 400, 320], radius=12, fill="#064e3b", outline="#10b981", width=1)
    draw.text((right_x + 15, 245), "TOTAL CALORIES", fill="#10b981", font=get_font(11, bold=True))
    draw.text((right_x + 15, 275), cal, fill="#10b981", font=get_font(26, bold=True))

    # Grid Metric Cards: Protein, Carbs, Fat
    metrics = [
        ("PROTEIN", prot, right_x, 340),
        ("CARBOHYDRATES", carbs, right_x + 205, 340),
        ("FAT", fat, right_x, 420),
        ("METRIC RATING", "Balanced", right_x + 205, 420)
    ]
    for label, val, mx, my in metrics:
        draw.rounded_rectangle([mx, my, mx + 195, my + 70], radius=10, fill="#0f172a", outline="#334155", width=1)
        draw.text((mx + 12, my + 12), label, fill="#94a3b8", font=get_font(10, bold=True))
        draw.text((mx + 12, my + 38), val, fill="#f8fafc", font=get_font(16, bold=True))

    # Action Button
    draw.rounded_rectangle([460, 510, 860, 560], radius=10, fill="#10b981")
    draw.text((660, 535), "Classify Another Food Image", fill="#ffffff", font=font_btn, anchor="mm")

    out_path = DOCS_DIR / filename
    img.save(out_path, quality=95)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    print("Generating clean demo screenshots in docs/screenshots/...")
    create_homepage_screenshot()
    
    create_prediction_screenshot(
        food_name="Biryani",
        conf_str="85.17%",
        conf_val=0.8517,
        serving="1 plate (~300g)",
        cal="450 kcal",
        prot="18.0 g",
        carbs="55.0 g",
        fat="16.0 g",
        source_img_path=BIRYAN_IMG,
        filename="prediction_biryani.png"
    )

    create_prediction_screenshot(
        food_name="Dosa",
        conf_str="96.10%",
        conf_val=0.9610,
        serving="1 plate (1 medium dosa / ~150g)",
        cal="250 kcal",
        prot="6.0 g",
        carbs="45.0 g",
        fat="6.0 g",
        source_img_path=DOSA_IMG,
        filename="prediction_dosa.png"
    )

    create_prediction_screenshot(
        food_name="Idli",
        conf_str="83.47%",
        conf_val=0.8347,
        serving="1 plate (2 pieces / ~100g)",
        cal="130 kcal",
        prot="4.0 g",
        carbs="28.0 g",
        fat="0.5 g",
        source_img_path=IDLI_IMG,
        filename="prediction_idli.png"
    )
    print("All clean demo screenshots generated successfully!")
