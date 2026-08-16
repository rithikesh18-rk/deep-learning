"""
Integration test script for Flask Web Demo.
Tests GET / home route and POST /predict file upload route across 3 sample food images.
"""

import sys
import re
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import urllib.request
import urllib.error

SERVER_URL = "http://127.0.0.1:5000"

SAMPLE_IMAGES = [
    ("Biryani", "dataset/validation/biryani/Hyderabadi-chicken-Biryani.jpg"),
    ("Idli", "dataset/validation/idli/30_Idly.jpg"),
    ("Dosa", "dataset/validation/dosa/1000_F_538749842_k3j1R9Q2sWJ9q3s6mS2w0X6t7L.jpg")
]


def encode_multipart_formdata(fields, files):
    """Simple multipart/form-data encoder using standard library."""
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    lines = []
    for (key, value) in fields:
        lines.append(f"--{boundary}".encode('utf-8'))
        lines.append(f'Content-Disposition: form-data; name="{key}"'.encode('utf-8'))
        lines.append(b'')
        lines.append(value.encode('utf-8'))
    for (key, filename, filepath) in files:
        lines.append(f"--{boundary}".encode('utf-8'))
        lines.append(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"'.encode('utf-8'))
        lines.append(b'Content-Type: image/jpeg')
        lines.append(b'')
        with open(filepath, 'rb') as f:
            lines.append(f.read())
    lines.append(f"--{boundary}--".encode('utf-8'))
    lines.append(b'')
    body = b'\r\n'.join(lines)
    content_type = f'multipart/form-data; boundary={boundary}'
    return content_type, body


def test_home_page():
    """Tests GET / route."""
    print("=" * 80)
    print("1. TESTING HOME PAGE (GET /)")
    print("=" * 80)
    req = urllib.request.Request(SERVER_URL)
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8')
            assert resp.status == 200
            assert "AI Food Calorie & Nutrition Estimation" in html
            assert "Analyze Food & Estimate Nutrition" in html
            print(" [OK] Home page loaded successfully (HTTP 200).")
    except Exception as e:
        print(f" [FAIL] Home page test failed: {e}")
        sys.exit(1)


def test_image_predictions():
    """Tests POST /predict with 3 sample food images."""
    print("\n" + "=" * 80)
    print("2. TESTING PREDICTION FLOW (POST /predict with 3 Sample Images)")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent

    for name, rel_path in SAMPLE_IMAGES:
        img_path = base_dir / rel_path
        if not img_path.exists():
            val_folder = base_dir / "dataset" / "validation" / name.lower()
            imgs = list(val_folder.glob("*.jpg")) + list(val_folder.glob("*.png"))
            if imgs:
                img_path = imgs[0]
            else:
                print(f" [SKIP] Image file for {name} not found.")
                continue

        content_type, body = encode_multipart_formdata([], [('image', img_path.name, str(img_path))])
        req = urllib.request.Request(f"{SERVER_URL}/predict", data=body)
        req.add_header('Content-Type', content_type)

        try:
            with urllib.request.urlopen(req) as resp:
                html = resp.read().decode('utf-8')
                assert resp.status == 200
                
                # Check required elements in output HTML
                has_img = "uploads/" in html
                has_pred = "pred-badge" in html
                has_conf = "Model Confidence" in html
                has_nutrition = "Nutritional Information" in html
                has_cal = "Total Calories" in html
                has_prot = "Protein" in html
                has_carbs = "Carbohydrates" in html
                has_fat = "Fat" in html

                # Extract predicted name and confidence
                pred_match = re.search(r'<div class="pred-badge">\s*([^<]+)\s*</div>', html)
                conf_match = re.search(r'Model Confidence:\s*<strong>([^<]+)</strong>', html)
                cal_match = re.search(r'<div class="metric-label">🔥 Total Calories</div>\s*<div class="metric-value">([^<]+)</div>', html)

                pred_name = pred_match.group(1).strip() if pred_match else "N/A"
                conf_val = conf_match.group(1).strip() if conf_match else "N/A"
                cal_val = cal_match.group(1).strip() if cal_match else "N/A"

                if has_img and has_pred and has_conf and has_nutrition and has_cal and has_prot and has_carbs and has_fat:
                    print(f" [OK] {name:<10} Image: Upload OK | Predicted: {pred_name:<10} | Conf: {conf_val:<8} | Calories: {cal_val}")
                else:
                    print(f" [FAIL] {name:<10} Missing required HTML elements in result page.")

        except Exception as e:
            print(f" [FAIL] Prediction test failed for {name}: {e}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_home_page()
    test_image_predictions()
