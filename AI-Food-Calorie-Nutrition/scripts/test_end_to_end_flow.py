"""
Automated test script to reproduce and verify the exact upload -> prediction -> response flow.
"""

import sys
import re
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import src.config as config
from src.predict import predict_image

SERVER_URL = "http://127.0.0.1:5000"
TEST_IMAGE = BASE_DIR / "dataset" / "validation" / "biryani" / "Hyderabadi-chicken-Biryani.jpg"


def test_direct_predict_image():
    """Direct predict_image() verification."""
    print("=" * 80)
    print("1. DIRECT predict_image() INFERENCE TEST")
    print("=" * 80)
    assert TEST_IMAGE.exists(), f"Test image missing: {TEST_IMAGE}"
    
    result = predict_image(image_path=TEST_IMAGE, model_path=config.MODEL_PATH)
    
    assert result["food_name"] == "Biryani"
    assert "confidence" in result
    assert "confidence_percentage" in result
    assert "serving_size" in result
    assert "calories" in result
    assert "protein" in result
    assert "carbohydrates" in result
    assert "fat" in result

    print(f" [OK] Direct Inference Result: {result['food_name']} ({result['confidence_percentage']})")
    print(f"      Calories: {result['calories']}, Serving: {result['serving_size']}")
    print(" [OK] All returned dictionary keys verified.\n")


def test_multipart_post_predict():
    """Real multipart POST /predict request test."""
    print("=" * 80)
    print("2. REAL MULTIPART POST /predict HTTP TEST")
    print("=" * 80)

    boundary = "----WebKitFormBoundaryE2ETestBoundary"
    lines = []
    lines.append(f"--{boundary}".encode('utf-8'))
    lines.append(f'Content-Disposition: form-data; name="image"; filename="{TEST_IMAGE.name}"'.encode('utf-8'))
    lines.append(b'Content-Type: image/jpeg')
    lines.append(b'')
    with open(TEST_IMAGE, 'rb') as f:
        lines.append(f.read())
    lines.append(f"--{boundary}--".encode('utf-8'))
    lines.append(b'')
    body = b'\r\n'.join(lines)

    req = urllib.request.Request(f"{SERVER_URL}/predict", data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8')
            assert resp.status == 200, f"Expected HTTP 200, got {resp.status}"
            assert "Prediction Result" in html
            assert "Biryani" in html
            assert "450 kcal" in html
            print(" [OK] HTTP POST /predict returned 200 OK.")
            print(" [OK] Result page HTML contains predicted food 'Biryani' and calories '450 kcal'.\n")
    except Exception as e:
        print(f" [FAIL] Multipart POST /predict failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_direct_predict_image()
    test_multipart_post_predict()
