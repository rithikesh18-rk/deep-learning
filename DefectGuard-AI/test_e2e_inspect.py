"""
End-to-End Verification for DefectGuard Inspection Dashboard & API.
"""

import io
import json
import base64
from pathlib import Path
import requests
from PIL import Image

def get_api_url():
    for port in [8001, 8000]:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                return f"http://127.0.0.1:{port}/api/inspect"
        except Exception:
            pass
    return "http://127.0.0.1:8001/api/inspect"


def main():
    test_img_path = Path("data/bottle/test/broken_large/000.png")
    if not test_img_path.exists():
        raise FileNotFoundError(f"Test image {test_img_path} does not exist.")

    url = get_api_url()
    print(f"[TEST] Sending {test_img_path} to {url} with threshold=7.5...")

    with open(test_img_path, "rb") as f:
        res = requests.post(url, files={"file": f}, data={"threshold": 7.5})

    if res.status_code != 200:
        raise RuntimeError(f"API returned status {res.status_code}: {res.text}")

    data = res.json()
    print("=" * 60)
    print("END-TO-END DEFECT VERIFICATION RESULT")
    print("=" * 60)
    print(f"Sample: {test_img_path.name}")
    print(f"Anomaly Score     : {data['anomaly_score']}")
    print(f"Threshold         : {data['threshold']}")
    print(f"Status            : {data['status']}")
    print(f"Passed            : {data['passed']}")
    print(f"Inference Latency : {data['inference_time_ms']} ms")

    # Assertions
    assert data["status"] == "DEFECT DETECTED", f"Expected 'DEFECT DETECTED', got {data['status']}"
    assert data["passed"] is False, f"Expected passed=False, got {data['passed']}"
    assert data["anomaly_score"] >= data["threshold"], "Score should exceed threshold for defect sample"

    # Decode and verify visual artifacts
    heatmap_bytes = base64.b64decode(data["heatmap_base64"].split(",")[1])
    overlay_bytes = base64.b64decode(data["overlay_base64"].split(",")[1])

    img_heat = Image.open(io.BytesIO(heatmap_bytes))
    img_over = Image.open(io.BytesIO(overlay_bytes))

    print(f"Decoded Heatmap   : {img_heat.size} px, {img_heat.format}")
    print(f"Decoded Overlay   : {img_over.size} px, {img_over.format}")

    # Save output artifacts for inspection
    out_dir = Path("data/test_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_heat.save(out_dir / "test_broken_large_heatmap.jpg")
    img_over.save(out_dir / "test_broken_large_overlay.jpg")
    print(f"[SUCCESS] Saved verified test artifacts to {out_dir}/")

if __name__ == "__main__":
    main()
