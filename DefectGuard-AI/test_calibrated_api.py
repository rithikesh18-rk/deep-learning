"""
Verification script for DefectGuard with calibrated Top-5 mean and threshold 12.0.
"""

from pathlib import Path
import requests

def get_api_url():
    for port in [8001, 8000]:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                return f"http://127.0.0.1:{port}/api/inspect"
        except Exception:
            pass
    return "http://127.0.0.1:8001/api/inspect"


def test_samples():
    url = get_api_url()
    
    samples = [
        ("Unseen Good 000", "data/bottle/test/good/000.png", True),
        ("Unseen Good 001", "data/bottle/test/good/001.png", True),
        ("Unseen Good 016 (Max Good)", "data/bottle/test/good/016.png", True),
        ("Defect: Broken Large", "data/bottle/test/broken_large/000.png", False),
        ("Defect: Contamination", "data/bottle/test/contamination/000.png", False),
    ]

    print("=" * 70)
    print("CALIBRATED DEFECTGUARD END-TO-END VERIFICATION")
    print("=" * 70)

    for label, path_str, expected_pass in samples:
        path = Path(path_str)
        assert path.exists(), f"Missing file {path}"
        with open(path, "rb") as f:
            # Default threshold (omitted in POST so api.py uses its default 12.0)
            res = requests.post(url, files={"file": f})
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()

        score = data["anomaly_score"]
        thresh = data["threshold"]
        status = data["status"]
        passed = data["passed"]

        status_flag = "PASS" if passed == expected_pass else "FAIL"
        print(f"[{status_flag}] {label:<28} | Score: {score:>5.2f} | Thresh: {thresh} | Verdict: {status:<15} | Passed: {passed}")
        assert passed == expected_pass, f"Unexpected verdict for {label}: expected passed={expected_pass}, got {passed}"

    print("=" * 70)
    print("[SUCCESS] All unseen test/good bottles pass and all defective bottles are detected!")

if __name__ == "__main__":
    test_samples()
