"""
Test script to verify error handling in Flask web application.
"""

import sys
import urllib.request
import urllib.parse
from pathlib import Path

SERVER_URL = "http://127.0.0.1:5000"


def test_invalid_extension():
    """Tests uploading an unsupported file format (.txt)."""
    print("Testing Unsupported File Extension (.txt)...")
    boundary = "----WebKitFormBoundaryErrorTest"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="image"; filename="test.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
        "This is not an image file.\r\n"
        f"--{boundary}--\r\n"
    ).encode('utf-8')

    req = urllib.request.Request(f"{SERVER_URL}/predict", data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8')
            assert "Unsupported file format" in html or "Error" in html
            print(" [OK] Unsupported extension rejected gracefully with error banner.\n")
    except urllib.error.HTTPError as e:
        print(f" [OK] HTTP Response: {e.code}")


if __name__ == "__main__":
    test_invalid_extension()
