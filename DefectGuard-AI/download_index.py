"""
DefectGuard - Index Downloader for Render Build & Deployment.
Ensures genuine PatchCore FAISS index exists on disk at models/bottle_patchcore.index.
If the index file is missing, empty, or a Git LFS pointer (<50MB), it downloads
the index binary from an external URL configured via the INDEX_DOWNLOAD_URL env var.
Fails the build process with exit code 1 if the index cannot be obtained.
"""

import os
import sys
import urllib.request
from pathlib import Path

INDEX_FILENAME = "bottle_patchcore.index"
MIN_FILE_SIZE = 50 * 1024 * 1024  # Genuine index is ~1.006 GB; Git LFS pointer is ~130 bytes


def ensure_index(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    index_path = target_dir / INDEX_FILENAME

    if index_path.exists():
        size = index_path.stat().st_size
        if size >= MIN_FILE_SIZE:
            print(f"[INDEX CHECK] Genuine FAISS index exists at {index_path} ({size:,} bytes).")
            return index_path
        print(
            f"[INDEX CHECK] File at {index_path} is {size} bytes (likely missing or Git LFS pointer). "
            "Downloading genuine binary..."
        )
    else:
        print(f"[INDEX CHECK] FAISS index not found at {index_path}. Downloading...")

    download_url = os.environ.get("INDEX_DOWNLOAD_URL", "").strip()
    if not download_url:
        print(
            "[FATAL] INDEX_DOWNLOAD_URL environment variable is not set, and genuine index "
            f"was not found on disk at {index_path}.\n"
            "Please configure INDEX_DOWNLOAD_URL in your Render Environment Variables with "
            "the direct download link (e.g. GitHub Releases asset URL).",
            file=sys.stderr,
        )
        sys.exit(1)

    temp_path = target_dir / f"{INDEX_FILENAME}.tmp"
    req = urllib.request.Request(download_url, headers={"User-Agent": "DefectGuard-Deploy/1.0"})
    print(f"[INDEX DOWNLOAD] Streaming genuine index from {download_url}...")

    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(temp_path, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(2 * 1024 * 1024)  # 2MB chunks
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if total % (50 * 1024 * 1024) < (2 * 1024 * 1024):
                    print(f"[INDEX DOWNLOAD] Downloaded {total // (1024 * 1024)} MB...")

        temp_path.replace(index_path)
        final_size = index_path.stat().st_size
        if final_size < MIN_FILE_SIZE:
            print(
                f"[FATAL] Downloaded file size ({final_size:,} bytes) is below expected threshold "
                f"({MIN_FILE_SIZE:,} bytes). Download may be corrupted or incomplete.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[INDEX DOWNLOAD COMPLETE] Successfully saved {index_path} ({final_size:,} bytes).")
        return index_path
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[FATAL] Error downloading model index: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    models_directory = base_dir / "models"
    result = ensure_index(models_directory)
    if not result.exists() or result.stat().st_size < MIN_FILE_SIZE:
        print(f"[FATAL] Index validation failed. File {result} missing or too small.", file=sys.stderr)
        sys.exit(1)
