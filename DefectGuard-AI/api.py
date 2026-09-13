"""
DefectGuard - FastAPI Inference Bridge.
Provides RESTful API endpoints for AI visual inspection using PatchCore and FAISS.
Designed for Stitch MCP frontend integration.
"""

import io
import time
import base64
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

from src.feature_extractor import PatchCoreFeatureExtractor, isolate_foreground, generate_patch_mask
from src.memory_bank import PatchCoreMemoryBank

ROOT_DIR = Path(__file__).resolve().parent
INDEX_PATH = ROOT_DIR / "models" / "bottle_patchcore.index"
META_PATH = ROOT_DIR / "models" / "bottle_patchcore_meta.json"
INDEX_HTML_PATH = ROOT_DIR / "index.html"

# Global model state container
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the model and FAISS memory bank on startup."""
    print("[API] Starting DefectGuard Inference Bridge...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[API] Initializing WideResNet-50-2 feature extractor on {device}...")
    extractor = PatchCoreFeatureExtractor(device=device)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. Please run build_index.py first."
        )

    print(f"[API] Loading FAISS index from {INDEX_PATH}...")
    memory_bank = PatchCoreMemoryBank(feature_dim=1536, index_path=INDEX_PATH)
    print(f"[API] FAISS index loaded successfully with {memory_bank.index.ntotal:,} vectors.")

    state["extractor"] = extractor
    state["memory_bank"] = memory_bank
    state["device"] = device
    state["grid_size"] = (28, 28)

    yield

    print("[API] Shutting down DefectGuard Inference Bridge...")
    state.clear()


app = FastAPI(
    title="DefectGuard Inference Bridge",
    description="FastAPI service for PatchCore industrial defect detection.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_image_to_base64_jpeg(image_rgb: np.ndarray, quality: int = 90) -> str:
    """Converts an RGB numpy image to a base64 Data URL JPEG string."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded_buf = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not success:
        raise RuntimeError("Failed to encode image to JPEG buffer.")
    b64_string = base64.b64encode(encoded_buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_string}"


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse(content={
            "service": "DefectGuard Inference Bridge",
            "status": "online",
            "model": "PatchCore (WideResNet-50-2 + FAISS)",
            "docs": "/docs",
            "inspect_endpoint": "/api/inspect",
        })
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DefectGuard AI Visual Inspection Service</h1>")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard HTML not found</h1>")


@app.get("/api/sample/{category}")
async def get_sample_image(category: str):
    """Provides sample test images for one-click testing in UI."""
    sample_map = {
        "broken_large": ROOT_DIR / "data" / "bottle" / "test" / "broken_large" / "000.png",
        "broken_small": ROOT_DIR / "data" / "bottle" / "test" / "broken_small" / "000.png",
        "contamination": ROOT_DIR / "data" / "bottle" / "test" / "contamination" / "000.png",
        "good": ROOT_DIR / "data" / "bottle" / "test" / "good" / "000.png",
    }
    path = sample_map.get(category.lower())
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"Sample for category '{category}' not found.")
    return FileResponse(path, media_type="image/png")


@app.api_route("/health", methods=["GET", "HEAD"])
@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "index_loaded": "memory_bank" in state and state["memory_bank"].index.ntotal > 0,
        "memory_bank_size": state["memory_bank"].index.ntotal if "memory_bank" in state else 0,
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)



# Calibrated default threshold separating isolated nominal (<12.0) from defects (>12.0)
CALIBRATED_DEFAULT_THRESHOLD: float = 12.0


@app.post("/api/inspect")
async def inspect_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    query_threshold: Optional[float] = Query(None, alias="threshold"),
):
    """
    Analyzes an uploaded image for defects using PatchCore memory bank similarity search.

    Args:
        file: Uploaded image file (PNG, JPEG, WebP, etc.).
        threshold: Anomaly threshold (form-data or query parameter, defaults to calibrated 12.0).

    Returns:
        JSON response with anomaly score, defect verdict, heatmap, and overlay images.
    """
    t_start = time.time()

    # Determine threshold (supports both form field and query param, defaults to calibrated 12.0)
    actual_threshold: float = CALIBRATED_DEFAULT_THRESHOLD
    if threshold is not None:
        actual_threshold = float(threshold)
    elif query_threshold is not None:
        actual_threshold = float(query_threshold)

    # 1. Read & Validate Uploaded Image
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded file is empty.")
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(err)}",
        )

    # 2. Resize to 224x224 for consistent processing & visualization
    target_size = (224, 224)
    orig_resized_pil = pil_image.resize(target_size, Image.Resampling.BILINEAR)
    orig_resized_rgb = np.array(orig_resized_pil)

    # 3. Automatic Background Isolation & Foreground Masking
    # Segments bottle from non-black backgrounds (Otsu thresholding + contour extraction)
    # and forces all background pixels to pure black [0, 0, 0]
    isolated_pil, binary_mask = isolate_foreground(orig_resized_pil)
    patch_mask = generate_patch_mask(binary_mask, grid_size=(28, 28))

    # 4. Preprocess & Extract Features from Isolated Bottle Image
    extractor: PatchCoreFeatureExtractor = state["extractor"]
    memory_bank: PatchCoreMemoryBank = state["memory_bank"]

    tensor = extractor.preprocess_image(isolated_pil)
    with torch.no_grad():
        patch_tokens, grid_size = extractor(tensor)

    # 5. Masked Nearest-Neighbor Anomaly Scoring
    # Only computes FAISS nearest-neighbor distances for patches falling within the
    # detected object foreground mask, ignoring surrounding background completely.
    prediction = memory_bank.predict(
        patch_tokens=patch_tokens,
        grid_size=grid_size,
        image_size=target_size,
        gaussian_sigma=4.0,
        patch_mask=patch_mask,
        full_mask=binary_mask,
    )

    raw_score = float(prediction["image_scores"])
    anomaly_map = prediction["anomaly_maps"]  # 2D float array [224, 224]

    # 6. Generate Visual Artifacts: Jet Heatmap & Blended Overlay
    mask_224 = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Normalize anomaly map to 0-255 uint8 for colormapping
    max_val = float(np.max(anomaly_map))
    if max_val > 0.0:
        norm_map = (np.clip(anomaly_map / max_val, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        norm_map = np.zeros(target_size, dtype=np.uint8)

    # Apply Jet Colormap (OpenCV returns BGR)
    heatmap_bgr = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    # Zero out heatmap outside the foreground bottle contour
    heatmap_rgb[mask_224 == 0] = [0, 0, 0]

    # Blend original image (60%) and heatmap (40%) on bottle, keep background clean
    blended_overlay_rgb = orig_resized_rgb.copy()
    fg_blended = cv2.addWeighted(
        orig_resized_rgb,
        0.6,
        heatmap_rgb,
        0.4,
        0,
    )
    blended_overlay_rgb[mask_224 > 0] = fg_blended[mask_224 > 0]

    # 7. Encode outputs to Base64 strings
    heatmap_b64 = encode_image_to_base64_jpeg(heatmap_rgb)
    overlay_b64 = encode_image_to_base64_jpeg(blended_overlay_rgb)

    # 8. Formulate Verdict & Response
    is_anomaly = raw_score >= actual_threshold
    status_text = "DEFECT DETECTED" if is_anomaly else "NOMINAL (PASS)"
    passed = not is_anomaly

    inference_ms = round((time.time() - t_start) * 1000, 2)

    print(
        f"[API] /api/inspect | File: {file.filename} | Score: {raw_score:.2f} | "
        f"Threshold: {actual_threshold:.2f} | Verdict: {status_text} (Passed: {passed})",
        flush=True,
    )

    response_data = {
        "anomaly_score": round(raw_score, 2),
        "threshold": round(actual_threshold, 2),
        "status": status_text,
        "passed": bool(passed),
        "heatmap_base64": heatmap_b64,
        "overlay_base64": overlay_b64,
        "inference_time_ms": inference_ms,
        "filename": file.filename,
        "foreground_patches": int(np.sum(patch_mask)),
        "total_patches": int(patch_mask.size),
    }

    return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
