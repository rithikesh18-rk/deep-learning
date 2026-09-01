"""FastAPI Inference Service for Deepfake & Synthetic AI-Image Detection.

Provides endpoints for dual-stream spatial + frequency image analysis,
2D-FFT spectrum visualization, and Grad-CAM explainability heatmaps.
"""

import os
import sys
import io
import base64
import hashlib
import logging
from pathlib import Path
from typing import Optional

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import DualStreamForensicNet
from frequency_utils import (
    extract_fft_spectrum,
    preprocess_rgb_image,
    compute_frequency_forensic_metrics,
    load_image_from_bytes,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spectra_backend")

app = FastAPI(
    title="SPECTRA AI Image Forensics Backend",
    description="Dual-stream spatial and frequency deepfake detection service",
    version="1.0.0"
)

# Production & Local CORS Configuration
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

env_cors = os.environ.get("CORS_ORIGINS") or os.environ.get("FRONTEND_URL")
if env_cors:
    custom_origins = [o.strip() for o in env_cors.split(",") if o.strip()]
    allowed_origins = list(set(default_origins + custom_origins))
else:
    allowed_origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using compute device: %s", device)

# Initialize DualStreamForensicNet once at startup
try:
    forensic_model = DualStreamForensicNet(pretrained=True).to(device)
    forensic_model.eval()
    model_instantiated = True
    logger.info("DualStreamForensicNet instantiated successfully.")
except Exception as e:
    forensic_model = None
    model_instantiated = False
    logger.error("Failed to instantiate DualStreamForensicNet: %s", e)

# Production Checkpoint Loader
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CWD = Path.cwd()

CHECKPOINT_CANDIDATES = [
    os.environ.get("CHECKPOINT_PATH"),
    str(BASE_DIR / "models" / "deepfake_detector_improved.pth"),
    str(PROJECT_ROOT / "backend" / "models" / "deepfake_detector_improved.pth"),
    str(CWD / "backend" / "models" / "deepfake_detector_improved.pth"),
    str(CWD / "models" / "deepfake_detector_improved.pth"),
    str(CWD / "DeepfakeAI-Image-Detector" / "backend" / "models" / "deepfake_detector_improved.pth"),
    str(BASE_DIR / "models" / "deepfake_detector_best.pth"),
    str(PROJECT_ROOT / "backend" / "models" / "deepfake_detector_best.pth"),
    str(BASE_DIR / "models" / "deepfake_detector.pth"),
    "models/deepfake_detector_improved.pth",
    "backend/models/deepfake_detector_improved.pth",
    "DeepfakeAI-Image-Detector/backend/models/deepfake_detector_improved.pth",
    "models/deepfake_detector_best.pth",
    "backend/models/deepfake_detector_best.pth",
]

checkpoint_loaded = False
loaded_checkpoint_path: Optional[str] = None
model_type = "DualStreamForensicNet (ConvNeXt-Tiny + 2D-FFT)"

for candidate in CHECKPOINT_CANDIDATES:
    if candidate and Path(candidate).is_file():
        try:
            logger.info("Attempting to load checkpoint from: %s", candidate)
            try:
                checkpoint = torch.load(candidate, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(candidate, map_location=device)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()

            if forensic_model is not None:
                forensic_model.load_state_dict(state_dict, strict=False)
                checkpoint_loaded = True
                loaded_checkpoint_path = str(Path(candidate).resolve())
                logger.info("[PRODUCTION MODEL ACTIVE] Successfully loaded weights from: %s", candidate)
                break
        except Exception as e:
            logger.warning("Could not load checkpoint from %s: %s", candidate, e)

if checkpoint_loaded:
    model_status = "trained_checkpoint_loaded"
else:
    model_status = "untrained_weights_using_frequency_forensics"
    logger.warning(
        "[WARNING] No trained checkpoint found. Operating in fallback frequency-domain forensic mode."
    )


def generate_gradcam_overlay(
    model: torch.nn.Module,
    rgb_tensor: torch.Tensor,
    freq_tensor: torch.Tensor,
    original_bgr: np.ndarray,
    target_class: int = 1,
) -> str:
    """Computes genuine Grad-CAM activation heatmap on ConvNeXt spatial backbone."""
    activations = []
    gradients = []

    target_layer = model.spatial_backbone.stages[-1].blocks[-1]

    def forward_hook(module, inp, output):
        activations.clear()
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.clear()
        gradients.append(grad_out[0])

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        rgb_var = rgb_tensor.clone().detach().to(device).requires_grad_(True)
        freq_var = freq_tensor.clone().detach().to(device)

        model.zero_grad()
        logits = model(rgb_var, freq_var)

        target_score = logits[0, target_class]
        target_score.backward()

        if not activations or not gradients:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        act = activations[0]
        grad = gradients[0]

        # Channel-wise global average pooling of gradients
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()
        h, w = original_bgr.shape[:2]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_rgb, 0.4, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode('.png', overlay_bgr)
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM overlay image.")

        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
    finally:
        f_handle.remove()
        b_handle.remove()


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Image Detector Backend",
        "architecture": "DualStreamForensicNet (ConvNeXt-Tiny + 2D-FFT)",
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": loaded_checkpoint_path,
        "model_status": model_status,
    }


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_instantiated,
        "model_type": model_type,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": loaded_checkpoint_path,
        "model_status": model_status,
        "device": str(device),
    }


@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 1. Validate file ingestion
    contents = await file.read()
    if not contents or len(contents) < 32:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty or corrupted.")

    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded image exceeds maximum payload size of 50MB (received {len(contents)/(1024*1024):.1f}MB)."
        )

    file_sha256 = hashlib.sha256(contents).hexdigest()
    filename = file.filename or "unknown"

    # 2. Decode raw image into BGR numpy array
    try:
        img_bgr = load_image_from_bytes(contents)
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Failed to decode image data into pixels.")
    except Exception as e:
        logger.error("Image decode error: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format or decoding error: {str(e)}"
        )

    img_h, img_w = img_bgr.shape[:2]

    # 3. Extract 2D-FFT spectrum and spectrum tensor via frequency_utils
    try:
        freq_tensor, fft_spectrum_base64 = extract_fft_spectrum(contents)
        freq_tensor = freq_tensor.to(device)
    except Exception as e:
        logger.error("FFT spectrum extraction error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Frequency domain transformation failed: {str(e)}"
        )

    # 4. Preprocess RGB image for spatial backbone
    try:
        rgb_tensor = preprocess_rgb_image(contents).to(device)
    except Exception as e:
        logger.error("RGB preprocessing error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Spatial preprocessing failed: {str(e)}"
        )

    # 5. Compute dynamic deterministic frequency and noise residual forensic metrics
    try:
        forensic_result = compute_frequency_forensic_metrics(img_bgr)
    except Exception as e:
        logger.error("Forensic metrics calculation error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Forensic metrics computation failed: {str(e)}"
        )

    # 6. Model Inference & Probability Determination
    if checkpoint_loaded and forensic_model is not None:
        try:
            with torch.no_grad():
                logits = forensic_model(rgb_tensor, freq_tensor)
                probs = torch.softmax(logits, dim=-1)
                ai_prob = float(probs[0, 1].item() * 100.0)
            ai_probability = round(ai_prob, 2)
            confidence = round(max(ai_probability, 100.0 - ai_probability), 2)
            verdict = "AI-GENERATED" if ai_probability >= 50.0 else "AUTHENTIC SENSOR CAPTURE"
            artifact_flags = forensic_result.get("artifact_flags", [])
        except Exception as e:
            logger.error("Neural model forward pass error: %s", e)
            ai_probability = forensic_result["ai_probability"]
            confidence = forensic_result["confidence"]
            verdict = forensic_result["verdict"]
            artifact_flags = forensic_result.get("artifact_flags", [])
    else:
        # Transparently use calibrated frequency and noise residual forensic analysis
        ai_probability = forensic_result["ai_probability"]
        confidence = forensic_result["confidence"]
        verdict = forensic_result["verdict"]
        artifact_flags = forensic_result.get("artifact_flags", [])

    # 7. Generate genuine Grad-CAM explainability heatmap on ConvNeXt-Tiny
    gradcam_heatmap_base64 = None
    if forensic_model is not None:
        try:
            target_class = 1 if ai_probability >= 50.0 else 0
            gradcam_heatmap_base64 = generate_gradcam_overlay(
                model=forensic_model,
                rgb_tensor=rgb_tensor,
                freq_tensor=freq_tensor,
                original_bgr=img_bgr,
                target_class=target_class,
            )
        except Exception as e:
            logger.warning("Grad-CAM generation failed: %s", e)
            gradcam_heatmap_base64 = None

    metrics_data = forensic_result.get("metrics", {})

    # Debug telemetry logging per request
    logger.info(
        "\n--- [ANALYZE REQUEST] ---"
        "\nfilename: %s | size: %d bytes | sha256: %s"
        "\ndimensions: %dx%d (HxW)"
        "\nverdict: %s | ai_probability: %.2f%% | confidence: %.2f%%"
        "\npeak_zscore: %.2f | top_1pct_diff: %.2f dB"
        "\ngrid_spike_score: %.1f | smooth_patch_ratio: %.2f | r_squared: %.3f"
        "\nartifact_flags: %s"
        "\n-------------------------",
        filename, len(contents), file_sha256[:16],
        img_h, img_w,
        verdict, ai_probability, confidence,
        metrics_data.get("peak_zscore", 0.0),
        metrics_data.get("top_1pct_diff", 0.0),
        metrics_data.get("grid_spike_score", 0.0),
        metrics_data.get("smooth_patch_ratio", 0.0),
        metrics_data.get("r_squared", 0.0),
        artifact_flags
    )

    return {
        "verdict": verdict,
        "ai_probability": ai_probability,
        "confidence": confidence,
        "artifact_flags": artifact_flags,
        "fft_spectrum_base64": fft_spectrum_base64,
        "gradcam_heatmap_base64": gradcam_heatmap_base64,
        "model_status": model_status,
        "checkpoint_loaded": checkpoint_loaded,
        "metrics": metrics_data,
    }