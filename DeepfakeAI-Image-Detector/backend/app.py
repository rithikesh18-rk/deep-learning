"""FastAPI Inference Service for Deepfake & Synthetic AI-Image Detection.

Provides endpoints for dual-stream spatial + frequency image analysis,
2D-FFT spectrum visualization, and Grad-CAM explainability heatmaps.
"""

import os
import sys
import io
import gc
import time
import uuid
import base64
import hashlib
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("spectra_backend")

logger.info("[SPECTRA INIT 1/7] Application module import started.")

import torch
# 1. Set PyTorch CPU thread counts conservatively BEFORE any model creation for cloud container stability
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
logger.info("[SPECTRA INIT 2/7] PyTorch CPU threads restricted to 1 (num_threads=1, num_interop_threads=1).")

import numpy as np
import cv2
from PIL import Image
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

# 2. Force CPU inference device
device = torch.device("cpu")
logger.info("[SPECTRA INIT 3/7] Using compute device: %s", device)

# Global single-flight inference lock: ensures serialized, race-free model execution
INFERENCE_LOCK = asyncio.Lock()

# 3. Instantiate DualStreamForensicNet on META device (0 MB RAM allocated until weights are assigned)
logger.info("[SPECTRA INIT 4/7] Instantiating DualStreamForensicNet on zero-memory meta device...")
try:
    with torch.device("meta"):
        forensic_model = DualStreamForensicNet(pretrained=False)
    model_instantiated = True
    logger.info("[SPECTRA INIT 4/7] DualStreamForensicNet meta-architecture created (0 duplicate memory).")
except Exception as e:
    forensic_model = None
    model_instantiated = False
    logger.error("[SPECTRA INIT 4/7 ERROR] Failed to instantiate DualStreamForensicNet: %s", e)

# 4. Production Checkpoint Search & In-Place Weight Assignment
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

def ensure_local_checkpoint(target_file: Path):
    """Ensures real trained model weights are on disk if file is missing, < 50MB, or LFS pointer."""
    MIN_SIZE = 50 * 1024 * 1024
    if target_file.exists() and target_file.stat().st_size >= MIN_SIZE:
        return
    try:
        import urllib.request
        logger.info("[SPECTRA CHECKPOINT] Checkpoint missing or LFS pointer (< 50MB) at %s. Downloading genuine weights...", target_file)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_file.with_suffix(".tmp")
        url = "https://media.githubusercontent.com/media/rithikesh18-rk/deep-learning/main/DeepfakeAI-Image-Detector/backend/models/deepfake_detector_improved.pth"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(temp_file, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        temp_file.replace(target_file)
        logger.info("[SPECTRA CHECKPOINT] Download complete: %s (%d bytes).", target_file, target_file.stat().st_size)
    except Exception as exc:
        logger.error("[SPECTRA CHECKPOINT ERROR] Failed to download checkpoint: %s", exc)

ensure_local_checkpoint(BASE_DIR / "models" / "deepfake_detector_improved.pth")

logger.info("[SPECTRA INIT 5/7] Searching for active trained checkpoint...")

for candidate in CHECKPOINT_CANDIDATES:
    if candidate and Path(candidate).is_file():
        try:
            logger.info("[SPECTRA INIT 5/7] Found candidate checkpoint: %s (loading to CPU)...", candidate)
            try:
                checkpoint = torch.load(candidate, map_location=device, weights_only=False, mmap=True)
            except Exception:
                checkpoint = torch.load(candidate, map_location=device, weights_only=False)

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
                # Direct in-place tensor assignment from disk buffers (zero duplicate RAM allocations)
                forensic_model.load_state_dict(state_dict, assign=True)
                forensic_model.eval()
                for p in forensic_model.parameters():
                    p.requires_grad = False
                checkpoint_loaded = True
                loaded_checkpoint_path = str(Path(candidate).resolve())
                logger.info("[SPECTRA INIT 6/7] [PRODUCTION MODEL ACTIVE] Successfully assigned weights from: %s", candidate)

                # Immediately delete temporary checkpoint dictionary and collect garbage
                del checkpoint
                del state_dict
                gc.collect()
                logger.info("[SPECTRA INIT 6/7] Checkpoint temporary objects released & garbage collected.")
                break
        except Exception as e:
            logger.warning("[SPECTRA INIT 5/7] Could not load checkpoint from %s: %s", candidate, e)

if checkpoint_loaded:
    model_status = "trained_checkpoint_loaded"
    logger.info("[SPECTRA INIT 7/7] Model ready for inference on CPU with trained weights.")
else:
    model_status = "untrained_weights_using_frequency_forensics"
    logger.warning(
        "[SPECTRA INIT 7/7 WARNING] No trained checkpoint found. Operating in fallback frequency-domain forensic mode."
    )


# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[SPECTRA LIFESPAN] Application startup event triggered.")
    logger.info("[SPECTRA LIFESPAN] Active checkpoint: %s", loaded_checkpoint_path)
    logger.info("[SPECTRA LIFESPAN] Health check endpoint ready at /api/v1/health")
    yield
    logger.info("[SPECTRA LIFESPAN] Application shutdown completed.")


# FastAPI Application Instance
app = FastAPI(
    title="SPECTRA AI Image Forensics Backend",
    description="Dual-stream spatial and frequency deepfake detection service",
    version="1.0.0",
    lifespan=lifespan
)

# Production & Local CORS Configuration
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://frontend-six-gold-14.vercel.app",
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


def generate_gradcam_overlay(
    model: torch.nn.Module,
    rgb_tensor: torch.Tensor,
    freq_tensor: torch.Tensor,
    original_bgr: np.ndarray,
    target_class: int = 1,
    req_id: str = "unknown"
) -> Optional[str]:
    """Computes Grad-CAM activation heatmap with strict lifecycle and bounded memory.
    
    Guarantees hook removal, parameter gradient clearing, and bounded overlay dimensions
    to prevent memory accumulation and cumulative state leakage across requests.
    """
    activations = []
    target_layer = model.spatial_backbone.stages[-1].blocks[-1]

    def forward_hook(module, inp, output):
        activations.clear()
        leaf = output.detach().clone().requires_grad_(True)
        activations.append(leaf)
        return leaf

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        rgb_var = rgb_tensor.to(device)
        freq_var = freq_tensor.to(device)

        logits = model(rgb_var, freq_var)
        target_score = logits[0, target_class]
        target_score.backward()

        if not activations:
            logger.warning("[%s] [GRAD-CAM] No activations captured.", req_id)
            return None

        act = activations[0]
        grad = act.grad
        if grad is None:
            logger.warning("[%s] [GRAD-CAM] Gradient not computed on leaf tensor.", req_id)
            return None

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

        # Bounded overlay dimensions (max 800px) to prevent large-image RAM spikes on Render
        h, w = original_bgr.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            target_w, target_h = int(round(w * scale)), int(round(h * scale))
            bgr_base = cv2.resize(original_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            target_w, target_h = w, h
            bgr_base = original_bgr

        cam_resized = cv2.resize(cam_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(bgr_base, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_rgb, 0.4, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode('.png', overlay_bgr)
        if not success:
            return None

        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
    except Exception as ex:
        logger.warning("[%s] [GRAD-CAM ERROR] Failed to generate overlay: %s", req_id, ex, exc_info=True)
        return None
    finally:
        try:
            handle.remove()
        except Exception:
            pass
        activations.clear()
        model.zero_grad(set_to_none=True)
        model.eval()
        gc.collect()


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Image Detector Backend",
        "architecture": model_type,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": loaded_checkpoint_path,
        "model_status": model_status,
    }


@app.get("/api/v1/health")
async def health_check():
    """Lightweight health/readiness check that returns instant status without running inference."""
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
    req_id = uuid.uuid4().hex[:8]
    t_start = time.perf_counter()
    filename = file.filename or "unknown"
    content_type = file.content_type or "application/octet-stream"

    logger.info(
        "[%s] [REQUEST_START] Ingestion started: filename='%s', content_type='%s', start_time=%.4f",
        req_id, filename, content_type, t_start
    )

    # Acquire single-flight lock: guarantees isolated sequential execution through model
    async with INFERENCE_LOCK:
        rgb_tensor = None
        freq_tensor = None
        img_bgr = None
        contents = None
        fft_spectrum_base64 = None
        gradcam_heatmap_base64 = None
        http_status = 500

        try:
            # 1. Validate file ingestion
            contents = await file.read()
            file_size = len(contents) if contents else 0

            logger.info("[%s] [INGESTED] File size=%d bytes", req_id, file_size)

            if not contents or file_size < 32:
                http_status = 400
                raise HTTPException(status_code=400, detail="Uploaded image file is empty or corrupted.")

            MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
            if file_size > MAX_UPLOAD_SIZE:
                http_status = 413
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded image exceeds maximum payload size of 50MB (received {file_size/(1024*1024):.1f}MB)."
                )

            file_sha256 = hashlib.sha256(contents).hexdigest()

            # 2. Decode raw image into BGR numpy array with safety resize cap
            try:
                img_bgr = load_image_from_bytes(contents)
                if img_bgr is None or img_bgr.size == 0:
                    raise ValueError("Failed to decode image data into pixels.")
                # Cap decoded image dimensions to prevent memory exhaustion on high-resolution camera photos
                h, w = img_bgr.shape[:2]
                if max(h, w) > 1280:
                    scale = 1280.0 / max(h, w)
                    img_bgr = cv2.resize(
                        img_bgr,
                        (int(round(w * scale)), int(round(h * scale))),
                        interpolation=cv2.INTER_AREA
                    )
            except Exception as e:
                http_status = 400
                logger.error("[%s] Image decode error: %s", req_id, e)
                raise HTTPException(
                    status_code=400,
                    detail="Failed to decode uploaded image. Unsupported or corrupted format."
                )

            img_h, img_w = img_bgr.shape[:2]

            # 3. Extract 2D Fast Fourier Transform log-magnitude spectrum
            try:
                freq_tensor, fft_spectrum_base64 = extract_fft_spectrum(contents)
                freq_tensor = freq_tensor.to(device)
            except Exception as e:
                http_status = 500
                logger.error("[%s] FFT spectrum extraction error: %s", req_id, e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Frequency domain transformation failed: {str(e)}"
                )

            # 4. Preprocess RGB image for spatial backbone
            try:
                rgb_tensor = preprocess_rgb_image(contents).to(device)
            except Exception as e:
                http_status = 500
                logger.error("[%s] RGB preprocessing error: %s", req_id, e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Spatial preprocessing failed: {str(e)}"
                )

            # 5. Compute dynamic deterministic frequency and noise residual forensic metrics
            try:
                forensic_result = compute_frequency_forensic_metrics(img_bgr)
            except Exception as e:
                http_status = 500
                logger.error("[%s] Forensic metrics calculation error: %s", req_id, e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Forensic metrics computation failed: {str(e)}"
                )

            # 6. Model Inference & Probability Determination (Inference Mode with zero autograd tracking)
            if checkpoint_loaded and forensic_model is not None:
                try:
                    forensic_model.eval()
                    with torch.inference_mode():
                        logits = forensic_model(rgb_tensor, freq_tensor)
                        probs = torch.softmax(logits, dim=-1)
                        ai_prob = float(probs[0, 1].item() * 100.0)
                    ai_probability = round(ai_prob, 2)
                    confidence = round(max(ai_probability, 100.0 - ai_probability), 2)
                    verdict = "AI-GENERATED" if ai_probability >= 50.0 else "AUTHENTIC SENSOR CAPTURE"
                    artifact_flags = forensic_result.get("artifact_flags", [])
                except Exception as e:
                    logger.error("[%s] Neural model forward pass error: %s", req_id, e)
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

            t_infer = time.perf_counter()
            logger.info(
                "[%s] [INFERENCE_COMPLETE] Verdict=%s, ai_probability=%.2f%%, inference_latency=%.3fs",
                req_id, verdict, ai_probability, (t_infer - t_start)
            )

            # 7. Generate genuine Grad-CAM explainability heatmap on ConvNeXt-Tiny (On-demand only)
            if forensic_model is not None:
                try:
                    target_class = 1 if ai_probability >= 50.0 else 0
                    gradcam_heatmap_base64 = generate_gradcam_overlay(
                        model=forensic_model,
                        rgb_tensor=rgb_tensor,
                        freq_tensor=freq_tensor,
                        original_bgr=img_bgr,
                        target_class=target_class,
                        req_id=req_id
                    )
                except Exception as e:
                    logger.warning("[%s] Grad-CAM generation failed: %s", req_id, e)
                    gradcam_heatmap_base64 = None

            metrics_data = forensic_result.get("metrics", {})
            http_status = 200

            t_end = time.perf_counter()
            logger.info(
                "[%s] [HTTP_STATUS_%d] Analysis completed in %.3fs"
                "\nfilename: %s | size: %d bytes | sha256: %s"
                "\ndimensions: %dx%d (HxW) | verdict: %s | prob: %.2f%% | conf: %.2f%%",
                req_id, http_status, (t_end - t_start),
                filename, file_size, file_sha256[:16],
                img_h, img_w,
                verdict, ai_probability, confidence
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
                "request_id": req_id,
            }

        except HTTPException as http_exc:
            http_status = http_exc.status_code
            logger.warning("[%s] [HTTP_STATUS_%d] HTTPException: %s", req_id, http_status, http_exc.detail)
            raise
        except Exception as exc:
            http_status = 500
            logger.exception("[%s] [HTTP_STATUS_500] Unexpected exception in analyze_image: %s", req_id, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Image analysis failed: {str(exc)}"
            )
        finally:
            if forensic_model is not None:
                try:
                    forensic_model.zero_grad(set_to_none=True)
                    forensic_model.eval()
                except Exception:
                    pass
            del rgb_tensor, freq_tensor, img_bgr, contents, fft_spectrum_base64, gradcam_heatmap_base64
            gc.collect()
            logger.info("[%s] [CLEANUP_COMPLETE] All request-specific buffers and tensors released (HTTP %d)", req_id, http_status)
