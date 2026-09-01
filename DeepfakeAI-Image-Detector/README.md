# SPECTRA — AI Digital Forensics & Deepfake Detection Suite

A deep-learning powered digital forensics system combining **Spatial Domain Convolutional Features** (via ConvNeXt-Tiny) and **Frequency Domain Log-Magnitude Spectra** (via windowed 2D Fast Fourier Transform & 4-Layer Spectrum CNN) to detect AI-generated imagery and synthetic manipulations with explainable Grad-CAM heatmaps.

---

## 🌟 Architecture Overview

```
Input Image (RGB)
  ├── 1. Spatial Stream: ConvNeXt-Tiny (Pretrained) -> 768-D Spatial Feature Vector
  ├── 2. Frequency Stream: 2D-FFT (Hann-Windowed + High-Pass Circular Mask) -> 4-Layer CNN -> 256-D Spectral Feature Vector
  └── 3. Fusion Classifier: Concatenation (1024-D) -> Linear(1024, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 2)
        └── Output: [Authentic vs. AI-Generated Probability & Confidence]
        └── Explainability: ConvNeXt Stage-4 Grad-CAM Localized Activation Heatmaps
```

---

## 🚀 Quick Start

### 1. Backend Service (FastAPI)

```bash
# Navigate to project root and run with the dedicated virtualenv:
.\backend\venv\Scripts\uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/v1/health`

### 2. Frontend Web Application (React + Vite + Tailwind CSS)

```bash
# Navigate to frontend/ directory:
cd frontend
npm.cmd run dev
```

- Web UI: `http://localhost:5173`

---

## 📡 API Endpoints

### `GET /api/v1/health`
Returns service health, loaded compute device (CPU/CUDA), and active model checkpoint metadata.

### `POST /api/v1/analyze`
Accepts `multipart/form-data` with an image file (`file`).

**Response JSON Structure:**
```json
{
  "verdict": "AI-GENERATED",
  "ai_probability": 94.25,
  "confidence": 94.25,
  "artifact_flags": [
    "Upsampling Grid Anomaly",
    "Diffusion Latent Smoothing"
  ],
  "fft_spectrum_base64": "data:image/png;base64,...",
  "gradcam_heatmap_base64": "data:image/png;base64,...",
  "model_status": "trained_checkpoint_loaded",
  "checkpoint_loaded": true,
  "metrics": {
    "peak_zscore": 4.62,
    "top_1pct_diff": 88.4,
    "grid_spike_score": 32.1,
    "smooth_patch_ratio": 0.38,
    "r_squared": 0.684
  }
}
```

---

## 🔬 Model Training & Evaluation

To train or fine-tune the dual-stream neural network on the CIFAKE dataset:

```bash
# Train on full dataset (7,000 train / 1,500 val / 1,500 test):
.\backend\venv\Scripts\python.exe scripts/train.py --epochs 5 --batch-size 64

# Fast training on a balanced sample subset:
.\backend\venv\Scripts\python.exe scripts/train.py --epochs 2 --batch-size 32 --max-train-samples 1000 --max-val-samples 300 --max-test-samples 300 --log-interval 5
```

Checkpoints are automatically saved to:
- `backend/models/deepfake_detector_best.pth` (Best validation F1)
- `backend/models/training_checkpoint_latest.pth` (Resumable latest state)
