# ??? DefectGuard AI — Cyber-Industrial Visual QA System

**DefectGuard AI** is an industrial-grade Automated Optical Inspection (AOI) anomaly detection platform engineered for manufacturing lines. Utilizing **PatchCore** memory-bank architecture with **WideResNet-50-2** feature extraction and **FAISS** vector search, DefectGuard detects surface defects, structural fractures, contamination, and structural anomalies with zero defect training data required (unsupervised anomaly detection).

---

## ?? Key Features

- **Unsupervised Anomaly Detection**: Trained exclusively on nominal/good industrial bottle samples; identifies any unforeseen defect.
- **Multi-Scale Patch Embeddings**: Extracts mid-level tokens across intermediate layers (layer2 + layer3) of WideResNet-50-2 for rich context.
- **Coreset Subsampling & FAISS Indexing**: Employs Greedy k-Center coreset reduction (1% retention) coupled with high-throughput L2 vector indexing for sub-100ms inference.
- **Foreground Isolation & Background Invariance**: Automatic Otsu foreground segmentation ensures robust anomaly scoring across varying illumination and backgrounds.
- **Cyber-Industrial Web Dashboard**: Interactive single-page dashboard featuring a Before/After split inspection slider, defect localization reticles, colormap legends, and tactical audio feedback.
- **Calibrated Decision Boundary**: Rigorously calibrated threshold (.0$) delivering zero false alarms on nominal items while detecting broken bodies, cracked necks, and foreign contaminants.

---

## ?? Repository Structure

`
DefectGuard-AI/
+-- api.py                      # FastAPI REST service & dashboard host
+-- build_index.py              # Feature extraction & FAISS memory bank builder
+-- calibrate_threshold.py      # Optimal classification threshold calibration
+-- setup_data.py               # MVTec AD bottle dataset setup & downloader
+-- index.html                  # Cyber-Industrial Visual QA Dashboard
+-- stitch_screen.html          # UI Design System template
+-- requirements.txt            # Python dependencies
+-- configs/
¦   +-- config.yaml             # System configuration parameters
+-- src/
¦   +-- __init__.py
¦   +-- feature_extractor.py    # WideResNet-50-2 patch extraction & foreground mask
¦   +-- memory_bank.py          # PatchCore memory bank & anomaly scoring engine
¦   +-- coreset.py              # Greedy k-Center coreset subsampling
+-- models/
¦   +-- bottle_patchcore_meta.json      # Metadata & feature dimensionality
¦   +-- threshold_calibration.json     # Calibrated score statistics
¦   +-- .gitkeep
+-- data/
¦   +-- bottle/                 # MVTec AD dataset partition (train/test/ground_truth)
+-- tests/
    +-- test_calibrated_api.py  # End-to-end API threshold validation
    +-- test_e2e_inspect.py     # End-to-end inspection pipeline test
    +-- test_white_bg_bottle.py # Robustness test against white backgrounds
    +-- verify_good_bottle_score.py
`

---

## ??? Quick Start

### 1. Environment Setup

Clone the repository and install the dependencies:

`powershell
cd DefectGuard-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
`

### 2. Dataset Preparation (Optional)

If downloading the dataset from scratch:

`powershell
python setup_data.py
`

### 3. Build the Memory Bank Index

Extracts nominal patch representations from the training set and constructs the FAISS index:

`powershell
python build_index.py
`
*The generated ottle_patchcore.index will be saved inside models/.*

### 4. Launch the Local Service

Start the FastAPI inference bridge and web interface:

`powershell
python api.py
# Or via uvicorn:
uvicorn api:app --host 0.0.0.0 --port 8000
`

- **Visual Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## ?? REST API Reference

### POST /api/inspect
Inspects an uploaded bottle image and outputs anomaly scoring with localization masks.

- **Parameters**:
  - ile: Image binary (multipart/form-data)
  - 	hreshold: *(Optional)* Anomaly score threshold (Default: 12.0)
- **Response**:
`json
{
  "anomaly_score": 14.82,
  "threshold": 12.0,
  "status": "DEFECT DETECTED",
  "passed": false,
  "heatmap_base64": "data:image/jpeg;base64,...",
  "overlay_base64": "data:image/jpeg;base64,...",
  "inference_time_ms": 68.4,
  "filename": "sample_broken_neck.png",
  "foreground_patches": 312,
  "total_patches": 784
}
`

### GET /api/sample/{category}
Retrieves built-in sample images for one-click testing (roken_large, roken_small, contamination, good).

---

## ?? Verification & Testing

Execute the automated test suite to confirm end-to-end functionality:

`powershell
# Verify calibrated threshold performance across sample defects and good bottles
python test_calibrated_api.py

# Verify end-to-end inspection pipeline
python test_e2e_inspect.py

# Verify background invariance
python test_white_bg_bottle.py
`

---

## ?? License
This project is licensed under the MIT License. MVTec AD dataset is subject to its respective non-commercial research terms.
