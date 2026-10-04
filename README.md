# 🧠 Deep Learning & AI Projects

A curated portfolio of production-grade **Deep Learning**, **Computer Vision**, **Generative & Forensic AI**, and **Retrieval-Augmented Generation (RAG)** systems developed by **[Rithikesh S](https://github.com/rithikesh18-rk)**.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 🧠 Overview

Welcome to my Deep Learning and Artificial Intelligence project portfolio repository. This repository contains my projects across **Deep Learning**, **Computer Vision**, **AI**, **image classification**, **object detection**, **generative/forensic AI**, and **RAG-based systems**.

Designed to showcase clean code practices, production-ready engineering, and applied machine learning research, each project is fully implemented with dedicated data pipelines, model training and evaluation scripts, and interactive web applications deployed across modern cloud platforms.

---

## 🚀 Project Highlights

- **[AI Helmet Detection System](./AI-Helmet-Detection-System/)**: Real-time safety compliance monitoring using custom fine-tuned YOLOv8 across images, video streams, and live webcam feeds.
- **[SPECTRA — AI Digital Forensics / Deepfake Detector](./DeepfakeAI-Image-Detector/)**: Dual-stream spatial (ConvNeXt-Tiny) and frequency-domain (2D-FFT) neural network with Grad-CAM explainability heatmaps for synthetic image detection.
- **[DocuMind — RAG Assistant](./docu-mind-rag-assistant/)**: Sub-second document Q&A and structured summarization powered by Groq LPUs (Llama-3.1-8B), FAISS dense vector search, and anti-hallucination guardrails.
- **[DefectGuard AI](./DefectGuard-AI/)**: Unsupervised Automated Optical Inspection (AOI) utilizing WideResNet-50-2 patch embeddings, 99% coreset reduction, and sub-100ms inference.
- **[AI Food Calorie & Nutrition](./AI-Food-Calorie-Nutrition/)**: Fine-tuned EfficientNet-B0 food classifier (88.33% validation accuracy) coupled with an instant local database lookup for nutritional metrics.
- **[Deep Learning Web Application](./app.py)**: Production monorepo ASGI application gateway and REST API deployed on Render Cloud.

---

## 🌐 Live Deployments & Application Access

| Project | Platform | Access Link / Button | Status |
| :--- | :--- | :--- | :--- |
| **Deep Learning Web Application** | ⚡ Render Cloud | 🚀 [Launch Live Deep Learning App](https://deep-learning-ap09.onrender.com/) | 🟢 Active |
| **AI Helmet Detection System** | ⚡ Render Cloud | 🚀 [Launch AI Helmet Detection](https://ai-helmet-detection-system.onrender.com/) | 🟢 Active |
| **SPECTRA — AI Digital Forensics / Deepfake Detector** | ▲ Vercel | 🚀 [Launch SPECTRA Forensics](https://spectra-forensics.vercel.app/) | 🟢 Active |
| **DocuMind — RAG Assistant** | 🎈 Streamlit Cloud | 🚀 [Launch DocuMind RAG Assistant](https://documind-rag-assistant.streamlit.app/) | 🟢 Active |

---

## 🎯 Key Areas

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Deep Learning Portfolio                         │
└────────────────────────────────────────────────────────────────────────┘
          │                   │                     │                  │
          ▼                   ▼                     ▼                  ▼
   Computer Vision     Forensic / GenAI       Enterprise RAG     Industrial AOI
 ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
 │ • YOLOv8 Engine │ │ • ConvNeXt-Tiny │ │ • Groq LPUs     │ │ • PatchCore     │
 │ • Real-Time NMS │ │ • 2D-FFT Spectra│ │ • Llama-3.1-8B  │ │ • WideResNet-50 │
 │ • Video Streams │ │ • Grad-CAM Maps │ │ • FAISS CPU     │ │ • Coreset (1%)  │
 │ • SQLite Logs   │ │ • Dual-Stream   │ │ • Citations     │ │ • Split-Slider  │
 └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 📂 Projects

### 1. 🧠 AI Helmet Detection System

- **Live Demo**: 🚀 [Launch AI Helmet Detection](https://ai-helmet-detection-system.onrender.com/)
- **Description**: An end-to-end computer vision safety compliance system engineered to detect whether motorcycle riders and industrial personnel are wearing safety helmets across static images, uploaded video streams, and live camera feeds.
- **Key Features**:
  - Real-time helmet compliance detection across images, video files (MP4, AVI), and live camera snapshots.
  - Distinct color-coded bounding box overlays (`Class 0: With Helmet` in green, `Class 1: Without Helmet` in red).
  - Interactive analytics dashboard displaying compliance percentages, violation counters, and Plotly visual trends.
  - SQLite database audit logging for every processed detection with search, filtering, and CSV export capabilities.
  - Adjustable detection confidence thresholds and Non-Maximum Suppression (NMS) post-processing.
- **Technologies Used**: Python 3.11, Ultralytics YOLOv8, PyTorch, Torchvision, OpenCV, Streamlit, SQLite3, Pandas, Plotly Express, Gunicorn
- **Model / Architecture**: Custom fine-tuned YOLOv8 nano model (`models/best.pt`) optimized for rapid bounding-box regression and two-class safety classification (`With Helmet`, `Without Helmet`).
- **Repository**: [AI-Helmet-Detection-System](./AI-Helmet-Detection-System/)

---

### 2. 🧠 SPECTRA — AI Digital Forensics / Deepfake Detector

- **Live Demo**: 🚀 [Launch SPECTRA Forensics](https://spectra-forensics.vercel.app/)
- **Description**: A dual-stream digital forensics and deepfake detection platform that combines spatial convolutional features with frequency-domain log-magnitude spectral analysis to identify synthetic imagery and generative manipulations with explainable Grad-CAM heatmaps.
- **Key Features**:
  - Dual-stream neural pipeline integrating spatial feature extraction with Hann-windowed 2D Fast Fourier Transform (FFT) spectral analysis.
  - Explainable AI heatmaps via Stage-4 Grad-CAM activation mapping, pinpointing localized synthetic artifacts.
  - Frequency anomaly detection identifying grid spikes, high-frequency rolloff discrepancies, and latent diffusion smoothing.
  - Production-grade decoupled architecture featuring a FastAPI inference backend and a reactive modern web frontend.
  - Comprehensive REST API endpoints (`/api/v1/analyze`, `/api/v1/health`) returning confidence scores, artifact flags, and base64 visualization overlays.
- **Technologies Used**: Python 3.11, PyTorch, ConvNeXt-Tiny, 2D Fast Fourier Transform (FFT), FastAPI, Uvicorn, React, Vite, Tailwind CSS, NumPy, SciPy, Pillow
- **Model / Architecture**: `DualStreamForensicNet` fusing a 768-D spatial feature vector (pretrained ConvNeXt-Tiny) and a 256-D frequency feature vector (Hann-windowed 2D-FFT + 4-Layer Spectrum CNN) into a 1024-D classification head with Stage-4 Grad-CAM interpretability.
- **Repository**: [DeepfakeAI-Image-Detector](./DeepfakeAI-Image-Detector/)

---

### 3. 🧠 DocuMind — RAG Assistant

- **Live Demo**: 🚀 [Launch DocuMind RAG Assistant](https://documind-rag-assistant.streamlit.app/)
- **Description**: A production-ready Retrieval-Augmented Generation (RAG) assistant designed for instant, grounded document question-answering and multi-mode structured summarization across complex document collections (PDF, TXT, MD) with zero hallucination.
- **Key Features**:
  - High-speed token generation and inference powered by Groq Cloud LPUs and Llama-3.1-8B-Instant (~560 tokens/sec throughput, ~190ms TTFT).
  - Sub-3ms dense vector retrieval utilizing local FAISS vector stores and sentence-transformers.
  - Strict context-bounding prompt engineering with automated fallback guardrails when queries exceed document context.
  - Verifiable source attribution displaying exact retrieved document chunks, page numbers, and cosine similarity metrics.
  - Multi-mode structured summarization (Executive Summary, Technical Key Points, Action Items, Comprehensive Overview).
  - Production containerization via Docker and native Streamlit Community Cloud support.
- **Technologies Used**: Python 3.11, LangChain, Groq Cloud API (Llama-3.1-8B-Instant), FAISS (Facebook AI Similarity Search - CPU), Sentence-Transformers (`all-MiniLM-L6-v2`), PyPDF, Streamlit, Docker
- **Model / Architecture**: Dense vector retrieval pipeline using 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` embeddings, FAISS IndexFlatL2 vector indexing, and Groq-accelerated Llama-3.1-8b-instant generation.
- **Repository**: [docu-mind-rag-assistant](./docu-mind-rag-assistant/)

---

### 4. 🧠 DefectGuard AI

- **Description**: An unsupervised industrial Automated Optical Inspection (AOI) system based on the PatchCore memory-bank framework, trained exclusively on nominal (defect-free) production items to detect novel surface flaws and contamination with sub-100ms inference times.
- **Key Features**:
  - Unsupervised cold-start anomaly detection capable of localizing previously unseen manufacturing flaws without negative training samples.
  - Interactive before/after split-slider interface allowing quality control operators to compare raw captures directly with anomaly heatmaps.
  - High-resolution bicubic defect heatmaps and automated bounding reticles isolating flaw coordinates.
  - 99% memory compression using greedy k-Center coreset subsampling, preserving boundary coverage with minimal RAM utilization.
  - Calibrated sensitivity threshold (baseline 12.0) guaranteeing zero false-positive flags on good manufacturing items.
  - Real-time audiovisual feedback synthesized using the Web Audio API (harmonic chime for pass, alert siren for defect).
- **Technologies Used**: Python 3.11, PyTorch, Torchvision (WideResNet-50-2), FAISS (CPU), OpenCV, SciPy, Scikit-learn, FastAPI, Uvicorn, HTML5, Tailwind CSS, JavaScript, Web Audio API
- **Model / Architecture**: PatchCore memory-bank framework utilizing multi-scale patch embeddings from WideResNet-50-2 (layer2 + layer3, 1536-D), 1% k-Center coreset subsampling, and nearest-neighbor distance scoring.
- **Repository**: [DefectGuard-AI](./DefectGuard-AI/)

---

### 5. 🧠 AI Food Calorie & Nutrition

- **Description**: A computer vision food classification and dietary intelligence system that identifies regional South Asian food dishes from photos and retrieves standard nutritional metrics per serving size.
- **Key Features**:
  - Automated image classification across 6 supported food categories: Biryani, Dosa, Idli, Chapati, Poori, and Sambar.
  - Transfer learning with a fine-tuned EfficientNet-B0 backbone achieving 88.33% validation accuracy.
  - Instant zero-latency local database lookup for Calories, Protein, Carbohydrates, and Fat per standard portion.
  - Responsive drag-and-drop web interface built with Flask, HTML5, and CSS3.
  - Alternate command-line interface (CLI) for rapid terminal-based single-image inference.
  - Zero external API dependencies, operating 100% locally with complete data privacy.
- **Technologies Used**: Python, PyTorch, Torchvision (EfficientNet-B0), OpenCV, Pillow (PIL), Flask, HTML5, CSS3
- **Model / Architecture**: Fine-tuned EfficientNet-B0 with ImageNet transfer learning, global average pooling, dropout (0.2), and 6-class softmax output layer.
- **Repository**: [AI-Food-Calorie-Nutrition](./AI-Food-Calorie-Nutrition/)

---

### 6. 🧠 Deep Learning Web Application

- **Live Demo**: 🚀 [Launch Live Deep Learning App](https://deep-learning-ap09.onrender.com/)
- **Description**: The production cloud gateway and REST API service powering live deep learning model inferences, health probes, and monorepo integrations on Render Cloud.
- **Key Features**:
  - Unified FastAPI / ASGI gateway hosting active neural network checkpoints.
  - Cloud health check probes (`/api/v1/health`) for uptime monitoring and compute device verification.
  - Configured Gunicorn / Uvicorn multi-worker process serving with dynamic CORS origin management.
  - Monorepo cloud deployment configuration via `render.yaml` and `Procfile`.
- **Technologies Used**: Python 3.11, FastAPI, Uvicorn, Gunicorn, PyTorch, Render Cloud
- **Model / Architecture**: Production ASGI cloud gateway hosting deep learning forensic and computer vision services.
- **Repository**: [backend](./backend/) / [app.py](./app.py)

---

## 🛠️ Technologies Used

| Category | Technologies & Tools |
| :--- | :--- |
| **Languages** | Python 3.11+, JavaScript (ES6+), HTML5, CSS3, SQL |
| **Deep Learning & CV** | PyTorch, Torchvision, Ultralytics YOLOv8, ConvNeXt-Tiny, EfficientNet-B0, WideResNet-50-2, 2D Fast Fourier Transform (FFT), OpenCV, Grad-CAM, Pillow |
| **Vector Search & Embeddings** | FAISS (Facebook AI Similarity Search), Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **LLMs & GenAI / RAG** | Groq Cloud LPUs, Llama-3.1-8B-Instant, LangChain |
| **Web & API Frameworks** | FastAPI, Streamlit, Flask, Uvicorn, Gunicorn, React, Vite, Tailwind CSS |
| **Deployment & Cloud** | Render Cloud, Vercel, Streamlit Community Cloud, Docker, Git / GitHub |

---

## 📌 Repository Structure

```text
deep-learning/
├── AI-Food-Calorie-Nutrition/     # EfficientNet-B0 Food Classification & Nutrition Lookup
│   ├── models/                    # Trained model weights (food_classifier_finetuned.pth)
│   ├── src/                       # Model architecture, training, and prediction pipeline
│   ├── templates/ & static/       # Flask web UI interface
│   ├── app.py                     # Flask web server
│   └── README.md
├── AI-Helmet-Detection-System/    # YOLOv8 Real-Time Helmet Compliance Detection
│   ├── models/                    # Fine-tuned YOLOv8 weights (best.pt)
│   ├── utils/                     # Detection and annotation utilities
│   ├── app.py                     # Streamlit multi-page web application
│   └── README.md
├── DeepfakeAI-Image-Detector/     # Dual-Stream Spatial-Frequency Deepfake Detection (SPECTRA)
│   ├── backend/                   # FastAPI inference service & PyTorch models
│   ├── frontend/                  # React + Vite + Tailwind CSS web interface
│   ├── scripts/                   # Model training and evaluation scripts
│   └── README.md
├── DefectGuard-AI/                # PatchCore Industrial Visual Anomaly Detection
│   ├── configs/                   # Anomaly threshold & model configurations
│   ├── models/                    # Memory bank metadata and calibration files
│   ├── src/                       # Feature extraction, memory bank, and coreset sampling
│   ├── index.html                 # Industrial split-screen inspection dashboard
│   ├── api.py                     # FastAPI REST inference service
│   └── README.md
├── docu-mind-rag-assistant/       # Groq-Powered Document Q&A & Summarization (RAG)
│   ├── src/                       # Document loaders, FAISS vector DB, and RAG chains
│   ├── app.py                     # Streamlit interactive UI with source citations
│   ├── Dockerfile                 # Containerized deployment definition
│   └── README.md
├── backend/                       # Monorepo FastAPI entrypoint & shared backend utilities
├── app.py                         # Root ASGI application entrypoint
├── render.yaml                    # Render Cloud deployment blueprint
├── Procfile                       # Production process declaration
├── requirements.txt               # Root environment dependencies
└── README.md                      # Master repository portfolio documentation
```

---

## 👨‍💻 Author & Connect

**Rithikesh S**  
GitHub: [@rithikesh18-rk](https://github.com/rithikesh18-rk)
