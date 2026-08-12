# 🪖 AI Helmet Detection System

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, real-time **AI Helmet Detection System** powered by **YOLOv8 Deep Learning**, **Streamlit**, and **SQLite**. Designed for high-risk industrial plants, construction sites, and smart traffic surveillance networks to enforce safety compliance automatically.

---

## 🌐 Live Demo

🚀 **Experience the Live Application**: [https://ai-helmet-detection-system.onrender.com/](https://ai-helmet-detection-system.onrender.com/)

---

## 🎯 Problem Being Solved

Industrial accidents and road traffic fatalities frequently result from failure to wear head protective equipment (safety helmets). Manual monitoring is labor-intensive, error-prone, and unscalable across large facility perimeter networks or multi-lane traffic intersections. 

This project automates safety compliance by deploying a real-time object detection pipeline that identifies individuals wearing helmets versus non-compliant workers/riders, logs violation events into an audit database, and generates visual analytics to help safety officers intervene rapidly.

---

## 🧠 How the Deep Learning Model Works

1. **Architecture:** Powered by **YOLOv8** (Ultralytics), a state-of-the-art single-stage convolutional object detection network optimized for high inference speed and precise spatial localization.
2. **Preprocessing & Ingestion:** Input images, video frames, or live camera snapshots are captured via OpenCV, resized, normalized, and converted into tensor representations.
3. **Feature Extraction & Bounding Box Prediction:** The YOLOv8 backbone extracts multi-scale feature maps, predicting bounding box coordinates, objectness scores, and class probability distributions simultaneously.
4. **Target Classes:**
   - **`Class 0: With Helmet`** – Rendered with a **Green Bounding Box** (`#00FF00`) indicating compliance.
   - **`Class 1: Without Helmet`** – Rendered with a **Red Bounding Box** (`#0000FF`) indicating a safety violation.
5. **Post-Processing & Non-Maximum Suppression (NMS):** Filters duplicate bounding boxes using confidence thresholds and Non-Maximum Suppression to produce clean visual output overlays and log detection metrics.

---

## 🏗️ System Architecture

```text
                               ┌─────────────────────────┐
                               │  Streamlit Frontend UI  │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │  YOLOv8 Engine Detector │
                               │     (models/best.pt)    │
                               └────────────┬────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   Annotated Output Frame  │                 │    SQLite History Logs    │
       │     (uploads/ & results/) │                 │   (database/detection.db) │
       └───────────────────────────┘                 └───────────────────────────┘
```

---

## 🛠️ Technology Stack

- **Programming Language:** Python 3.11.9
- **Computer Vision & AI:** YOLOv8 (Ultralytics), OpenCV (`opencv-python-headless`), PyTorch, torchvision, NumPy, Pillow
- **Frontend Framework:** Streamlit
- **Database:** SQLite3
- **Data Analytics & Charts:** Pandas, Plotly Express
- **Deployment Platform:** Render (`render.yaml`, `Procfile`, `runtime.txt`), Gunicorn

---

## 🌟 Key Features

- 🖼️ **Image Helmet Detection**: Detect helmet compliance on uploaded static images (JPG, JPEG, PNG) with confidence scores and custom bounding box visual overlays.
- 🎥 **Video Surveillance Analytics**: Process video feeds (MP4, AVI) frame-by-frame and export annotated output videos.
- 📹 **Live Webcam Feed**: Instant real-time helmet checking via live camera snapshot feeds.
- 📊 **Real-Time Analytics Dashboard**: Visual metrics tracking compliance rates, violation counts, and historical trends with interactive Plotly charts.
- 💾 **SQLite Audit Logging**: Permanent database storage of all processed detections with search, filter, and CSV export capabilities.
- 🎨 **Modern Dark AI UI**: Glassmorphism dashboard aesthetics, status badges, and responsive multi-page sidebar navigation.
- 🚀 **Cloud Production Ready**: Fully configured for 1-click deployment on **Render** and **GitHub**.

---

## 🏷️ Model Classes

The YOLOv8 detection engine is trained/mapped on the following target classes:
- **`Class 0: With Helmet`** - Green Bounding Box (`#00FF00`) - Person wearing safety helmet.
- **`Class 1: Without Helmet`** - Red Bounding Box (`#0000FF`) - Safety violation detected.

---

## 📂 Project Structure

```text
AI-Helmet-Detection-System/
├── database/
│   └── .gitkeep             # SQLite database directory placeholder
├── models/
│   └── best.pt              # Fine-tuned YOLOv8 helmet detection model weights
├── results/
│   └── .gitkeep             # Directory placeholder for output detections
├── test/                    # Test dataset split (images & labels)
├── train/                   # Training dataset split (images & labels)
├── uploads/
│   └── .gitkeep             # Directory placeholder for user uploaded files
├── utils/
│   ├── detector.py          # Helper detector utilities
│   └── __init__.py
├── valid/                   # Validation dataset split (images & labels)
├── .python-version          # Python version marker (3.11.9)
├── app.py                   # Main Streamlit web application & multi-page navigation
├── config.py                # Application configurations, paths & theme variables
├── create_model.py          # Helper script to initialize YOLOv8 weights
├── data.yaml                # Dataset configuration & class definitions
├── database.py              # SQLite database management & analytics queries
├── detector.py              # YOLOv8 inferencing engine & frame annotation
├── fix_model_names.py       # Helper script for model class name mapping
├── Procfile                 # Render process configuration
├── README.dataset.txt       # Dataset metadata from Roboflow
├── README.md                # Project documentation & execution guide
├── README.roboflow.txt      # Roboflow export specification
├── real_model_names.txt     # Target class list
├── render.yaml              # Render deployment specification
├── requirements.txt         # Python dependency packages
├── runtime.txt              # Python 3.11.9 environment specification
├── test_accuracy.py         # Model accuracy evaluation script
├── test_model.py            # Unit test script for detection pipeline
├── test_suite.py            # Automated test suite
├── verify_model.py          # Model verification script
└── yolov8n.pt               # Base pre-trained YOLOv8 nano weights
```

---

## 🚀 Installation & Local Execution

### 1. Clone / Prepare Repository
```bash
git clone https://github.com/rithikesh18-rk/deep-learning.git
cd deep-learning/AI-Helmet-Detection-System
```

### 2. Create & Activate Virtual Environment
```bash
# On Windows
py -3.11 -m venv venv
venv\Scripts\activate

# On Linux / macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Model Weights
```bash
python create_model.py
```

### 5. Run Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🔗 Original GitHub Repository

- **Original Repository URL:** [https://github.com/rithikesh18-rk/AI-Helmet-Detection-System.git](https://github.com/rithikesh18-rk/AI-Helmet-Detection-System.git)

---

## 👤 Author

Developed by **[rithikesh18-rk](https://github.com/rithikesh18-rk)** (Rithikesh S)
