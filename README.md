# Deep Learning Projects

This repository contains my **Deep Learning** subject project: **AI Helmet Detection System**.

---

# 🪖 AI Helmet Detection System

A real-time, computer-vision-powered **AI Helmet Detection System** developed using **YOLOv8 Deep Learning**, **PyTorch**, **OpenCV**, **Streamlit**, and **SQLite3**. Designed for high-risk industrial facilities, construction sites, and smart traffic surveillance networks to enforce safety helmet compliance automatically.

---

## 🌐 Live Demo

🚀 **Live Website:** [https://ai-helmet-detection-system.onrender.com/](https://ai-helmet-detection-system.onrender.com/)

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

## 🛠️ Technologies Actually Used

- **Programming Language:** Python 3.11.9
- **Deep Learning Framework:** PyTorch, torchvision
- **Object Detection Engine:** Ultralytics YOLOv8 (`models/best.pt`, `yolov8n.pt`)
- **Computer Vision:** OpenCV (`opencv-python-headless`), Pillow
- **Web Application Framework:** Streamlit
- **Data Analysis & Charts:** Pandas, Plotly Express
- **Database:** SQLite3
- **Cloud Deployment:** Render (`render.yaml`, `Procfile`, `runtime.txt`), Gunicorn

---

## 🌟 Main Features

- 🖼️ **Image Detection:** Upload static images (JPG, JPEG, PNG) to detect helmet compliance with confidence scores and custom bounding box overlays.
- 🎥 **Video Surveillance Analytics:** Process recorded video feeds (MP4, AVI) frame-by-frame and export annotated output videos.
- 📹 **Live Webcam Feed:** Instant real-time helmet detection using live camera snapshots.
- 📊 **Analytics Dashboard:** Real-time compliance metrics, total detection logs, violation counts, and interactive Plotly visualization charts.
- 💾 **SQLite Audit Logging:** Database tracking of all processed detections with search, filter, and CSV export capabilities.
- 🎨 **Modern Dark AI UI:** Glassmorphism dashboard styling with multi-page navigation.

---

## 🚀 Installation & Local Execution

### 1. Clone / Prepare Repository
```bash
git clone https://github.com/rithikesh18-rk/deep-learning.git
cd deep-learning
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Linux / macOS
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
Open your web browser and navigate to `http://localhost:8501`.

---

## 📁 Repository Structure

```text
deep-learning/
├── database/
│   └── .gitkeep             # SQLite database directory placeholder
├── models/
│   └── best.pt              # Trained YOLOv8 helmet detection model weights
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
├── .gitignore               # Git exclusion rules
├── .python-version          # Python version marker (3.11.9)
├── app.py                   # Main Streamlit web application & UI navigation
├── config.py                # Application configurations & path definitions
├── create_model.py          # Script to download/verify YOLOv8 base model
├── data.yaml                # Dataset configuration & class definitions
├── database.py              # SQLite database management & analytics queries
├── detector.py              # Core YOLOv8 inference engine & annotation pipeline
├── fix_model_names.py       # Helper script for model class name mapping
├── Procfile                 # Process deployment definition for Render
├── README.dataset.txt       # Roboflow dataset information
├── README.md                # Project documentation
├── README.roboflow.txt      # Roboflow export reference
├── real_model_names.txt     # Target class list
├── render.yaml              # Render deployment configuration
├── requirements.txt         # Dependency package requirements
├── runtime.txt              # Python 3.11.9 runtime specification
├── test_accuracy.py         # Model accuracy evaluation script
├── test_model.py            # Unit test script for detection pipeline
├── test_suite.py            # Automated test suite
├── verify_model.py          # Model verification script
└── yolov8n.pt               # YOLOv8 nano pre-trained base model weights
```

---

## 🔗 Original GitHub Repository

- **Original Repository URL:** [https://github.com/rithikesh18-rk/AI-Helmet-Detection-System.git](https://github.com/rithikesh18-rk/AI-Helmet-Detection-System.git)

---

## 👤 Author

**Rithikesh S** ([rithikesh18-rk](https://github.com/rithikesh18-rk))
