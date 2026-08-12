# Deep Learning Projects

This repository contains my **Deep Learning** subject projects.

---

## Projects

### 1. AI Helmet Detection System

- **Description:** An enterprise-grade, real-time computer vision system built with **YOLOv8 Deep Learning**, **PyTorch**, **OpenCV**, **Streamlit**, and **SQLite3**. Designed for high-risk industrial facilities, construction sites, and smart traffic surveillance networks to automatically detect and enforce safety helmet compliance.
- **Technologies Used:**
  - **Deep Learning & Computer Vision:** PyTorch, torchvision, Ultralytics YOLOv8 (`models/best.pt`, `yolov8n.pt`), OpenCV (`opencv-python-headless`), NumPy, Pillow
  - **Web Application Framework:** Streamlit (`app.py`)
  - **Data Analysis & Charts:** Pandas, Plotly Express
  - **Database:** SQLite3 (`database.py`)
  - **Cloud Deployment:** Render (`render.yaml`, `Procfile`, `runtime.txt`), Gunicorn
  - **Language & Runtime:** Python 3.11.9
- **Project Directory:**
  [`AI-Helmet-Detection-System/`](./AI-Helmet-Detection-System/)
- **Original GitHub Repository:**
  https://github.com/rithikesh18-rk/AI-Helmet-Detection-System.git

---

## Repository Structure

```text
deep-learning/
├── AI-Helmet-Detection-System/
│   ├── database/
│   │   └── .gitkeep             # SQLite database directory placeholder
│   ├── models/
│   │   └── best.pt              # Fine-tuned YOLOv8 helmet detection model weights
│   ├── results/
│   │   └── .gitkeep             # Output directory placeholder
│   ├── test/                    # Test dataset split (images & labels - 128 items)
│   ├── train/                   # Training dataset split (images & labels - 2370 items)
│   ├── uploads/
│   │   └── .gitkeep             # User uploads directory placeholder
│   ├── utils/
│   │   ├── detector.py          # Detection pipeline utilities
│   │   └── __init__.py
│   ├── valid/                   # Validation dataset split (images & labels - 254 items)
│   ├── .python-version          # Python version marker (3.11.9)
│   ├── app.py                   # Main Streamlit web application & UI navigation
│   ├── config.py                # Application configurations & path definitions
│   ├── create_model.py          # Script to download/verify YOLOv8 base model
│   ├── data.yaml                # Dataset configuration & class definitions
│   ├── database.py              # SQLite database management & analytics queries
│   ├── detector.py              # Core YOLOv8 inference engine & annotation pipeline
│   ├── fix_model_names.py       # Helper script for model class name mapping
│   ├── Procfile                 # Process deployment definition for Render
│   ├── README.dataset.txt       # Roboflow dataset information
│   ├── README.roboflow.txt      # Roboflow export reference
│   ├── real_model_names.txt     # Target class list
│   ├── render.yaml              # Render deployment configuration
│   ├── requirements.txt         # Dependency package requirements
│   ├── runtime.txt              # Python 3.11.9 runtime specification
│   ├── test_accuracy.py         # Model accuracy evaluation script
│   ├── test_model.py            # Unit test script for detection pipeline
│   ├── test_suite.py            # Automated test suite
│   ├── verify_model.py          # Model verification script
│   └── yolov8n.pt               # YOLOv8 nano pre-trained base model weights
├── .gitignore
└── README.md
```

---

## Author

**Rithikesh S** ([rithikesh18-rk](https://github.com/rithikesh18-rk))
