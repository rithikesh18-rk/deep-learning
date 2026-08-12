import os

# Ensure YOLO_CONFIG_DIR is set for cloud environments like Render
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

# Base Directories using absolute path derived from __file__
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DB_DIR = os.path.join(BASE_DIR, "database")

# Ensure required directories exist
for directory in [MODEL_DIR, UPLOAD_DIR, RESULT_DIR, DB_DIR]:
    os.makedirs(directory, exist_ok=True)

# File Paths
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
DB_PATH = os.path.join(DB_DIR, "detection.db")

# Allowed Upload Extensions
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# YOLO Model Class Configuration
# 0 = With Helmet, 1 = Without Helmet
CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet"
}

# BGR Colors: 0 = Emerald Green, 1 = Vibrant Red
CLASS_COLORS = {
    0: (0, 225, 100),   # BGR Green for With Helmet
    1: (40, 40, 255)    # BGR Red for Without Helmet
}

# Detection Parameters
DEFAULT_CONFIDENCE = 0.50
DEFAULT_IOU = 0.45
TARGET_IMAGE_SIZE = 640

# App Metadata
APP_TITLE = "AI Helmet Detection System"
APP_SUBTITLE = "Real-Time AI Powered Safety & Helmet Compliance Monitoring System"
AUTHOR = "rithikesh18-rk"
VERSION = "1.0.0"
