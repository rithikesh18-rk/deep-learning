"""
Configuration settings and hyperparameters for AI Food Calorie & Nutrition Estimation.
"""

from pathlib import Path
import torch

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "food_classifier_finetuned.pth" if (MODELS_DIR / "food_classifier_finetuned.pth").exists() else MODELS_DIR / "food_classifier.pth"

# 6 Target Food Classes
FOOD_CLASSES = [
    "dosa", "idli", "biryani", "chapati", "poori", "sambar"
]


# Model Architecture
# Choices: "efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large"
MODEL_NAME = "efficientnet_b0"

# Data Preprocessing & Augmentation
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 0 for Windows multiprocessing stability
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Computation Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
