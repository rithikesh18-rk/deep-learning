"""FastAPI Inference Service for Deepfake & Synthetic AI-Image Detection.

Root backend module that routes to the FastAPI application from
DeepfakeAI-Image-Detector/backend/app.py.
Supports running directly via:
uvicorn app:app --app-dir backend --host 0.0.0.0 --port $PORT
from either repository root or DeepfakeAI-Image-Detector directory.
"""

import os
import sys
import importlib.util
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent
_target_app_path = (_repo_root / "DeepfakeAI-Image-Detector" / "backend" / "app.py").resolve()

# Ensure the backend directory is in sys.path for internal imports (model, frequency_utils)
if _target_app_path.parent.is_dir() and str(_target_app_path.parent) not in sys.path:
    sys.path.insert(0, str(_target_app_path.parent))

# Load module dynamically
spec = importlib.util.spec_from_file_location("deepfake_detector_backend", str(_target_app_path))
_mod = importlib.util.module_from_spec(spec)
sys.modules["deepfake_detector_backend"] = _mod
spec.loader.exec_module(_mod)

# Expose FastAPI application instance and key symbols
app = _mod.app
health_check = getattr(_mod, "health_check", None)
analyze_image = getattr(_mod, "analyze_image", None)
forensic_model = getattr(_mod, "forensic_model", None)
device = getattr(_mod, "device", None)
checkpoint_loaded = getattr(_mod, "checkpoint_loaded", False)
loaded_checkpoint_path = getattr(_mod, "loaded_checkpoint_path", None)

__all__ = ["app"]
