"""Root-level FastAPI Application Entrypoint for Monorepo Hosting.
Re-exports the production FastAPI application from DeepfakeAI-Image-Detector/backend/app.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
candidates = [
    os.path.join(current_dir, "DeepfakeAI-Image-Detector", "backend"),
    os.path.join(current_dir, "DeepfakeAI-Image Detector", "backend"),
    os.path.join(current_dir, "backend"),
]

for c in candidates:
    if os.path.isdir(c) and c not in sys.path:
        sys.path.insert(0, c)

try:
    from backend.app import app
except ImportError:
    from app import app

try:
    from a2wsgi import ASGIMiddleware
    application = ASGIMiddleware(app)
except Exception:
    application = app

__all__ = ["app", "application"]
