"""FastAPI Application Entrypoint for Render and Production Hosting.
Re-exports the production FastAPI application from backend.app.
Compatible with Uvicorn (ASGI) and Gunicorn (via UvicornWorker or a2wsgi).
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "backend")
if os.path.isdir(backend_dir) and backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

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
