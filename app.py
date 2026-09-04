"""Universal Application Entrypoint for Render and Production Hosting.
Compatible with Uvicorn (ASGI), Gunicorn ASGI (UvicornWorker), and Gunicorn WSGI (via a2wsgi).
Automatically discovers the DeepfakeAI-Image-Detector backend.
"""
import os
import sys
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
candidates = [
    os.path.join(current_dir, "backend"),
    os.path.join(current_dir, "DeepfakeAI-Image-Detector", "backend"),
    os.path.join(current_dir, "DeepfakeAI-Image Detector", "backend"),
]

backend_app_path = None
for b_dir in candidates:
    if os.path.isdir(b_dir) and b_dir not in sys.path:
        sys.path.insert(0, b_dir)
    cp = os.path.join(b_dir, "app.py")
    if os.path.isfile(cp):
        backend_app_path = cp
        break

if not backend_app_path:
    raise ImportError(f"Cannot locate backend/app.py in candidate directories: {candidates}")

spec = importlib.util.spec_from_file_location("spectra_backend_module", backend_app_path)
spectra_backend = importlib.util.module_from_spec(spec)
sys.modules["spectra_backend_module"] = spectra_backend
spec.loader.exec_module(spectra_backend)

fastapi_app = spectra_backend.app

try:
    from a2wsgi import ASGIMiddleware
    wsgi_app = ASGIMiddleware(fastapi_app)
except Exception:
    wsgi_app = None

class UniversalApp:
    def __init__(self, asgi_app, wsgi_wrapper):
        self.asgi_app = asgi_app
        self.wsgi_wrapper = wsgi_wrapper

    def __call__(self, *args, **kwargs):
        if len(args) == 2 and self.wsgi_wrapper is not None:
            return self.wsgi_wrapper(*args, **kwargs)
        return self.asgi_app(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.asgi_app, name)

app = UniversalApp(fastapi_app, wsgi_app)
application = app
