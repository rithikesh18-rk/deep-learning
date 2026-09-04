import os

port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 180
keepalive = 5
preload_app = False
