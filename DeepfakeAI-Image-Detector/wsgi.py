"""WSGI Application Entrypoint for Production Web Servers."""
from app import application, app

__all__ = ["application", "app"]
