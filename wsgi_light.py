# Lightweight WSGI entrypoint avoiding heavy optional imports.
# Use by setting STARTUP_MODULE=wsgi_light (and adjusting startup command) if memory issues.
from app import app as application, init_database

try:
    init_database()
except Exception as e:
    print(f"[wsgi_light] DB init warning: {e}")

app = application
