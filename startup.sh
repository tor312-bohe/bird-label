#!/usr/bin/env bash
set -euo pipefail
# Azure App Service startup script
# Install dependencies if not already installed (App Service Oryx usually handles this)
if [ -f requirements.txt ]; then
  echo "[Startup] Installing Python dependencies..."
  pip install --no-cache-dir -r requirements.txt
fi
# Run gunicorn with 4 workers (adjust based on SKU) and timeout for long model loads
exec gunicorn --workers=${WORKERS:-2} --timeout=${TIMEOUT:-180} --bind=0.0.0.0:${PORT:-8000} wsgi:app
