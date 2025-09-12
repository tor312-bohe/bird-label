#!/usr/bin/env bash
set -euo pipefail
# Azure App Service startup script
# Install dependencies if not already installed (App Service Oryx usually handles this)
if [ "${MINIMAL_DEPS:-0}" = "1" ] && [ -f requirements-minimal.txt ]; then
  echo "[Startup] Installing minimal dependencies (requirements-minimal.txt)..."
  pip install --no-cache-dir -r requirements-minimal.txt
elif [ -f requirements.txt ]; then
  echo "[Startup] Installing full dependencies (requirements.txt)..."
  pip install --no-cache-dir -r requirements.txt || echo "[Startup] Full install failed; consider setting MINIMAL_DEPS=1"
fi
# Run gunicorn with 4 workers (adjust based on SKU) and timeout for long model loads
echo "[Startup] Launching gunicorn..."
exec gunicorn --workers=${WORKERS:-2} --threads=${THREADS:-2} --timeout=${TIMEOUT:-180} --bind=0.0.0.0:${PORT:-8000} wsgi:app
