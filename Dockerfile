# Lightweight production Docker image for Bird Label app
# Build: docker build -t bird-label:latest .
# Run:  docker run -p 8000:8000 -e PORT=8000 bird-label:latest

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    MINIMAL_DEPS=1

WORKDIR /app

# System deps (fonts, build essentials for pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements first for caching
COPY requirements-minimal.txt ./
RUN pip install -r requirements-minimal.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
 CMD curl -f http://127.0.0.1:8000/healthz || exit 1

# Start gunicorn
CMD ["gunicorn", "--workers=2", "--threads=2", "--timeout=180", "--bind=0.0.0.0:8000", "wsgi_light:app"]
