#!/bin/bash

# Azure App Service startup script for Flask application
echo "🚀 Starting Bird Label App on Azure..."

# Set environment variables for production
export FLASK_ENV=production
export PYTHONPATH=/home/site/wwwroot

# Create necessary directories
mkdir -p /home/site/wwwroot/uploads
mkdir -p /home/site/wwwroot/outputs
mkdir -p /home/site/wwwroot/data
mkdir -p /home/site/wwwroot/fonts
mkdir -p /home/site/wwwroot/models

# Initialize database if it doesn't exist
cd /home/site/wwwroot
python -c "
from app import init_database
try:
    init_database()
    print('✅ Database initialized successfully')
except Exception as e:
    print(f'⚠️ Database initialization warning: {e}')
"

# Start the application with Gunicorn
echo "🌐 Starting Gunicorn server..."
gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --keep-alive 2 --max-requests 1000 app:app
