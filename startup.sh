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

# Start the application with Gunicorn
echo "🌐 Starting Gunicorn server..."
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --keep-alive 2 --max-requests 1000 wsgi:app
