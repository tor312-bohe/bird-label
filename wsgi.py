#!/usr/bin/env python3
"""
WSGI entry point for Azure App Service deployment
This file is used by Azure App Service to start the Flask application
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask application
from app import app

# This is what Azure App Service will use to start the app
if __name__ == "__main__":
    # For Azure App Service, use the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
