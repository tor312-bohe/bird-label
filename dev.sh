#!/bin/bash
# Development helper script for Bird Label iterations

echo "🐦 Bird Label Development Helper"
echo "================================"

# Function to start local development
dev() {
    echo "🔧 Starting local development server..."
    python3 app.py
}

# Function to deploy changes
deploy() {
    echo "🚀 Deploying to Azure..."
    git add .
    read -p "📝 Commit message: " msg
    git commit -m "$msg"
    git push origin main
    echo "✅ Deployed! Check Azure in 2-3 minutes"
}

# Function to test locally first
test() {
    echo "🧪 Testing application..."
    python3 -c "from app import app; print('✅ App imports successfully')"
}

# Show usage
usage() {
    echo "Usage:"
    echo "  ./dev.sh dev     - Start local development"
    echo "  ./dev.sh test    - Test app imports"  
    echo "  ./dev.sh deploy  - Deploy to Azure"
}

# Handle commands
case "$1" in
    dev)    dev ;;
    deploy) deploy ;;
    test)   test ;;
    *)      usage ;;
esac
