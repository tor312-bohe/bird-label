#!/bin/bash
# Create Azure deployment package

set -e
cd "/Users/binbin_new/Personal projects/Bird Label"

echo "🗂️ Creating Azure deployment package..."

# Create temporary directory
TEMP_DIR="azure_deploy_package"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Copy essential files
echo "📦 Copying application files..."
cp app.py "$TEMP_DIR/"
cp wsgi_light.py "$TEMP_DIR/"
cp requirements-minimal.txt "$TEMP_DIR/"
cp startup.sh "$TEMP_DIR/"
cp .env.example "$TEMP_DIR/"

# Copy database and resources
echo "📦 Copying database and resources..."
cp -r data/ "$TEMP_DIR/"
cp -r Resources/ "$TEMP_DIR/"

# Create the ZIP file
echo "🗜️ Creating ZIP archive..."
cd "$TEMP_DIR"
zip -r ../bird-label-azure-deploy.zip .
cd ..

# Clean up
rm -rf "$TEMP_DIR"

echo "✅ Deployment package created: bird-label-azure-deploy.zip"
echo "📊 Package size: $(ls -lh bird-label-azure-deploy.zip | awk '{print $5}')"
echo ""
echo "🚀 Next steps:"
echo "1. Go to Azure Portal: https://portal.azure.com"
echo "2. Create new Web App (Python 3.10, Linux)"
echo "3. Use Kudu to upload this ZIP file"
echo "4. Set environment variable HHOLOVE_API_KEY"
echo "5. Set startup command: /bin/bash startup.sh"
