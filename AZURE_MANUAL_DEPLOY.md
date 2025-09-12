# Azure Manual Deployment Guide

## ✅ READY: Deployment Package Created
**File**: `bird-label-azure-deploy.zip` (33KB)
**Contains**: app.py, wsgi_light.py, minimal dependencies, database, resources

## Method 1: Azure App Service (Recommended for Flask apps)

### Step 1: Create Azure App Service
1. Go to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource" → "Web App"
3. Fill in the details:
   - **Subscription**: Your subscription
   - **Resource Group**: Create new "bird-label-rg" 
   - **Name**: `bird-label-app1` (try variations if taken: bird-label-app2, etc.)
   - **Publish**: Code
   - **Runtime stack**: Python 3.10
   - **Operating System**: Linux
   - **Region**: East US (or closest to you)
   - **Pricing plan**: Free F1 (for testing)
4. Click "Review + create" → "Create"
5. **Wait for deployment to complete** (this is important!)

### Step 2: Upload Deployment Package (ZIP Deploy)
**🎯 THIS IS THE KEY STEP THAT OFTEN GETS MISSED**

1. **Wait for App Service creation to FULLY complete**
2. In your App Service page, look for "Advanced Tools" in the left menu
3. Click "Advanced Tools" → "Go" (this opens Kudu)
4. In Kudu, click "Tools" → "Zip Push Deploy"  
5. **Drag and drop** the file `bird-label-azure-deploy.zip`
6. Wait for the upload and extraction to complete
7. You should see green checkmarks indicating success

### Step 3: Configure Environment Variables
1. Back in Azure Portal, go to your App Service
2. Click "Configuration" in the left menu
3. Click "Application settings" tab
4. Click "New application setting" and add:
   - **Name**: `HHOLOVE_API_KEY`
   - **Value**: Your actual API key (from .env file)
5. Add another setting:
   - **Name**: `MINIMAL_DEPS` 
   - **Value**: `1`
6. Click "Save" at the top

### Step 4: Set Startup Command  
1. Still in "Configuration", click "General settings" tab
2. In **Startup Command** field, enter: `/bin/bash startup.sh`
3. Click "Save" at the top
4. **Restart** the App Service (Overview page → Restart button)

### Step 5: Test Your Deployment
1. Go to "Overview" page of your App Service
2. Click the **URL** (should be `https://bird-label-app1.azurewebsites.net`)
3. First test the health endpoint: add `/healthz` to the URL
4. If health check works, try the main app

## ⚠️ TROUBLESHOOTING - Previous Issues We Had

### Issue 1: DNS_PROBE_FINISHED_NXDOMAIN
**Cause**: App Service was never properly created or took too long
**Solution**: 
- Wait 5-10 minutes after creation says "complete"
- Try a different app name if the URL doesn't resolve
- Check if the resource actually exists in Resource Groups

### Issue 2: App Won't Start
**Check these in order**:
1. **Logs**: App Service → "Log stream" to see errors
2. **Startup Command**: Must be exactly `/bin/bash startup.sh` 
3. **Environment Variables**: HHOLOVE_API_KEY must be set
4. **Deployment**: ZIP file must be uploaded via Kudu

### Issue 3: 502 Bad Gateway
**Most common causes**:
- Startup command incorrect
- Port binding issue (app should bind to 0.0.0.0:8000)
- Dependencies failed to install

## Alternative: Try a Different Region
If `bird-label-app1` keeps failing, try:
- West US 2
- West Europe  
- Southeast Asia

Sometimes specific regions have temporary issues.

## Method 2: Azure Container Instances (Alternative)

### Prerequisites
1. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install Azure CLI (after Homebrew is set up)

### Steps
1. Build container locally:
   ```bash
   cd "/Users/binbin_new/Personal projects/Bird Label"
   docker build -t bird-label-app .
   ```

2. Push to Azure Container Registry:
   ```bash
   # Login to Azure
   az login
   
   # Create container registry
   az acr create --resource-group bird-label-rg --name birdlabelregistry --sku Basic
   
   # Push image
   az acr build --registry birdlabelregistry --image bird-label-app:latest .
   ```

3. Deploy to Container Instances:
   ```bash
   az container create \
     --resource-group bird-label-rg \
     --name bird-label-app \
     --image birdlabelregistry.azurecr.io/bird-label-app:latest \
     --cpu 1 --memory 1 \
     --ports 8000 \
     --environment-variables HHOLOVE_API_KEY="your-api-key-here"
   ```

## Troubleshooting

### If deployment fails:
1. Check logs in Azure Portal → App Service → "Log stream"
2. Verify `requirements-minimal.txt` has all needed packages
3. Ensure startup command is correct: `/bin/bash startup.sh`

### If app doesn't start:
1. Check if port 8000 is configured correctly
2. Verify environment variable `HHOLOVE_API_KEY` is set
3. Check that database file `data/species.sqlite` exists

### Common issues:
- **502 Bad Gateway**: Usually startup command or port configuration
- **500 Internal Server Error**: Missing environment variables or dependencies
- **404 Not Found**: App not deployed correctly

## Testing
Once deployed, your app will be available at:
`https://bird-label-app1.azurewebsites.net`

Test the health endpoint first:
`https://bird-label-app1.azurewebsites.net/healthz`
