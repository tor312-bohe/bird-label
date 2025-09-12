#!/usr/bin/env bash
# Deploy Bird Label app to Azure App Service
# Usage: ./deploy_azure.sh
set -euo pipefail
APP_NAME=${APP_NAME:-bird-label-app1}
RG=${RG:-bird-label-rg}
PLAN=${PLAN:-bird-label-plan}
LOCATION=${LOCATION:-eastasia}
PYTHON_VERSION=${PYTHON_VERSION:-PYTHON:3.10}

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI (az) not found. Install: https://learn.microsoft.com/cli/azure/install-azure-cli" >&2
  exit 1
fi

echo "Logging into Azure (interactive)..."
az login --only-show-errors

echo "Ensuring resource group $RG"
az group create -n "$RG" -l "$LOCATION" --only-show-errors

echo "Ensuring plan $PLAN"
az appservice plan create -g "$RG" -n "$PLAN" --sku B1 --is-linux --only-show-errors || true

echo "Creating webapp $APP_NAME (or skipping if exists)"
if ! az webapp show -g "$RG" -n "$APP_NAME" >/dev/null 2>&1; then
  az webapp create -g "$RG" -p "$PLAN" -n "$APP_NAME" --runtime "$PYTHON_VERSION" --only-show-errors
fi

if [ -z "${HHOLOVE_API_KEY:-}" ]; then
  echo "Warning: HHOLOVE_API_KEY env var not set in shell; you'll need to add it manually later." >&2
else
  echo "Setting app settings (HHOLOVE_API_KEY, WEBSITES_PORT, PYTHONUNBUFFERED)"
  az webapp config appsettings set -g "$RG" -n "$APP_NAME" --settings HHOLOVE_API_KEY="$HHOLOVE_API_KEY" WEBSITES_PORT=8000 PYTHONUNBUFFERED=1
fi

echo "Configuring startup command"
az webapp config set -g "$RG" -n "$APP_NAME" --startup-file "./startup.sh" --only-show-errors

echo "Creating deployment zip"
ZIP_FILE=deploy.zip
rm -f "$ZIP_FILE"
zip -rq "$ZIP_FILE" . -x '*.git*' '*.pyc' '__pycache__/*'

echo "Deploying..."
az webapp deployment source config-zip -g "$RG" -n "$APP_NAME" --src "$ZIP_FILE" --only-show-errors

echo "Enabling log streaming (filesystem logs)"
az webapp log config -g "$RG" -n "$APP_NAME" --application-logging filesystem --level info --only-show-errors

echo "Deployment complete. URL: https://$APP_NAME.azurewebsites.net"
echo "Health check: https://$APP_NAME.azurewebsites.net/healthz"
echo "Tail logs: az webapp log tail -g $RG -n $APP_NAME"
