#!/usr/bin/env bash
# Azure Web App verification and creation script
set -euo pipefail

APP_NAME=${APP_NAME:-bird-label-app1}
RG=${RG:-bird-label-rg}
PLAN=${PLAN:-bird-label-plan}
LOCATION=${LOCATION:-eastasia}

echo "🔍 Checking if Azure resources exist..."

# Check if logged in
if ! az account show >/dev/null 2>&1; then
    echo "❌ Not logged into Azure. Run: az login"
    exit 1
fi

# Check resource group
if ! az group show -n "$RG" >/dev/null 2>&1; then
    echo "❌ Resource group $RG does not exist. Creating..."
    az group create -n "$RG" -l "$LOCATION"
else
    echo "✅ Resource group $RG exists"
fi

# Check app service plan
if ! az appservice plan show -g "$RG" -n "$PLAN" >/dev/null 2>&1; then
    echo "❌ App service plan $PLAN does not exist. Creating..."
    az appservice plan create -g "$RG" -n "$PLAN" --sku B1 --is-linux
else
    echo "✅ App service plan $PLAN exists"
fi

# Check web app
if ! az webapp show -g "$RG" -n "$APP_NAME" >/dev/null 2>&1; then
    echo "❌ Web app $APP_NAME does not exist. Creating with PYTHON runtime..."
    az webapp create -g "$RG" -p "$PLAN" -n "$APP_NAME" --runtime "PYTHON:3.10"
    echo "✅ Created web app $APP_NAME"
else
    echo "✅ Web app $APP_NAME exists"
    # Show current state
    STATE=$(az webapp show -g "$RG" -n "$APP_NAME" --query state -o tsv)
    HOST=$(az webapp show -g "$RG" -n "$APP_NAME" --query defaultHostName -o tsv)
    echo "   State: $STATE"
    echo "   URL: https://$HOST"
fi

# Test if hostname resolves
echo "🌐 Testing DNS resolution for $APP_NAME.azurewebsites.net..."
if nslookup "$APP_NAME.azurewebsites.net" >/dev/null 2>&1; then
    echo "✅ DNS resolves successfully"
    # Test HTTP
    if curl -Is "https://$APP_NAME.azurewebsites.net" | head -1 | grep -q "200\|404\|502\|503"; then
        echo "✅ Web app responds to HTTP requests"
    else
        echo "⚠️ Web app not responding or returning error"
    fi
else
    echo "❌ DNS does not resolve - app may not be properly created"
fi

echo ""
echo "🔧 Next steps if app exists but doesn't work:"
echo "1. Deploy code: ./deploy_azure.sh"
echo "2. Check logs: az webapp log tail -g $RG -n $APP_NAME"
echo "3. Test health: curl https://$APP_NAME.azurewebsites.net/healthz"
