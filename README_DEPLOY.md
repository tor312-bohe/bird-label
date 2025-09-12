# Bird Label Azure Deployment Guide

## Overview
Two deployment modes:
1. Container (recommended) – deterministic, uses provided `Dockerfile`.
2. App Service zip (legacy) – heavier dependency issues.

We provide a GitHub Actions workflow `.github/workflows/deploy-azure.yml` to build & push the container to Azure Container Registry (ACR) then update the Web App.

## Prerequisites (One-time in Azure)
1. Create Resource Group:
   az group create -n bird-label-rg -l eastasia
2. Create App Service Plan (Linux):
   az appservice plan create -g bird-label-rg -n bird-label-plan --sku B1 --is-linux
3. Create Web App (container placeholder):
   az webapp create -g bird-label-rg -p bird-label-plan -n bird-label-app1 --runtime "PYTHON:3.10"
4. Create ACR:
   az acr create -g bird-label-rg -n <ACR_NAME> --sku Basic
5. Assign Web App pull permission:
   az role assignment create --assignee <PRINCIPAL_ID> --role "AcrPull" --scope $(az acr show -n <ACR_NAME> --query id -o tsv)

## GitHub Secrets Required
- AZURE_CREDENTIALS: Output of `az ad sp create-for-rbac --name bird-label-deployer --sdk-auth --role contributor --scopes /subscriptions/<SUB_ID>/resourceGroups/bird-label-rg`
- ACR_NAME: Your ACR name (no domain).
- HHOLOVE_API_KEY: API key.

## Workflow Trigger
Push a tag like `v1.5-test` or manually dispatch in Actions tab.

## Health Check
After deploy:
https://bird-label-app1.azurewebsites.net/healthz

If it fails: `az webapp log tail -g bird-label-rg -n bird-label-app1`

## Minimal Dependencies Mode
Environment variable `MINIMAL_DEPS=1` keeps only lightweight libs. Heavy ML libs can be reintroduced by editing Dockerfile to use full `requirements.txt`.

## Switching to Full Requirements
Edit Dockerfile: replace `requirements-minimal.txt` with `requirements.txt` and re-run workflow (may need larger plan).

## Common Issues
- DNS not found: Web App name not created or wrong subscription.
- Continuous restarts: Out-of-memory; reduce workers (2) and use minimal deps.
- 502/503 immediately: Container failed; check logs.

## Manual Container Deploy (CLI)
1. docker build -t bird-label .
2. docker tag bird-label <ACR_NAME>.azurecr.io/bird-label:latest
3. docker push <ACR_NAME>.azurecr.io/bird-label:latest
4. az webapp config container set -g bird-label-rg -n bird-label-app1 \
   --docker-custom-image-name <ACR_NAME>.azurecr.io/bird-label:latest \
   --docker-registry-server-url https://<ACR_NAME>.azurecr.io

## Rollback
Point image back to a previous digest or re-run workflow on older tag.
