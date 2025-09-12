# Railway.app deployment - free tier, no Azure CLI needed
# Build: railway up
# One-click deploy: https://railway.app/template/...

web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 180 wsgi_light:app
