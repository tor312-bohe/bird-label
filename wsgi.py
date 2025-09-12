from app import app as application, init_database
import os

# Ensure database initialized when running under WSGI (gunicorn / Azure)
try:
    init_database()
except Exception as e:
    print(f"⚠️ Database init error (non-fatal): {e}")

# Azure / gunicorn looks for 'application'
app = application  # convenience alias

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
