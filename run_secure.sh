#!/bin/bash

# Check if SSL certificate files exist
if [ ! -f "app/cert.pem" ] || [ ! -f "app/key.pem" ]; then
    echo "Error: SSL certificate files not found. Please generate them first."
    exit 1
fi

# Start FastAPI server with SSL
uvicorn app.main:app --reload --host 0.0.0.0 --port 8443 --ssl-keyfile=app/key.pem --ssl-certfile=app/cert.pem