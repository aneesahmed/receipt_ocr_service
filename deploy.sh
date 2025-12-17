#!/bin/bash

# Define strict path to ensure commands work in automation
export PATH=$PATH:/usr/bin:/usr/local/bin

IMAGE_NAME="receipt-ocr-service"
CONTAINER_NAME="receipt-ocr"

echo "=========================================="
echo "🚀 STARTING SERVER DEPLOYMENT"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Path: $PWD"
echo "=========================================="

# 1. Force Git Pull
echo "⬇️  Pulling latest code..."
git fetch --all
git reset --hard origin/main
if [ $? -ne 0 ]; then
    echo "❌ Git pull failed!"
    exit 1
fi

# 2. Build Image (No Cache = Forces Update)
echo "🔨 Building Docker Image (Forcing Rebuild)..."
docker build --no-cache -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

# 3. Stop & Remove Old Container
echo "🛑 Removing old container..."
docker rm -f $CONTAINER_NAME || true

# 4. Run New Container
echo "🏃 Starting new container..."
docker run --gpus all -d \
  -p 8002:8002 \
  --name $CONTAINER_NAME \
  --restart always \
  $IMAGE_NAME

echo "=========================================="
echo "✅ DEPLOYMENT FINISHED SUCCESSFULLY"
echo "=========================================="