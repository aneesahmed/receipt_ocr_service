#!/bin/bash

# --- CONFIGURATION ---
CONTAINER_NAME="receipt-ocr"
IMAGE_NAME="receipt-ocr-service"
PORT=8002

echo "=========================================="
echo "🚀 DEPLOYING RECEIPT OCR SERVICE"
echo "=========================================="

# 1. Pull latest code
echo "⬇️  Step 1: Pulling latest code..."
git pull
if [ $? -ne 0 ]; then
    echo "❌ Git pull failed! Check your internet or credentials."
    exit 1
fi

# 2. Build the new Image FIRST (If this fails, we don't kill the old app)
echo "🔨 Step 2: Building Docker image..."
docker build -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed! Aborting deployment."
    exit 1
fi

# 3. Force Kill & Remove Old Container
# '|| true' ensures the script doesn't crash if the container doesn't exist
echo "💀 Step 3: Force removing old container..."
docker rm -f $CONTAINER_NAME || true

# 4. Start New Container
echo "🏃 Step 4: Starting new container..."
docker run --gpus all -d \
  -p $PORT:$PORT \
  --name $CONTAINER_NAME \
  --restart always \
  $IMAGE_NAME

echo "=========================================="
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "=========================================="
docker logs --tail 5 $CONTAINER_NAME