#!/bin/bash
# This script downloads the YOLOv8n model if it doesn't exist

echo "Checking for YOLOv8n model..."
if [ ! -f "yolov8n.pt" ]; then
    echo "Downloading YOLOv8n model..."
    pip install ultralytics
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    echo "Model downloaded successfully!"
else
    echo "Model already exists!"
fi
