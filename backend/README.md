# Hazard Spotter AI - Backend

This is the backend API for the Hazard Spotter AI application that performs garbage classification using deep learning.

## Features

- **YOLO-based Object Detection**: Uses YOLOv8 for detecting garbage objects in images
- **Health/Hazard Classification**: Categorizes detected objects as "Healthy" or "Hazardous"
- **REST API**: FastAPI-based API for easy integration with frontend
- **Image Upload Support**: Supports both file upload and base64 image processing

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Windows Users**:
   ```bash
   start_server.bat
   ```

2. **Linux/Mac Users**:
   ```bash
   chmod +x start_server.sh
   ./start_server.sh
   ```

3. **Manual Installation**:
   ```bash
   # Create virtual environment
   python -m venv hazard_env
   
   # Activate virtual environment
   # Windows:
   hazard_env\Scripts\activate
   # Linux/Mac:
   source hazard_env/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the server
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

### Health Check
- **GET** `/health` - Check if the API and model are running

### Detection
- **POST** `/detect` - Upload an image file for hazard detection
- **POST** `/detect-base64` - Send base64 encoded image for detection

### Model Information
- **GET** `/model-info` - Get information about the loaded model

## Example Usage

### Using curl to test the API:

```bash
# Health check
curl http://localhost:8000/health

# Upload image for detection
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

## Model Details

The backend uses a YOLOv8 model trained on the TACO dataset for garbage detection. The model categorizes detected objects into:

- **Hazardous**: Items like broken glass, cigarettes, plastic waste, etc.
- **Healthy**: Items that are not considered hazardous

## Development

The backend consists of:
- `main.py` - FastAPI application and routes
- `model_manager.py` - YOLO model management and inference logic
- `requirements.txt` - Python dependencies

## Notes

- The first run will download the dataset and train the model, which may take some time
- Subsequent runs will use the pre-trained model for faster startup
- The API supports CORS for frontend integration
- Images are temporarily saved during processing and cleaned up automatically
