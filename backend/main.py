from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import io
from PIL import Image
import base64
from model_manager import GarbageDetectionModel
import json

app = FastAPI(title="Hazard Spotter AI API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model (this will be loaded once when the server starts)
model_manager = None

@app.on_event("startup")
async def startup_event():
    global model_manager
    print("Initializing Garbage Detection Model...")
    model_manager = GarbageDetectionModel()
    print("Model initialized successfully!")

@app.get("/")
async def root():
    return {"message": "Hazard Spotter AI API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_manager is not None}

@app.post("/detect")
async def detect_hazards(file: UploadFile = File(...)):
    """
    Upload an image and get hazard detection results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save the image temporarily
        temp_image_path = f"temp_{file.filename}"
        image.save(temp_image_path)
        
        try:
            # Run detection
            results = model_manager.detect_and_categorize(temp_image_path)
            
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            return JSONResponse(content={
                "status": "success",
                "filename": file.filename,
                "results": results
            })
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-base64")
async def detect_hazards_base64(data: dict):
    """
    Upload an image as base64 and get hazard detection results
    """
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data))
        
        # Save the image temporarily
        temp_image_path = "temp_base64_image.jpg"
        image.save(temp_image_path)
        
        try:
            # Run detection
            results = model_manager.detect_and_categorize(temp_image_path)
            
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            return JSONResponse(content={
                "status": "success",
                "results": results
            })
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded model
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "YOLOv8 Garbage Detection",
        "classes": model_manager.get_class_names(),
        "health_hazard_mapping": model_manager.get_health_hazard_mapping()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
