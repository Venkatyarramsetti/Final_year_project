from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Hazard Spotter AI API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hazard Spotter AI API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/detect")
async def detect_hazards():
    """
    Simplified detection endpoint for testing frontend-backend communication
    """
    # This is a mock response for testing
    return {
        "status": "success",
        "filename": "test_image.jpg",
        "results": {
            "total_detections": 3,
            "hazardous_count": 2,
            "healthy_count": 1,
            "overall_assessment": "Hazardous",
            "detections": [
                {
                    "class_name": "Plastic bottle",
                    "confidence": 0.92,
                    "category": "Hazardous",
                    "bounding_box": {
                        "x1": 100,
                        "y1": 100,
                        "x2": 200,
                        "y2": 300
                    }
                },
                {
                    "class_name": "Cigarette",
                    "confidence": 0.85,
                    "category": "Hazardous",
                    "bounding_box": {
                        "x1": 300,
                        "y1": 200,
                        "x2": 350,
                        "y2": 250
                    }
                },
                {
                    "class_name": "Paper",
                    "confidence": 0.78,
                    "category": "Healthy",
                    "bounding_box": {
                        "x1": 400,
                        "y1": 300,
                        "x2": 500,
                        "y2": 400
                    }
                }
            ]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
