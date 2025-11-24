from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
import uvicorn
import os
import io
import logging
import sys
import traceback
from typing import Annotated
from PIL import Image
import base64
from model_manager import ModelManager
from database import get_db, User, users, init_db
from models import UserCreate, UserLogin, Token
from auth import authenticate_user, create_access_token, get_password_hash, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta
import json
import tempfile

# Helper function for temporary file paths
def get_temp_path(filename: str) -> str:
    """Get temporary file path"""
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, filename)

# Import enhanced features (waste categorization & IoT integration)
try:
    from api_extensions import router as extensions_router
    EXTENSIONS_AVAILABLE = True
    logger_ext = logging.getLogger("extensions")
    logger_ext.info("✓ Advanced features loaded (waste categorization + IoT)")
except ImportError as e:
    EXTENSIONS_AVAILABLE = False
    logger_ext = logging.getLogger("extensions")
    logger_ext.warning(f"Advanced features not available: {e}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Automated Garbage Classification System",
    version="2.0.0",
    description="Deep learning-based waste classification with IoT integration for smart waste management"
)

# Include advanced feature endpoints if available
if EXTENSIONS_AVAILABLE:
    app.include_router(extensions_router)
    logger.info("✓ Advanced API endpoints registered")

# Global exception handler for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = {
        "message": str(exc),
        "traceback": traceback.format_exc()
    }
    logging.error(f"Unhandled exception: {error_detail}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

# Configure CORS
allowed_origins = [
    "http://localhost:3000", 
    "http://localhost:8080", 
    "http://localhost:8081", 
    "https://hazard-spotter-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model (this will be loaded once when the server starts)
model_manager = None

@app.on_event("startup")
async def startup_event():
    global model_manager
    try:
        # Initialize database connection
        logger.info("Initializing database connection...")
        await init_db()
        
        # Initialize model (lazy loading - will load when first detection request comes)
        logger.info("Model will be loaded on first detection request...")
        logger.info("Startup complete!")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.warning("Warning: Some features may not work properly")

@app.get("/")
async def root():
    return {"message": "Hazard Spotter AI API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model_manager is not None,
        "api_version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.post("/detect")
async def detect_hazards(file: UploadFile = File(...)):
    """
    Upload an image and get hazard detection results
    """
    global model_manager
    # Lazy load model on first request
    if model_manager is None:
        try:
            logger.info("Loading model for first detection request...")
            model_manager = ModelManager()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=503, detail="Model initialization failed. Please try again later.")
        
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save the image temporarily
        temp_image_path = get_temp_path(f"temp_{file.filename}")
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
            logger.error(f"Error during detection: {str(e)}")
            # Clean up temporary file in case of error
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-base64")
async def detect_hazards_base64(data: dict):
    """
    Upload an image as base64 and get hazard detection results
    """
    # Check if model is loaded
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Please try again later.")
        
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data))
        
        # Save the image temporarily
        temp_image_path = get_temp_path("temp_base64_image.jpg")
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
            logger.error(f"Error during base64 detection: {str(e)}")
            # Clean up temporary file in case of error
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
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

@app.post("/auth/register")
async def register(user: UserCreate):
    db_user = await User.get_by_email(user.email)
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    db_user = User(
        email=user.email,
        name=user.name,
        hashed_password=get_password_hash(user.password)
    )
    
    user_data = {
        "email": db_user.email,
        "name": db_user.name,
        "hashed_password": db_user.hashed_password
    }
    await users.insert_one(user_data)
    return {"message": "User created successfully"}

@app.post("/auth/login")
async def login(request: Request):
    try:
        form_data = await request.form()
        username = str(form_data.get("username", ""))
        password = str(form_data.get("password", ""))
        
        logger.info(f"Login attempt for username: {username}")
        
        user = await User.get_by_email(username)
        if not user:
            logger.warning(f"User not found: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
            
        if not await authenticate_user(user, password):
            logger.warning(f"Invalid password for user: {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}"
        )

@app.get("/auth/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "name": current_user.name
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
