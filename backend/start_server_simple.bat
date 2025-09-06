@echo off

REM Install dependencies directly (no virtual environment)
pip install fastapi uvicorn[standard] python-multipart pillow ultralytics opencv-python numpy matplotlib torch torchvision kagglehub python-dotenv pydantic

REM Run the FastAPI server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
