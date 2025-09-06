@echo off

REM Create virtual environment
python -m venv hazard_env

REM Activate virtual environment
call hazard_env\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt

REM Run the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
