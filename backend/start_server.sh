#!/bin/bash

# Create virtual environment
python -m venv hazard_env

# Activate virtual environment (Linux/Mac)
source hazard_env/bin/activate

# For Windows, use:
# hazard_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
