# Hazard Spotter AI - Backend Deployment

This document provides instructions for deploying the Hazard Spotter AI backend to Render.

## Prerequisites

- GitHub repository with your code
- MongoDB Atlas account for database hosting
- Render account

## Deployment Steps

### 1. Set Up MongoDB Atlas

1. Create a MongoDB Atlas account at https://www.mongodb.com/cloud/atlas
2. Create a new cluster (the free tier works fine)
3. Create a database user with a strong password
4. Configure network access (allow connections from anywhere for development)
5. Get your MongoDB connection string

### 2. Deploy to Render

#### Option 1: Manual Deployment

1. Log in to your Render account
2. Click "New" > "Web Service"
3. Connect to your GitHub repository
4. Configure your web service:
   - Name: hazard-spotter-backend
   - Environment: Python
   - Build Command: `pip install -r backend/requirements.txt && cd backend && bash download_model.sh`
   - Start Command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `SECRET_KEY`: A random secure string (generate with `openssl rand -hex 32`)
   - `ACCESS_TOKEN_EXPIRE_MINUTES`: 30
   - `MONGODB_URI`: Your MongoDB connection string

#### Option 2: Using Blueprint

1. Push your code with the included `render.yaml` file
2. In Render, go to "Blueprints"
3. Connect to your repository
4. Render will use the `render.yaml` configuration
5. Manually add the `MONGODB_URI` environment variable

### 3. Verify Deployment

1. Once deployed, visit the service URL
2. Check the `/health` endpoint to ensure the service is running
3. Update your frontend to use the new backend URL

## Troubleshooting

- **Model Loading Issues**: Check the logs in Render to ensure the model is downloading correctly
- **Database Connection Issues**: Verify your MongoDB connection string and network settings
- **Memory/CPU Limitations**: Consider upgrading your Render plan if you encounter resource constraints

## Notes

- The free tier of Render may spin down after periods of inactivity, causing slow initial responses
- The deployment includes the lightweight YOLOv8n model for compatibility with Render's resource constraints
