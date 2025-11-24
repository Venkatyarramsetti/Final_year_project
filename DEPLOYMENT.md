# Deployment Guide

## Prerequisites
- GitHub account
- Vercel account (for frontend)
- Render account (for backend)
- MongoDB Atlas account (for database)

## Backend Deployment (Render)

### 1. Setup MongoDB Atlas
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a free cluster
3. Create a database user
4. Get your connection string
5. Whitelist Render's IP (or use 0.0.0.0/0 for all IPs)

### 2. Deploy to Render
1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: hazard-spotter-backend
   - **Root Directory**: backend
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variables:
   - `MONGODB_URL`: Your MongoDB Atlas connection string
   - `SECRET_KEY`: Generate a secure random string (32+ characters)
   - `PYTHON_VERSION`: 3.12.0
7. Click "Create Web Service"
8. Wait for deployment (first deploy may take 10-15 minutes due to model download)
9. Copy your backend URL (e.g., https://hazard-spotter-backend.onrender.com)

### 3. Important Notes
- Free tier sleeps after 15 min of inactivity (first request may be slow)
- Model file (yolov8n.pt) will be downloaded on first run (~6MB)
- Increase disk space if needed in Render settings

## Frontend Deployment (Vercel)

### 1. Update API URL
1. Edit `frontend/vercel.json`
2. Replace `VITE_API_URL` with your Render backend URL

### 2. Update Frontend Code
1. Create/update `frontend/.env.production`:
   ```
   VITE_API_URL=https://your-backend-app.onrender.com
   ```

### 3. Deploy to Vercel
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "Add New" → "Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: frontend
   - **Build Command**: `npm run build`
   - **Output Directory**: dist
5. Add Environment Variable:
   - `VITE_API_URL`: Your Render backend URL
6. Click "Deploy"
7. Wait for deployment (2-3 minutes)
8. Copy your frontend URL (e.g., https://your-app.vercel.app)

### 4. Update Backend CORS
1. Go to Render dashboard
2. Edit environment variables
3. Update `CORS_ORIGINS` to include your Vercel URL

## Post-Deployment

### Test the Application
1. Visit your Vercel frontend URL
2. Register a new account
3. Try uploading an image for detection
4. Verify results are displayed correctly

### Common Issues

**Backend won't start:**
- Check environment variables are set correctly
- Verify MongoDB connection string
- Check Render logs for errors

**CORS errors:**
- Ensure frontend URL is in backend CORS_ORIGINS
- Check both URLs use HTTPS

**Model loading errors:**
- First deployment is slow (model download)
- Check Render logs
- May need to upgrade disk space

**Database connection errors:**
- Verify MongoDB Atlas connection string
- Check IP whitelist in MongoDB Atlas
- Ensure database user has correct permissions

## Monitoring
- **Backend**: Check Render logs and metrics
- **Frontend**: Check Vercel deployment logs
- **Database**: Monitor MongoDB Atlas metrics

## Cost Estimates
- **Vercel**: Free (hobby tier) - 100GB bandwidth/month
- **Render**: Free tier - 750 hours/month (sleeps after inactivity)
- **MongoDB Atlas**: Free (M0 cluster) - 512MB storage

## Scaling
For production use, consider:
- Render paid plan ($7/month) - no sleep, more resources
- MongoDB Atlas paid tier - better performance
- Custom domain names
- CDN for static assets
