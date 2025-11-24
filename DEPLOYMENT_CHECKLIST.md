# Deployment Checklist ✅

## Pre-Deployment
- [x] Code pushed to GitHub
- [x] Environment files created (.env.example, .env.development, .env.production)
- [x] Deployment configs added (vercel.json, render.yaml)
- [x] Model accuracy improved (confidence threshold increased to 0.45)
- [x] Classification logic fixed (carrots now classified as safe)
- [x] Unnecessary files removed

## Backend Deployment (Render)

### Step 1: Setup MongoDB Atlas
- [ ] Create MongoDB Atlas account at https://www.mongodb.com/cloud/atlas
- [ ] Create a free cluster (M0)
- [ ] Create database user with password
- [ ] Get connection string (format: mongodb+srv://username:password@cluster.mongodb.net/)
- [ ] Whitelist all IPs (0.0.0.0/0) in Network Access

### Step 2: Deploy to Render
- [ ] Go to https://dashboard.render.com/
- [ ] Click "New" → "Web Service"
- [ ] Connect your GitHub account
- [ ] Select repository: `Final_year_project` or `hazard-spotter-ai`
- [ ] Configure:
  - **Name**: hazard-spotter-backend
  - **Root Directory**: backend
  - **Environment**: Python 3
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] Add Environment Variables:
  - `MONGODB_URL`: (paste your MongoDB Atlas connection string)
  - `SECRET_KEY`: (generate random 32+ character string)
  - `PYTHON_VERSION`: 3.12.0
- [ ] Click "Create Web Service"
- [ ] Wait 10-15 minutes for first deployment (model will download)
- [ ] Copy your backend URL (e.g., https://hazard-spotter-backend.onrender.com)

### Step 3: Test Backend
- [ ] Visit: https://your-backend-url.onrender.com/docs
- [ ] Verify API documentation loads
- [ ] Test health endpoint: https://your-backend-url.onrender.com/health

## Frontend Deployment (Vercel)

### Step 1: Update Configuration
- [ ] Edit `frontend/vercel.json`
- [ ] Replace `VITE_API_URL` with your Render backend URL
- [ ] Commit and push changes:
  ```bash
  git add frontend/vercel.json
  git commit -m "Update backend URL for production"
  git push origin main
  ```

### Step 2: Deploy to Vercel
- [ ] Go to https://vercel.com/dashboard
- [ ] Click "Add New" → "Project"
- [ ] Import your GitHub repository
- [ ] Configure:
  - **Framework Preset**: Vite
  - **Root Directory**: frontend
  - **Build Command**: `npm run build` (auto-detected)
  - **Output Directory**: dist (auto-detected)
- [ ] Add Environment Variable:
  - Key: `VITE_BACKEND_URL`
  - Value: (your Render backend URL)
- [ ] Click "Deploy"
- [ ] Wait 2-3 minutes for deployment
- [ ] Copy your frontend URL (e.g., https://your-app.vercel.app)

### Step 3: Update Backend CORS
- [ ] Go back to Render dashboard
- [ ] Select your backend service
- [ ] Go to Environment
- [ ] Add/Update environment variable:
  - Key: `CORS_ORIGINS`
  - Value: `https://your-app.vercel.app,http://localhost:8080`
- [ ] Save changes (backend will redeploy)

## Post-Deployment Testing

### Test Complete Flow
- [ ] Visit your Vercel frontend URL
- [ ] Register a new test account
- [ ] Login with test account
- [ ] Navigate to Detection page
- [ ] Upload a test image (e.g., image with carrots, broccoli)
- [ ] Verify:
  - [ ] Detection completes successfully
  - [ ] Carrots/vegetables show as "Safe"
  - [ ] Results display correctly
  - [ ] Annotated image shows bounding boxes
  - [ ] Model accuracy is reasonable (>60%)

### Common Issues & Solutions

**Backend won't start:**
- Check Render logs for errors
- Verify MongoDB connection string is correct
- Ensure SECRET_KEY is set

**CORS Errors:**
- Verify frontend URL is in CORS_ORIGINS
- Check both URLs use HTTPS (not HTTP)
- Clear browser cache

**Model Loading Errors:**
- First deployment takes 10-15 minutes
- Check Render logs for download progress
- May need to increase instance size if memory errors

**Frontend can't connect to backend:**
- Verify VITE_BACKEND_URL is correct in Vercel
- Check backend is running (visit /docs endpoint)
- Ensure backend URL includes https://

## URLs to Save

- **GitHub Repo**: https://github.com/Venkatyarramsetti/Final_year_project
- **Backend (Render)**: ___________________________
- **Frontend (Vercel)**: ___________________________
- **MongoDB Atlas**: ___________________________
- **Backend API Docs**: ___________________________/docs

## Monitoring

- [ ] Setup Vercel analytics (optional)
- [ ] Monitor Render logs for errors
- [ ] Check MongoDB Atlas metrics
- [ ] Test app daily for first week

## Cost Summary

- **Vercel**: Free (100GB bandwidth/month)
- **Render**: Free (750 hours/month, sleeps after 15 min)
- **MongoDB Atlas**: Free (512MB storage)
- **Total**: $0/month (free tier)

## Notes

- Free Render instance sleeps after 15 minutes of inactivity
- First request after sleep may take 30-60 seconds
- Consider paid plan ($7/month) for production use
- Model downloads on first backend deployment (~6MB)

## Support

If you encounter issues:
1. Check DEPLOYMENT.md for detailed instructions
2. Review Render/Vercel logs
3. Test locally first
4. Verify all environment variables are set correctly
