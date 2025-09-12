# Deployment Instructions for Render

## 1. Prepare Your Repository
Make sure all your changes are committed to your GitHub repository. You can use these commands:

```
git add .
git commit -m "Enhance backend for Render deployment"
git push
```

If you encounter git errors, ensure that:
- You're in the correct directory
- Your repository is properly initialized with `git init`
- You've set up the remote correctly with `git remote add origin https://github.com/Venkatyarramsetti/hazard-spotter-ai.git`
- You have permission to push to the repository

## 2. Set Up Render Account
1. Create an account on Render (https://render.com/) if you don't have one
2. Connect your GitHub account to Render

## 3. Create a New Web Service
1. Click "New" and select "Web Service"
2. Select your GitHub repository
3. Render will automatically detect your `render.yaml` file and use its configuration

## 4. Configure Environment Variables
Make sure to add these environment variables in Render dashboard:
- `MONGODB_URI`: Your MongoDB connection string
- `SECRET_KEY`: Will be auto-generated as specified in render.yaml
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Set to 30 as specified in render.yaml

## 5. Database Setup
1. Create a MongoDB database (MongoDB Atlas offers a free tier)
2. Get the connection string and add it to Render's environment variables

## 6. Deploy
1. Click "Create Web Service"
2. Render will automatically deploy your application according to the configuration in render.yaml

## 7. Verify Deployment
1. After deployment completes, visit the provided URL + "/health" to check if the API is running
2. Test the API endpoints using Postman or your frontend application

## Troubleshooting
If you encounter issues during deployment:
1. Check Render logs for detailed error messages
2. Ensure your MongoDB connection string is correct
3. Verify that the Python version in runtime.txt is supported by Render
4. Check that the download_model.sh script has execute permissions