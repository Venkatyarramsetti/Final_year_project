# Frontend Deployment Instructions

This document provides instructions for deploying the Hazard Spotter AI frontend to Vercel.

## Prerequisites

- A GitHub account
- A Vercel account (you can sign up using your GitHub account)
- Your project pushed to a GitHub repository

## Preparation Steps

1. Make sure all your changes are committed and pushed to GitHub
2. Ensure your `.env.vercel` file contains the correct backend URL:
   ```
   VITE_BACKEND_URL=https://final-year-project-7xiw.onrender.com/
   ```
3. Ensure your `.gitignore` file includes the following to protect sensitive information:
   ```
   # Environment files
   .env
   .env.*
   !.env.example
   ```

## Deploying to Vercel

1. **Log in to Vercel**
   - Go to [vercel.com](https://vercel.com/)
   - Sign in with your GitHub account

2. **Import your project**
   - Click "Add New..." and select "Project"
   - Select your GitHub repository
   - Vercel will automatically detect that it's a Vite project

3. **Configure project settings**
   - Set the following build settings:
     - Framework Preset: Vite
     - Build Command: `npm run build`
     - Output Directory: `dist`
   - In the Environment Variables section, add:
     - Name: `VITE_BACKEND_URL`
     - Value: `https://final-year-project-7xiw.onrender.com/`

4. **Deploy**
   - Click "Deploy"
   - Vercel will build and deploy your project
   - Once complete, you'll receive a URL where your app is hosted

5. **Verify deployment**
   - Visit the provided URL to make sure your app is working correctly
   - Test the connection to the backend by trying to log in or perform other API-dependent actions

## Updating Your Deployment

When you make changes to your project:

1. Commit and push your changes to GitHub
2. Vercel will automatically redeploy your project

## Troubleshooting

- If your app fails to connect to the backend, verify that the environment variable is set correctly in Vercel
- If you encounter build errors, check the build logs in Vercel for more information
- For routing issues, ensure your `vercel.json` file is properly configured

## Custom Domains

To use a custom domain:

1. Go to your project settings in Vercel
2. Click on "Domains"
3. Add your domain and follow the instructions to configure DNS settings