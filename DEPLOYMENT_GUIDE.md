# Deployment Guide for Render

## Quick Start

### 1. Prepare Your GitHub Repository

```bash
cd C:\Users\PC\Desktop\Datathon\Streamlit
git init
git add .
git commit -m "Initial commit: Stillbirth Risk Assessment with XGBoost"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### 2. Deploy on Render

#### Option A: Using render.yaml (Recommended)

1. Go to https://dashboard.render.com/
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and configure the service
5. Click "Apply" to deploy

#### Option B: Manual Setup

1. Go to https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the following:

**Basic Settings:**
- Name: `stillbirth-risk-assessment`
- Region: Choose closest to your users
- Branch: `main`
- Root Directory: Leave empty (or specify if your Streamlit folder is in a subdirectory)
- Environment: `Python 3`
- Python Version: `3.9.18`

**Build & Deploy:**
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

**Instance Type:**
- Free (512 MB RAM) - Good for testing
- Starter ($7/month) - Better for production

**Advanced Settings (Optional):**
- Auto-Deploy: Yes (deploy on every push to main)
- Health Check Path: Leave empty

5. Click "Create Web Service"

### 3. Environment Variables (Optional)

If you want to use AI-powered explanations (requires OpenRouter API):

1. In Render dashboard, go to your service
2. Click "Environment" tab
3. Add the following variables:
   - Key: `OPENROUTER_API_KEY`, Value: Your API key from https://openrouter.ai/keys
   - Key: `OPENROUTER_MODEL`, Value: `openai/gpt-oss-20b:free`

**Note**: The app works without these - it will use rule-based explanations instead.

### 4. Verify Deployment

After deployment completes (usually 2-5 minutes):

1. Click on the URL provided by Render (e.g., `https://stillbirth-risk-assessment.onrender.com`)
2. Test the application with sample data
3. Try both English and Arabic interfaces
4. Generate a PDF report to verify all features work

## Files Checklist

Make sure these files are in your Streamlit directory:

- ✅ `app.py` - Main application
- ✅ `preprocessing.py` - Model preprocessing
- ✅ `xgb_model.joblib` - Trained model (20 MB)
- ✅ `features_used.txt` - Feature list
- ✅ `requirements.txt` - Dependencies
- ✅ `AI4Life.png` - Logo (optional)
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules
- ✅ `render.yaml` - Render config

## Troubleshooting

### Build Fails

**Error**: `Could not find a version that satisfies the requirement...`
- Solution: Check `requirements.txt` versions are compatible
- Try removing version constraints temporarily

**Error**: `ModuleNotFoundError: No module named 'xgboost'`
- Solution: Ensure `xgboost` is in `requirements.txt`

### App Crashes

**Error**: `FileNotFoundError: xgb_model.joblib`
- Solution: Verify the model file is pushed to GitHub (not in .gitignore)
- Check file paths in `preprocessing.py`

**Error**: `Memory limit exceeded`
- Solution: Upgrade to Starter plan ($7/month with 2GB RAM)

### Slow Loading

- XGBoost model takes ~5-10 seconds to load on first request
- Use Render's "Keep Service Alive" feature to prevent cold starts
- Consider upgrading to a paid plan for better performance

## Update Deployed App

To update the app after making changes:

```bash
git add .
git commit -m "Update: description of changes"
git push origin main
```

Render will automatically rebuild and redeploy (if auto-deploy is enabled).

## Cost Estimate

**Free Tier:**
- 750 hours/month free
- Services spin down after 15 minutes of inactivity
- Cold start: ~30 seconds

**Starter Plan ($7/month):**
- Always on
- 2GB RAM
- No cold starts
- Better performance

## Security Notes

1. **Never commit** `.env` file with real API keys
2. Use Render's environment variables for secrets
3. The model file (`xgb_model.joblib`) contains no patient data
4. Patient history is stored in session only (lost on refresh)

## Support

For Render-specific issues:
- Render Docs: https://render.com/docs
- Render Community: https://community.render.com/

For app issues, check the logs in Render dashboard → Logs tab.

