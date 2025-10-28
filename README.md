# Stillbirth Risk Assessment App

A bilingual (English/Arabic) Streamlit web application for assessing stillbirth risk using an XGBoost machine learning model.

## Features

- **Real ML Model**: Uses XGBoost trained on 20 clinical features
- **Bilingual Interface**: Full support for English and Arabic (RTL)
- **Risk Assessment**: 3-level risk stratification (Low, Moderate, High)
- **PDF Reports**: Generate downloadable PDF reports with risk assessment
- **AI Explanations**: Optional LLM-powered risk factor explanations
- **Patient History**: Track and export patient assessments

## Model Features

The XGBoost model uses 20 input features:

1. Pregnancy duration (gestational weeks)
2. Baby weight (kg)
3. Prenatal clinic visits
4. Emergency visits
5. Height (cm)
6. BMI
7. Weight (kg)
8. Systolic blood pressure
9. Diastolic blood pressure
10. Diabetes status
11. Hypertension status
12. Creatinine levels (optional)
13. HbA1c levels (optional)
14. Potassium levels (optional)
15. Ferric carboxymaltose medication (optional)
16. Metoprolol medication (optional)
17. Inpatient visits
18. Twins
19. Delivery type
20. Year

## Deployment on Render

### Prerequisites

- GitHub account
- Render account (free tier available)

### Steps

1. **Push to GitHub**:
   ```bash
   cd Streamlit
   git init
   git add .
   git commit -m "Initial commit: Stillbirth risk assessment app"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `stillbirth-risk-assessment`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
     - **Instance Type**: Free

3. **Environment Variables** (Optional):
   If using OpenRouter LLM features, add:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `OPENROUTER_MODEL`: Model name (default: `openai/gpt-oss-20b:free`)

4. **Deploy**: Click "Create Web Service"

## Local Development

### Installation

```bash
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Environment Variables

Create a `.env` file in the Streamlit directory:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-oss-20b:free
```

## Files Structure

```
Streamlit/
├── app.py                  # Main Streamlit application
├── preprocessing.py        # Feature preprocessing and model inference
├── xgb_model.joblib       # Trained XGBoost model
├── features_used.txt      # List of model features
├── requirements.txt       # Python dependencies
├── AI4Life.png           # Logo image
├── .env                  # Environment variables (not in git)
└── README.md             # This file
```

## Model Information

- **Model Type**: XGBoost Classifier
- **Training Data**: Historical birth data with stillbirth outcomes
- **Output**: Probability of stillbirth (0-100%)
- **Risk Levels**: 
  - Low: 0-33%
  - Moderate: 34-66%
  - High: 67-100%

## Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: XGBoost, scikit-learn
- **PDF Generation**: ReportLab
- **LLM Integration**: OpenRouter API (optional)
- **Language**: Python 3.9+

## License

Internal use only - Healthcare application

## Support

For issues or questions, please contact the development team.

