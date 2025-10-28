"""
Preprocessing Module for Stillbirth Risk Assessment
Handles feature engineering and prediction for Streamlit app with XGBoost model
"""

import joblib
import pandas as pd
import numpy as np
import os

# Load trained model and feature names
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgb_model.joblib')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'features_used.txt')


def load_model_artifacts():
    """Load trained XGBoost model and feature names"""
    # Load the XGBoost model
    model = joblib.load(MODEL_PATH)
    
    # Load feature names from features_used.txt
    with open(FEATURES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip() and line.strip() != 'feature']
    
    return model, feature_names


def preprocess_input(user_input):
    """
    Convert user input from Streamlit form to model features
    
    Expected features (20 total):
    1. pregnancyduration
    2. babyweight
    3. visit_pregnancy_clinic
    4. total_emergency_visits
    5. height
    6. bmi
    7. weight
    8. systolic
    9. diastolic
    10. has_diabetes
    11. has_hypertension
    12. Creatinine (Mass/volume) in Serum or Plasma_mean
    13. Hemoglobin A1c/Hemoglobin. Total in Blood_mean
    14. Potassium (Moles/volume) in Serum or Plasma_mean
    15. ferric carboxymaltose_times
    16. metoprolol_times
    17. total_inpatient_visits
    18. twins
    19. deliverytype
    20. year
    
    Args:
        user_input (dict): Dictionary with user inputs from Streamlit
        
    Returns:
        pd.DataFrame: Single-row dataframe with all features
    """
    # Initialize feature dictionary with exact feature names
    features = {}
    
    # Feature 1: pregnancyduration (gestational weeks)
    features['pregnancyduration'] = user_input.get('gestational_weeks', 39)
    
    # Feature 2: babyweight (kg)
    features['babyweight'] = user_input.get('babyweight', 3.2)
    
    # Feature 3: visit_pregnancy_clinic (number of prenatal visits)
    features['visit_pregnancy_clinic'] = user_input.get('prenatal_visits', 4)
    
    # Feature 4: total_emergency_visits
    features['total_emergency_visits'] = user_input.get('total_emergency_visits', 0)
    
    # Feature 5: height (cm)
    features['height'] = user_input.get('height', 165)
    
    # Feature 6: bmi
    bmi = user_input.get('bmi', 27.0)
    features['bmi'] = bmi
    
    # Feature 7: weight (kg) - calculated from BMI and height
    height_m = features['height'] / 100.0
    features['weight'] = bmi * (height_m ** 2)
    
    # Feature 8: systolic (blood pressure)
    features['systolic'] = user_input.get('systolic_bp', 120)
    
    # Feature 9: diastolic (blood pressure)
    features['diastolic'] = user_input.get('diastolic_bp', 75)
    
    # Feature 10: has_diabetes (0=no, 1=yes)
    diabetes = user_input.get('diabetes', 'no')
    features['has_diabetes'] = 1 if (diabetes == 'yes' or diabetes == 'نعم') else 0
    
    # Feature 11: has_hypertension (0=no, 1=yes)
    hypertension = user_input.get('hypertension', 'no')
    features['has_hypertension'] = 1 if (hypertension == 'yes' or hypertension == 'نعم') else 0
    
    # Feature 12: Creatinine (Mass/volume) in Serum or Plasma_mean (mg/dL)
    features['Creatinine (Mass/volume) in Serum or Plasma_mean'] = user_input.get('creatinine_mean', 0.0)
    
    # Feature 13: Hemoglobin A1c/Hemoglobin. Total in Blood_mean (%)
    features['Hemoglobin A1c/Hemoglobin. Total in Blood_mean'] = user_input.get('hba1c_mean', 0.0)
    
    # Feature 14: Potassium (Moles/volume) in Serum or Plasma_mean (mmol/L)
    features['Potassium (Moles/volume) in Serum or Plasma_mean'] = user_input.get('potassium_mean', 0.0)
    
    # Feature 15: ferric carboxymaltose_times (number of times prescribed)
    features['ferric carboxymaltose_times'] = user_input.get('ferric_carboxymaltose_times', 0)
    
    # Feature 16: metoprolol_times (number of times prescribed)
    features['metoprolol_times'] = user_input.get('metoprolol_times', 0)
    
    # Feature 17: total_inpatient_visits
    features['total_inpatient_visits'] = user_input.get('total_inpatient_visits', 0)
    
    # Feature 18: twins (0=no, 1=yes)
    features['twins'] = user_input.get('twins', 0)
    
    # Feature 19: deliverytype (1=vaginal, 2=cesarean, etc.)
    features['deliverytype'] = user_input.get('deliverytype', 1)
    
    # Feature 20: year
    from datetime import datetime
    features['year'] = user_input.get('year', datetime.now().year)
    
    return pd.DataFrame([features])


def align_features(df, required_features):
    """
    Align dataframe columns with model's expected features
    Ensure exact order and presence of all features
    """
    # Add missing features with 0
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match model's expected order
    df = df[required_features]
    
    return df


def predict_risk(user_input):
    """
    Make prediction using the trained XGBoost model
    
    Args:
        user_input (dict): User inputs from Streamlit form
    
    Returns:
        dict: {
            'risk_score': float (0-1, probability of stillbirth),
            'risk_percentage': int (0-100),
            'risk_level': str ('Low', 'Moderate', 'High'),
            'risk_band': str ('low', 'mod', 'high')
        }
    """
    # Load model and features
    model, feature_names = load_model_artifacts()
    
    # Preprocess input
    features_df = preprocess_input(user_input)
    
    # Align with model features
    features_df = align_features(features_df, feature_names)
    
    # Make prediction - predict probability of survival (isalive=1)
    # The model predicts probability of being alive
    alive_probability = model.predict_proba(features_df)[0, 1]
    
    # Stillbirth risk is the complement
    death_probability = 1 - alive_probability
    
    # Convert to percentage (0-100)
    risk_percentage = int(round(death_probability * 100))
    
    # Classify risk level based on percentage
    if risk_percentage <= 33:
        risk_level = 'Low'
        risk_band = 'low'
    elif risk_percentage <= 66:
        risk_level = 'Moderate'
        risk_band = 'mod'
    else:
        risk_level = 'High'
        risk_band = 'high'
    
    return {
        'risk_score': death_probability,
        'risk_percentage': risk_percentage,
        'risk_level': risk_level,
        'risk_band': risk_band
    }
