# app.py
# Pro styled bilingual dashboard — 3-level gauge (Low / Moderate / High)
# Real XGBoost ML model for stillbirth risk prediction
# PDF export: branded header, risk badge, 3-segment gauge with labels, inputs table.
import math
import io
import os
import base64
from datetime import datetime
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import preprocessing  # Import our preprocessing module with XGBoost model

# Load environment variables from .env file
load_dotenv()

# ---- OpenRouter LLM Setup ----
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

# ---- PDF deps ----
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---- Arabic text support ----
try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    st.warning("⚠️ For proper Arabic text in PDFs, install: pip install arabic-reshaper python-bidi")

st.set_page_config(page_title="Stillbirth Risk Assessment", page_icon="🏥", layout="wide")

# =============================
# Styling (green theme + lab-style 3-level gauge)
# =============================
st.markdown("""
<style>
:root{
  --bg1: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%);
  --bg2: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%);
  --text: #0f172a;
  --muted: #475569;

  --card: rgba(255,255,255,0.95);
  --card-border: rgba(34, 197, 94, 0.2);
  --shadow: 0 8px 30px rgba(16, 185, 129, 0.15);

  --low-start: #22c55e;
  --low-end: #16a34a;
  --mod-start: #f59e0b;
  --mod-end: #d97706;
  --high-start: #ef4444;
  --high-end: #dc2626;

  --accent-primary: #10b981;
  --accent-secondary: #059669;
  --gradient-primary: linear-gradient(135deg, #10b981 0%, #059669 100%);
  --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
  --gradient-warning: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  --gradient-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  --gradient-professional: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
}

[data-testid="stAppViewContainer"]{
  background: var(--bg1);
  color: var(--text);
  font-family: 'Inter', 'Tajawal', sans-serif;
}

.hero {
  background: rgba(255, 255, 255, 0.95);
  color: var(--text);
  padding: 32px 40px;
  border-radius: 24px;
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
  border: 2px solid rgba(16, 185, 129, 0.1);
  display: flex;
  align-items: center;
  gap: 24px;
}

.hero::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, transparent 70%);
  animation: float 6s ease-in-out infinite;
  z-index: 0;
}

@keyframes float {
  0%, 100% { transform: translateY(0px) rotate(0deg); }
  50% { transform: translateY(-10px) rotate(180deg); }
}

.hero-logo {
  width: 120px;
  height: 120px;
  object-fit: contain;
  border-radius: 20px;
  box-shadow: 0 6px 20px rgba(16, 185, 129, 0.2);
  border: none;
  background: transparent;
  padding: 0;
  transition: all 0.3s ease;
  position: relative;
  z-index: 1;
  flex-shrink: 0;
  animation: logoPulsate 2s ease-in-out infinite;
}

.hero-logo:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
  animation-play-state: paused;
}

@keyframes logoPulsate {
  0%, 100% {
    transform: scale(1);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.2);
  }
  50% {
    transform: scale(1.08);
    box-shadow: 0 8px 30px rgba(16, 185, 129, 0.4);
  }
}

.hero-logo-placeholder {
  width: 120px;
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 3rem;
  border-radius: 20px;
  box-shadow: 0 6px 20px rgba(16, 185, 129, 0.2);
  border: 3px solid rgba(16, 185, 129, 0.2);
  background: rgba(16, 185, 129, 0.05);
  position: relative;
  z-index: 1;
  flex-shrink: 0;
  animation: logoPulsate 2s ease-in-out infinite;
}

.hero-content {
  flex: 1;
  position: relative;
  z-index: 1;
}

.hero h1 {
  margin: 0 0 12px 0;
  font-size: 2.3rem;
  font-weight: 800;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero .sub {
  color: var(--muted);
  font-size: 1.2rem;
  font-weight: 500;
  line-height: 1.6;
}

.lab-wrap {
  position: relative;
  border: 2px solid var(--card-border);
  border-radius: 20px;
  padding: 28px 26px;
  background: var(--card);
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow);
  background-image:
    radial-gradient(circle at 100% 0%, rgba(16, 185, 129, 0.06) 0%, transparent 50%),
    radial-gradient(circle at 0% 100%, rgba(34, 197, 94, 0.06) 0%, transparent 50%);
}

.lab-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.lab-name {
  font-weight: 800;
  letter-spacing: 0.3px;
  font-size: 1.4rem;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.lab-ref {
  color: var(--muted);
  font-size: 1.1rem;
  font-weight: 600;
}

.lab-band {
  display: flex;
  gap: 4px;
  align-items: center;
  margin-top: 20px;
  position: relative;
  height: 20px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.seg {
  height: 100%;
  border-radius: 0;
  position: relative;
  transition: all 0.3s ease;
}

.seg.green {
  background: linear-gradient(90deg, var(--low-start), var(--low-end));
  box-shadow: inset 0 1px 3px rgba(255,255,255,0.4);
}

.seg.amber {
  background: linear-gradient(90deg, var(--mod-start), var(--mod-end));
  box-shadow: inset 0 1px 3px rgba(255,255,255,0.4);
}

.seg.red {
  background: linear-gradient(90deg, var(--high-start), var(--high-end));
  box-shadow: inset 0 1px 3px rgba(255,255,255,0.4);
}

.seg:hover {
  filter: brightness(1.1);
  transform: scaleY(1.1);
}

.lab-labels {
  display: flex;
  justify-content: space-between;
  color: var(--muted);
  font-size: 1rem;
  margin-top: 16px;
  font-weight: 700;
}

.marker {
  position: relative;
  height: 52px;
  transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.marker .pin {
  position: absolute;
  top: -14px;
  transform: translateX(-50%);
  width: 0;
  height: 0;
  border-left: 12px solid transparent;
  border-right: 12px solid transparent;
  border-top: 18px solid #10b981;
  filter: drop-shadow(0 3px 6px rgba(16, 185, 129, 0.4));
  transition: all 0.3s ease;
}

.marker .pill {
  position: absolute;
  top: 10px;
  transform: translateX(-50%);
  background: linear-gradient(135deg, #10b981, #059669);
  color: #fff;
  padding: 0.5rem 1.4rem;
  border-radius: 999px;
  font-weight: 900;
  font-size: 1.1rem;
  box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
  border: 2px solid rgba(255,255,255,0.15);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { transform: translateX(-50%) scale(1); }
  50% { transform: translateX(-50%) scale(1.05); }
}

.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
  border-radius: 14px;
  border: 2px solid #e2e8f0;
  transition: all 0.3s ease;
  background: rgba(255,255,255,0.95);
  padding: 12px 16px;
  font-size: 1rem;
}

.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.15);
  background: rgba(255,255,255,1);
}

.stButton>button {
  border-radius: 14px;
  border: none;
  background: var(--gradient-primary);
  color: white;
  font-weight: 700;
  padding: 1rem 2rem;
  transition: all 0.3s ease;
  box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35);
  font-size: 1.1rem;
}

.stButton>button:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 25px rgba(16, 185, 129, 0.45);
  background: linear-gradient(135deg, #059669 0%, #047857 100%);
}

.stDownloadButton>button {
  border-radius: 14px;
  border: none;
  background: var(--gradient-success);
  color: white;
  font-weight: 700;
  padding: 1rem 2rem;
  transition: all 0.3s ease;
  box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35);
  font-size: 1.1rem;
}

.stDownloadButton>button:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 25px rgba(16, 185, 129, 0.45);
}

.dataframe {
  border-radius: 20px;
  overflow: hidden;
  box-shadow: var(--shadow);
  border: 1px solid rgba(16, 185, 129, 0.1);
}

hr.soft {
  border: none;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
  margin: 2.5rem 0;
}

.small {
  color: var(--muted);
  font-size: 0.95rem;
  text-align: center;
  margin-top: 16px;
  font-style: italic;
}

.risk-badge {
  padding: 0.5rem 1rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.risk-low {
  background: var(--gradient-success);
  color: white;
}

.risk-mod {
  background: var(--gradient-warning);
  color: white;
}

.risk-high {
  background: var(--gradient-danger);
  color: white;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.lab-wrap, .hero, .dashboard-card {
  animation: fadeInUp 0.6s ease-out;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f0fdf4 0%, #ecfdf5 100%);
  border-right: 2px solid rgba(16, 185, 129, 0.1);
}

::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f5f9;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: var(--accent-primary);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #059669;
}

.stForm {
  background: var(--card);
  backdrop-filter: blur(10px);
  border-radius: 24px;
  padding: 32px;
  box-shadow: var(--shadow);
  border: 2px solid var(--card-border);
  background-image:
    radial-gradient(circle at 100% 0%, rgba(16, 185, 129, 0.06) 0%, transparent 50%),
    radial-gradient(circle at 0% 100%, rgba(34, 197, 94, 0.06) 0%, transparent 50%);
}

.history-container {
  background: var(--card);
  backdrop-filter: blur(10px);
  border-radius: 24px;
  padding: 32px;
  box-shadow: var(--shadow);
  border: 2px solid var(--card-border);
  margin-top: 2.5rem;
  background-image:
    radial-gradient(circle at 0% 0%, rgba(16, 185, 129, 0.06) 0%, transparent 50%),
    radial-gradient(circle at 100% 100%, rgba(34, 197, 94, 0.06) 0%, transparent 50%);
}

.download-history-btn button {
  border-radius: 14px;
  border: none;
  background: var(--gradient-professional) !important;
  color: white !important;
  font-weight: 700;
  padding: 1rem 2rem;
  transition: all 0.3s ease;
  box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
  font-size: 1.1rem;
}

.download-history-btn button:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 25px rgba(16, 185, 129, 0.5);
  filter: brightness(1.05);
}

/* Statistics Cards */
.stats-container {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin: 24px 0;
}

.stat-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
  border: 2px solid rgba(16, 185, 129, 0.1);
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 12px 30px rgba(16, 185, 129, 0.2);
}

.stat-number {
  font-size: 2.2rem;
  font-weight: 800;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: block;
  margin-bottom: 8px;
}

.stat-title {
  font-size: 1rem;
  color: var(--muted);
  font-weight: 600;
}

/* Dashboard Cards */
.dashboard-container {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
  margin-bottom: 30px;
}

.dashboard-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
  border: 2px solid rgba(16, 185, 129, 0.1);
  text-align: center;
  transition: all 0.3s ease;
  cursor: pointer;
}

.dashboard-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 16px 40px rgba(16, 185, 129, 0.25);
  border-color: rgba(16, 185, 129, 0.3);
}

.dashboard-icon {
  font-size: 2.8rem;
  margin-bottom: 16px;
  color: var(--accent-primary);
}

.dashboard-title {
  font-weight: 800;
  font-size: 1.3rem;
  margin-bottom: 12px;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.dashboard-desc {
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.5;
}

/* Dashboard Button Styling */
div[data-testid="column"] > div > button[key="lab_btn"],
div[data-testid="column"] > div > button[key="med_btn"],
div[data-testid="column"] > div > button[key="rec_btn"] {
  height: 180px !important;
  white-space: pre-line !important;
  font-size: 1rem !important;
  line-height: 1.6 !important;
  background: rgba(255, 255, 255, 0.95) !important;
  border: 2px solid rgba(16, 185, 129, 0.15) !important;
  color: var(--text) !important;
  box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15) !important;
}

div[data-testid="column"] > div > button[key="lab_btn"]:hover,
div[data-testid="column"] > div > button[key="med_btn"]:hover,
div[data-testid="column"] > div > button[key="rec_btn"]:hover {
  transform: translateY(-5px) !important;
  box-shadow: 0 16px 40px rgba(16, 185, 129, 0.25) !important;
  border-color: rgba(16, 185, 129, 0.3) !important;
  background: rgba(255, 255, 255, 1) !important;
}

.section-header {
  color: #000000;
  font-size: 2rem;
  font-weight: 800;
  margin: 2rem 0 1rem 0;
}

/* Required Field Styling */
label:has(+ div input[aria-label*="*"]),
div[data-testid="stTextInput"] label:contains("*") {
  font-weight: 600;
}

/* Responsive Design */
@media (max-width: 768px) {
  .hero {
    flex-direction: column !important;
    text-align: center;
    padding: 24px;
  }
  
  .hero-content {
    text-align: center !important;
  }
  
  .hero h1 {
    font-size: 1.8rem;
  }
  
  .hero .sub {
    font-size: 1rem;
  }
  
  .stats-container {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .dashboard-container {
    grid-template-columns: 1fr;
  }
}
</style>
""", unsafe_allow_html=True)

# =============================
# Language
# =============================
lang = st.sidebar.radio("Language / اللغة", ["English", "العربية"], horizontal=True, index=0)
AR = (lang == "العربية")

def L(en, ar):
    return ar if AR else en

# ===== RTL layout when Arabic is active (keep sidebar mechanics working) =====
if AR:
    st.markdown("""
    <style>
    .block-container, .stMain { direction: rtl; }
    [data-testid="stAppViewContainer"] { direction: ltr; }
    [data-testid="stSidebar"] { direction: ltr; }
    [data-testid="stSidebar"] .stRadio { direction: rtl; text-align: right; }
    [data-testid="collapsedControl"] {
      right: 12px !important;
      left: auto !important;
      direction: ltr;
      z-index: 9999;
      pointer-events: auto;
    }
    .hero { 
      flex-direction: row-reverse;
    }
    .hero-content {
      text-align: right;
    }
    .hero h1, .hero .sub { text-align: right; }
    .stMarkdown, .stText { text-align: right; }
    label, .stTextInput, .stNumberInput, .stSelectbox, .stDateInput, .stTimeInput { text-align: right; }
    .lab-head { flex-direction: row-reverse; }
    .lab-labels { direction: rtl; }
    .small { text-align: right; }
    .stat-card, .dashboard-card { direction: rtl; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .block-container, .stMain { direction: ltr; }
    [data-testid="stAppViewContainer"] { direction: ltr; }
    [data-testid="stSidebar"] { direction: ltr; }
    .hero { 
      flex-direction: row;
    }
    .hero-content {
      text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# App State (History)
# =============================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "timestamp", "patient_id", "patient_name",
        "risk_level", "score_pct", "explanation",
        "gestational_weeks", "babyweight", "bmi", "height",
        "systolic_bp", "diastolic_bp", "prenatal_visits",
        "emergency_visits", "inpatient_visits",
        "diabetes", "hypertension", "twins", "deliverytype"
    ])

# =============================
# Statistics Calculation
# =============================
def calculate_statistics():
    """Calculate statistics from the history"""
    df = st.session_state.history
    if len(df) == 0:
        return {
            "total_cases": 0,
            "low_risk": 0,
            "moderate_risk": 0,
            "high_risk": 0,
            "avg_score": 0
        }
    
    total_cases = len(df)
    
    # Count risk levels - handle both English and Arabic
    low_risk = len(df[df["risk_level"].str.contains("Low|منخفض", case=False, na=False)])
    moderate_risk = len(df[df["risk_level"].str.contains("Moderate|متوسط", case=False, na=False)])
    high_risk = len(df[df["risk_level"].str.contains("High|مرتفع", case=False, na=False)])
    
    avg_score = df["score_pct"].mean() if total_cases > 0 else 0
    
    return {
        "total_cases": total_cases,
        "low_risk": low_risk,
        "moderate_risk": moderate_risk,
        "high_risk": high_risk,
        "avg_score": round(avg_score, 1)
    }

# =============================
# XGBoost Model Integration
# =============================

def band_from_percent(pct):
    if pct <= 33:
        return (L("Low", "منخفض"), "low")
    if pct <= 66:
        return (L("Moderate", "متوسط"), "mod")
    return (L("High", "مرتفع"), "high")

def openrouter_explain_risk(band_text, pct, inputs, arabic=False):
    """
    Use GPT-OSS-20B from OpenRouter to explain the risk level.
    Returns up to 4 short bullet points.
    """
    language = "Arabic" if arabic else "English"
    system_prompt = (
        "You are a clinical assistant explaining stillbirth risk levels "
        "in 2–4 short, factual bullet points. "
        "Avoid diagnosis or treatment; focus on explaining the contributing factors."
    )

    user_prompt = f"""
    Language: {language}
    Risk Level: {band_text} ({pct}%)
    Inputs: {inputs}
    """

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=200,
        extra_headers={
            "HTTP-Referer": "https://yourappname.streamlit.app",
            "X-Title": "Stillbirth Risk Assessment"
        }
    )

    text = completion.choices[0].message.content.strip()
    bullets = [line.strip("-• ") for line in text.split("\n") if line.strip()]
    return bullets[:4] or [text]

def explanation_for_band(d, band_text, user_input=None):
    """Fallback explanation when LLM is unavailable"""
    base_map = {
        "High": "High risk — increase monitoring.",
        "Moderate": "Moderate risk — tighten follow-up.",
        "Low": "Low risk — continue standard care.",
        "مرتفع": "خطر مرتفع — عزّز المراقبة.",
        "متوسط": "خطر متوسط — شدد المتابعة.",
        "منخفض": "خطر منخفض — استمر بالرعاية المعتادة.",
    }
    notes = []
    
    # Use user_input if available, otherwise use d
    input_data = user_input if user_input else d
    
    if input_data.get("gestational_weeks", 39) < 34:
        notes.append(L("Preterm pregnancy detected.", "حمل مبكر."))
    if input_data.get("babyweight", 3.2) < 2.5:
        notes.append(L("Low birth weight detected.", "وزن منخفض عند الولادة."))
    if input_data.get("systolic_bp", 120) >= 140 or input_data.get("diastolic_bp", 75) >= 90:
        notes.append(L("Elevated blood pressure.", "ضغط دم مرتفع."))
    if str(input_data.get("diabetes", "no")).lower() in ["yes", "نعم"]:
        notes.append(L("Diabetes present.", "وجود سكري."))
    if str(input_data.get("hypertension", "no")).lower() in ["yes", "نعم"]:
        notes.append(L("Hypertension present.", "ارتفاع ضغط الدم."))
    if input_data.get("prenatal_visits", 4) < 3:
        notes.append(L("Limited prenatal care.", "قلّة المتابعة قبل الولادة."))
    if input_data.get("bmi", 27.0) >= 30:
        notes.append(L("Elevated BMI.", "ارتفاع مؤشر كتلة الجسم."))
    if input_data.get("total_emergency_visits", 0) > 2:
        notes.append(L("Multiple emergency visits.", "زيارات طوارئ متعددة."))
    if input_data.get("hba1c_mean", 0) > 6.5:
        notes.append(L("Elevated HbA1c levels.", "ارتفاع مستوى السكر التراكمي."))
    
    return [base_map[band_text]] + notes[:4]

# =============================
# PDF helpers (nicer layout)
# =============================
BRAND_NAME_EN = "Stillbirth Risk Assessment"
BRAND_NAME_AR = "تقييم خطر الجنين"

def _arabic_text(text):
    """Reshape Arabic text for proper display in PDF"""
    if not text or not ARABIC_SUPPORT:
        return text
    try:
        # Check if text contains Arabic characters
        if any('\u0600' <= c <= '\u06FF' for c in str(text)):
            reshaped = reshape(str(text))
            bidi_text = get_display(reshaped)
            return bidi_text
    except:
        pass
    return text

def _setup_pdf_font(use_arabic):
    if use_arabic:
        # Try multiple font paths for Arabic support
        font_paths = [
            "NotoNaskhArabic-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/simpo.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "arial.ttf"
        ]
        
        for ttf in font_paths:
            if os.path.exists(ttf):
                try:
                    pdfmetrics.registerFont(TTFont("ArabicFont", ttf))
                    return "ArabicFont"
                except:
                    continue
        
        # Fallback: Try to use Helvetica but warn
        if not ARABIC_SUPPORT:
            st.warning("⚠️ Arabic font not found. PDF may not display Arabic text correctly.")
    
    return "Helvetica"

def _wrap_lines(c, text, max_width, font, size):
    c.setFont(font, size)
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        probe = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(probe, font, size) <= max_width:
            cur = probe
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]

def _draw_header(c, W, H, AR, font):
    # Light green matching website background
    c.setFillColor(colors.HexColor("#ecfdf5"))  # Soft green matching site background
    c.rect(0, H - 40 * mm, W, 40 * mm, stroke=0, fill=1)
    
    # Try to add logo
    try:
        logo_paths = ["Streamlit/AI4Life.png", "AI4Life.png"]
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                c.drawImage(logo_path, 20 * mm, H - 38 * mm, width=25 * mm, height=25 * mm, preserveAspectRatio=True, mask='auto')
                break
    except:
        pass
    
    # Dark text for light background
    c.setFillColor(colors.HexColor("#0f172a"))  # Dark slate for better contrast
    c.setFont(font, 18)
    title = _arabic_text(BRAND_NAME_AR) if AR else BRAND_NAME_EN
    # For Arabic (RTL), draw from right side
    if AR:
        title_width = pdfmetrics.stringWidth(title, font, 18)
        c.drawString(W - 20 * mm - title_width, H - 26 * mm, title)
    else:
        c.drawString(50 * mm, H - 26 * mm, title)
    
    c.setFont(font, 10)
    c.setFillColor(colors.HexColor("#475569"))  # Muted gray for subtitle
    subtitle = "Risk assessment report" if not AR else _arabic_text("تقرير تقييم الخطورة")
    if AR:
        subtitle_width = pdfmetrics.stringWidth(subtitle, font, 10)
        c.drawString(W - 20 * mm - subtitle_width, H - 32 * mm, subtitle)
    else:
        c.drawString(50 * mm, H - 32 * mm, subtitle)

def _draw_badge(c, x, y, band_code, band_text, font):
    colors_map = {"low": "#16a34a", "mod": "#d97706", "high": "#dc2626"}
    c.setFillColor(colors.HexColor(colors_map[band_code]))
    c.roundRect(x, y, 42 * mm, 10 * mm, 5 * mm, stroke=0, fill=1)
    # Add subtle border
    c.setStrokeColor(colors.HexColor(colors_map[band_code]))
    c.setLineWidth(0.5)
    c.roundRect(x, y, 42 * mm, 10 * mm, 5 * mm, stroke=1, fill=0)
    c.setFillColor(colors.white)
    c.setFont(font, 11)
    c.drawCentredString(x + 21 * mm, y + 3.2 * mm, _arabic_text(str(band_text)))

def _draw_gauge(c, x, y, w, h, pct, font, AR):
    if AR:
        # Arabic (RTL): High (red) on left, Low (green) on right
        c.setFillColor(colors.HexColor("#fecaca"))  # High/Red
        c.rect(x, y, w * 0.33, h, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#fde68a"))  # Moderate/Yellow
        c.rect(x + w * 0.33, y, w * 0.34, h, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#86efac"))  # Low/Green
        c.rect(x + w * 0.67, y, w * 0.33, h, stroke=0, fill=1)
    else:
        # English (LTR): Low (green) on left, High (red) on right
        c.setFillColor(colors.HexColor("#86efac"))  # Low/Green
        c.rect(x, y, w * 0.33, h, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#fde68a"))  # Moderate/Yellow
        c.rect(x + w * 0.33, y, w * 0.34, h, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#fecaca"))  # High/Red
        c.rect(x + w * 0.67, y, w * 0.33, h, stroke=0, fill=1)
    
    # Green border matching website theme
    c.setStrokeColor(colors.HexColor("#10b981"))
    c.setLineWidth(1.5)
    c.rect(x, y, w, h, stroke=1, fill=0)
    c.setFont(font, 9)
    c.setFillColor(colors.HexColor("#334155"))
    
    if AR:
        # Arabic labels (RTL): High - Moderate - Low
        labels = [_arabic_text("مرتفع"), _arabic_text("متوسط"), _arabic_text("منخفض")]
        c.drawString(x, y - 5 * mm, labels[0])  # High on left
        c.drawCentredString(x + w * 0.50, y - 5 * mm, labels[1])  # Moderate center
        c.drawRightString(x + w, y - 5 * mm, labels[2])  # Low on right
    else:
        # English labels (LTR): Low - Moderate - High
        c.drawString(x, y - 5 * mm, "Low")
        c.drawCentredString(x + w * 0.50, y - 5 * mm, "Moderate")
        c.drawRightString(x + w, y - 5 * mm, "High")
    
    # Draw marker - reverse position for Arabic
    if AR:
        marker_pos = x + ((100 - max(2, min(98, pct))) / 100.0) * w
    else:
        marker_pos = x + (max(2, min(98, pct)) / 100.0) * w
    
    # Green marker line matching website theme
    c.setStrokeColor(colors.HexColor("#10b981"))
    c.setLineWidth(3)
    c.line(marker_pos, y - 2 * mm, marker_pos, y + h + 2 * mm)
    c.setFont(font, 9)
    c.setFillColor(colors.HexColor("#10b981"))
    c.drawCentredString(marker_pos, y + h + 3.5 * mm, str(pct))

def _draw_kv(c, x, y, font, label, value, AR=False):
    c.setFont(font, 10)
    c.setFillColor(colors.HexColor("#334155"))
    
    if AR:
        # Right-to-left layout
        label_ar = _arabic_text(label)
        value_ar = _arabic_text(str(value))
        text = f"{label_ar}: {value_ar}"
        c.drawRightString(x + 170 * mm, y, text)
    else:
        c.drawString(x, y, f"{label}: ")
        c.setFont(font, 10)
        c.setFillColor(colors.black)
        c.drawString(x + 42 * mm, y, str(value))

def _draw_inputs_table(c, x, y, font, rows, AR):
    col_w = [60 * mm, 50 * mm]
    row_h = 7 * mm
    c.setFont(font, 10)
    for i, (k, v) in enumerate(rows):
        yy = y - i * row_h
        if i % 2 == 0:
            # Light green tint for alternating rows
            c.setFillColor(colors.HexColor("#f0fdf4"))
            c.rect(x, yy - row_h + 1.5 * mm, sum(col_w), row_h, stroke=0, fill=1)
        c.setFillColor(colors.black)
        
        key_text = _arabic_text(str(k)) if AR else str(k)
        val_text = _arabic_text(str(v)) if AR else str(v)
        
        if AR:
            # Right-to-left: draw key from right, value from left
            c.drawRightString(x + col_w[0] + col_w[1] - 3 * mm, yy - 4.7 * mm, key_text)
            c.drawString(x + 3 * mm, yy - 4.7 * mm, val_text)
        else:
            c.drawString(x + 3 * mm, yy - 4.7 * mm, key_text)
            c.drawRightString(x + col_w[0] + col_w[1] - 3 * mm, yy - 4.7 * mm, val_text)

def build_pdf(patient_id, patient_name, timestamp, pct, band_text, band_code, bullets, d, AR=False):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    font_main = _setup_pdf_font(AR)
    c.setTitle("Stillbirth Risk Assessment Report")
    _draw_header(c, W, H, AR, font_main)
    y = H - 48 * mm
    c.setFont(font_main, 11)
    _draw_kv(c, 20 * mm, y, font_main, ("Timestamp" if not AR else "التاريخ"), timestamp, AR)
    y -= 7 * mm
    _draw_kv(c, 20 * mm, y, font_main, ("Patient ID" if not AR else "رقم المريضة"), patient_id, AR)
    y -= 7 * mm
    _draw_kv(c, 20 * mm, y, font_main, ("Patient Name" if not AR else "اسم المريضة"), patient_name, AR)
    y -= 12 * mm
    _draw_badge(c, 20 * mm, y, band_code, band_text, font_main)
    c.setFont(font_main, 11)
    risk_text = f"Risk Index (0–100): {pct}" if not AR else _arabic_text(f"مؤشر الخطورة (٠–١٠٠): {pct}")
    if AR:
        c.drawRightString(W - 20 * mm, y - 8 * mm, risk_text)
    else:
        c.drawString(20 * mm, y - 8 * mm, risk_text)
    _draw_gauge(c, 20 * mm, y - 22 * mm, 170 * mm, 8 * mm, pct, font_main, AR)
    y = y - 40 * mm
    c.setFont(font_main, 12)
    if AR:
        c.drawRightString(W - 20 * mm, y, _arabic_text("ملاحظات"))
    else:
        c.drawString(20 * mm, y, "Notes")
    y -= 7 * mm
    c.setFont(font_main, 10)
    max_w = W - 40 * mm
    for b in bullets:
        if AR:
            line = _arabic_text(f"{b} •")
        else:
            line = f"• {b}"
        for ln in _wrap_lines(c, line, max_w, font_main, 10):
            if AR:
                c.drawRightString(W - 20 * mm, y, ln)
            else:
                c.drawString(20 * mm, y, ln)
            y -= 6 * mm
    y -= 6 * mm
    c.setFont(font_main, 12)
    if AR:
        c.drawRightString(W - 20 * mm, y, _arabic_text("المدخلات"))
    else:
        c.drawString(20 * mm, y, "Inputs")
    y -= 4 * mm
    rows = [
        ("Gestational age (weeks)" if not AR else "عمر الحمل (أسابيع)", d["gestational_weeks"]),
        ("BMI", d["bmi"]),
        ("Systolic BP" if not AR else "الضغط الانقباضي", d["systolic_bp"]),
        ("Diastolic BP" if not AR else "الضغط الانبساطي", d["diastolic_bp"]),
        ("Prenatal visits" if not AR else "زيارات ما قبل الولادة", d["prenatal_visits"]),
        ("Diabetes" if not AR else "سكري", "yes" if d["diabetes"] == "yes" else "no"),
        ("Hypertension" if not AR else "ارتفاع ضغط", "yes" if d["hypertension"] == "yes" else "no"),
    ]
    if AR:
        rows = [(k, ("نعم" if v == "yes" else "لا") if isinstance(v, str) else v) for k, v in rows]
    _draw_inputs_table(c, 20 * mm, y, font_main, rows, AR)
    c.showPage()
    c.save()
    return buf.getvalue()

# =============================
# Header
# =============================
# Hero card with logo inside
def get_base64_image_local(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return ""

# Try to load the logo
logo_b64 = get_base64_image_local("AI4Life.png")
if not logo_b64:
    logo_b64 = get_base64_image_local("Streamlit/AI4Life.png")

logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="hero-logo" alt="Logo">' if logo_b64 else '<div class="hero-logo-placeholder">🏥</div>'

st.markdown(
    f'<div class="hero">'
    f'{logo_html}'
    f'<div class="hero-content">'
    f'<h1>{L("Stillbirth Risk Assessment", "تقييم خطر الجنين")}</h1>'
    f'<div class="sub">{L("Clinical Decision Support System • Three-level risk stratification", "نظام دعم القرار السريري • تصنيف ثلاثي المستويات للخطورة")}</div>'
    f'</div></div>',
    unsafe_allow_html=True
)
st.markdown("<br/>", unsafe_allow_html=True)

# =============================
# Patient Info form (ID + Name first)
# =============================
st.markdown(f"### {L('Patient Information', 'بيانات المريضة')}")
with st.form("patient_info_form"):
    pid_col, pname_col = st.columns([1, 2])
    patient_id = pid_col.text_input(
        L("Patient ID *", "رقم المريضة *"), 
        placeholder=L("e.g., 23-001", "مثال: ٢٣-٠٠١"),
        help=L("Required field", "حقل إلزامي")
    )
    patient_name = pname_col.text_input(
        L("Patient Name *", "اسم المريضة *"), 
        placeholder=L("e.g., Sara A.", "مثال: سارة"),
        help=L("Required field", "حقل إلزامي")
    )

    st.markdown(f"### {L('Clinical Inputs', 'المدخلات السريرية')}")
    
    # Basic Demographics & Pregnancy Info
    st.markdown(f"**{L('Pregnancy Information', 'معلومات الحمل')}**")
    c1, c2, c3, c4 = st.columns(4)
    gestational_weeks = c1.number_input(L("Gestational age (weeks)", "عمر الحمل (بالأسابيع)"), 20, 42, 39)
    babyweight = c2.number_input(L("Baby weight (kg)", "وزن الطفل (كجم)"), 0.5, 6.0, 3.2, step=0.1)
    twins = c3.selectbox(L("Twins", "توأم"), [L("No", "لا"), L("Yes", "نعم")])
    twins_val = 1 if twins in [L("Yes", "نعم"), "Yes", "نعم"] else 0
    deliverytype = c4.selectbox(L("Delivery type", "نوع الولادة"), [L("Vaginal", "طبيعية"), L("Cesarean", "قيصرية")])
    deliverytype_val = 2 if deliverytype in [L("Cesarean", "قيصرية"), "Cesarean", "قيصرية"] else 1
    
    # Maternal Physical Measurements
    st.markdown(f"**{L('Maternal Measurements', 'قياسات الأم')}**")
    c1, c2, c3 = st.columns(3)
    height = c1.number_input(L("Height (cm)", "الطول (سم)"), 130, 200, 165)
    bmi = c2.number_input("BMI", 16.0, 45.0, 27.0, step=0.1)
    year = c3.number_input(L("Year", "السنة"), 2020, 2025, datetime.now().year)

    # Vital Signs
    st.markdown(f"**{L('Vital Signs', 'العلامات الحيوية')}**")
    c1, c2 = st.columns(2)
    systolic_bp = c1.number_input(L("Systolic BP (mmHg)", "الضغط الانقباضي (ملم زئبق)"), 80, 220, 120)
    diastolic_bp = c2.number_input(L("Diastolic BP (mmHg)", "الضغط الانبساطي (ملم زئبق)"), 50, 140, 75)

    # Healthcare Visits
    st.markdown(f"**{L('Healthcare Visits', 'الزيارات الطبية')}**")
    c1, c2, c3 = st.columns(3)
    prenatal_visits = c1.number_input(L("Prenatal visits", "زيارات ما قبل الولادة"), 0, 30, 4)
    emergency_visits = c2.number_input(L("Emergency visits", "زيارات الطوارئ"), 0, 20, 0)
    inpatient_visits = c3.number_input(L("Inpatient visits", "الزيارات الداخلية"), 0, 10, 0)

    # Medical Conditions
    st.markdown(f"**{L('Medical Conditions', 'الحالات الطبية')}**")
    c1, c2 = st.columns(2)
    diabetes = c1.selectbox(L("Diabetes", "سكري"), [L("no", "لا"), L("yes", "نعم")])
    hypertension = c2.selectbox(L("Hypertension", "ارتفاع ضغط"), [L("no", "لا"), L("yes", "نعم")])

    # Laboratory Tests (Optional)
    with st.expander(L("📊 Laboratory Test Results (Optional)", "📊 نتائج الفحوصات المخبرية (اختياري)"), expanded=False):
        c1, c2, c3 = st.columns(3)
        creatinine_mean = c1.number_input(L("Creatinine (mg/dL)", "الكرياتينين"), 0.0, 5.0, 0.0, step=0.1, help=L("Leave 0 if not available", "اترك 0 إذا لم يكن متاحًا"))
        hba1c_mean = c2.number_input(L("HbA1c (%)", "الهيموغلوبين السكري"), 0.0, 15.0, 0.0, step=0.1, help=L("Leave 0 if not available", "اترك 0 إذا لم يكن متاحًا"))
        potassium_mean = c3.number_input(L("Potassium (mmol/L)", "البوتاسيوم"), 0.0, 10.0, 0.0, step=0.1, help=L("Leave 0 if not available", "اترك 0 إذا لم يكن متاحًا"))

    # Medications (Optional)
    with st.expander(L("💊 Medications (Optional)", "💊 الأدوية (اختياري)"), expanded=False):
        c1, c2 = st.columns(2)
        ferric_times = c1.number_input(L("Ferric carboxymaltose (times)", "حقن الحديد (عدد المرات)"), 0, 20, 0, help=L("Number of times prescribed", "عدد مرات الوصف"))
        metoprolol_times = c2.number_input(L("Metoprolol (times)", "ميتوبرولول (عدد المرات)"), 0, 50, 0, help=L("Number of times prescribed", "عدد مرات الوصف"))

    submitted = st.form_submit_button(L("Evaluate", "تقييم"), use_container_width=True)

# =============================
# Result (3-level gauge + PDF)
# =============================
if submitted:
    # Validate Patient ID and Name
    if not patient_id or not patient_id.strip():
        st.error(L(
            "❌ **Patient ID is required!** Please enter a Patient ID before evaluating.",
            "❌ **رقم المريضة مطلوب!** الرجاء إدخال رقم المريضة قبل التقييم."
        ))
        st.stop()
    
    if not patient_name or not patient_name.strip():
        st.error(L(
            "❌ **Patient Name is required!** Please enter a Patient Name before evaluating.",
            "❌ **اسم المريضة مطلوب!** الرجاء إدخال اسم المريضة قبل التقييم."
        ))
        st.stop()
    
    def yn2en(x):
        return "yes" if (AR and x == "نعم") else ("no" if (AR and x == "لا") else x)

    # Prepare input dictionary for XGBoost model
    user_input = {
        "gestational_weeks": gestational_weeks,
        "babyweight": babyweight,
        "prenatal_visits": prenatal_visits,
        "total_emergency_visits": emergency_visits,
        "height": height,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "diabetes": yn2en(diabetes),
        "hypertension": yn2en(hypertension),
        "creatinine_mean": creatinine_mean,
        "hba1c_mean": hba1c_mean,
        "potassium_mean": potassium_mean,
        "ferric_carboxymaltose_times": ferric_times,
        "metoprolol_times": metoprolol_times,
        "total_inpatient_visits": inpatient_visits,
        "twins": twins_val,
        "deliverytype": deliverytype_val,
        "year": year
    }
    
    # Get prediction from XGBoost model
    try:
        prediction = preprocessing.predict_risk(user_input)
        pct = prediction['risk_percentage']
        band_text = L(prediction['risk_level'], 
                     "منخفض" if prediction['risk_level'] == "Low" else 
                     ("متوسط" if prediction['risk_level'] == "Moderate" else "مرتفع"))
        badge_code = prediction['risk_band']
    except Exception as e:
        st.error(L(f"Error loading model: {str(e)}", f"خطأ في تحميل النموذج: {str(e)}"))
        st.stop()
    
    # For backward compatibility with PDF generation
    d = {
        "maternal_age": 0,  # Not used in new model
        "gestational_weeks": gestational_weeks,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "prenatal_visits": prenatal_visits,
        "prev_stillbirth": "no",  # Not used in new model
        "diabetes": yn2en(diabetes),
        "hypertension": yn2en(hypertension),
        "smoker": "no"  # Not used in new model
    }
    
    # Try to use LLM for intelligent explanation, fallback to rule-based
    try:
        bullets = openrouter_explain_risk(band_text, pct, user_input, AR)
    except Exception as e:
        # Use fallback explanation
        bullets = explanation_for_band(d, band_text, user_input)

    st.markdown(f"### {L('Risk Assessment', 'تقييم الخطورة')}", unsafe_allow_html=True)
    range_txt = L("Bands: Low 0–33 • Moderate 34–66 • High 67–100", "المستويات: منخفض ٠–٣٣ • متوسط ٣٤–٦٦ • مرتفع ٦٧–١٠٠")
    st.markdown(
        f'<div class="lab-wrap"><div class="lab-head"><div class="lab-name">{L("Risk Index", "مؤشر الخطورة")}</div>'
        f'<div class="lab-ref"><div style="text-align:right"><div style="font-size:1.25rem;font-weight:800">{pct}</div>'
        f'<div>{range_txt}</div></div></div></div>',
        unsafe_allow_html=True
    )
    # For Arabic (RTL), reverse the gauge colors and labels
    if AR:
        # Arabic: High (red) on left, Moderate (yellow) in center, Low (green) on right
        lbls = ["مرتفع", "متوسط", "منخفض"]
        st.markdown(
            f'<div class="lab-band"><div class="seg green" style="width:33%"></div>'
            f'<div class="seg amber" style="width:34%"></div>'
            f'<div class="seg red" style="width:33%"></div></div>'
            f'<div class="lab-labels"><span>{lbls[2]}</span><span>{lbls[1]}</span><span>{lbls[0]}</span></div>',
            unsafe_allow_html=True
        )
        # Reverse marker position for RTL
        marker_position = max(2, min(98, 100 - pct))
    else:
        # English: Low (green) on left, Moderate (yellow) in center, High (red) on right
        lbls = ["Low", "Moderate", "High"]
        st.markdown(
            f'<div class="lab-band"><div class="seg green" style="width:33%"></div>'
            f'<div class="seg amber" style="width:34%"></div>'
            f'<div class="seg red" style="width:33%"></div></div>'
            f'<div class="lab-labels"><span>{lbls[0]}</span><span>{lbls[1]}</span><span>{lbls[2]}</span></div>',
            unsafe_allow_html=True
        )
        marker_position = max(2, min(98, pct))
    
    st.markdown(
        f'<div class="marker" style="height:38px;"><div class="pin" style="left:{marker_position}%"></div>'
        f'<div class="pill" style="left:{marker_position}%">{pct}</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
    st.markdown(f"**{L('Explanation', 'توضيح')}**")
    for line in bullets:
        st.write(f"- {line}")

    # Save to history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.session_state.history.loc[len(st.session_state.history)] = [
        timestamp, patient_id, patient_name, band_text, pct,
        " | ".join(bullets),
        gestational_weeks, babyweight, bmi, height,
        systolic_bp, diastolic_bp, prenatal_visits,
        emergency_visits, inpatient_visits,
        L("yes", "نعم") if diabetes in [L("yes", "نعم"), "yes", "نعم"] else L("no", "لا"),
        L("yes", "نعم") if hypertension in [L("yes", "نعم"), "yes", "نعم"] else L("no", "لا"),
        twins_val,
        deliverytype_val,
    ]

    # Build and download PDF
    pdf_bytes = build_pdf(
        patient_id=patient_id,
        patient_name=patient_name,
        timestamp=timestamp,
        pct=pct,
        band_text=band_text,
        band_code=badge_code,
        bullets=bullets,
        d=d,
        AR=AR
    )

    st.download_button(
        label=L("⬇️ Download Result (PDF)", "⬇️ تنزيل النتيجة (PDF)"),
        data=pdf_bytes,
        file_name=f"{patient_id}_risk_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# =============================
# Statistics Cards
# =============================
stats = calculate_statistics()
st.markdown(f"<h2 class='section-header'>{L('Case Statistics', 'إحصائيات الحالات')}</h2>", unsafe_allow_html=True)

stats_cols = st.columns(4)
with stats_cols[0]:
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="stat-number">{stats['total_cases']}</span>
            <span class="stat-title">{L('Total Cases', 'إجمالي الحالات')}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with stats_cols[1]:
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="stat-number">{stats['low_risk']}</span>
            <span class="stat-title">{L('Low Risk', 'منخفض الخطورة')}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with stats_cols[2]:
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="stat-number">{stats['moderate_risk']}</span>
            <span class="stat-title">{L('Moderate Risk', 'متوسط الخطورة')}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with stats_cols[3]:
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="stat-number">{stats['high_risk']}</span>
            <span class="stat-title">{L('High Risk', 'مرتفع الخطورة')}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================
# Dashboard Containers with Interactivity
# =============================
st.markdown(f"<h2 class='section-header'>{L('Clinical Dashboard', 'لوحة التحكم السريرية')}</h2>", unsafe_allow_html=True)

# Initialize session state for dashboard sections
if "show_labs" not in st.session_state:
    st.session_state.show_labs = False
if "show_meds" not in st.session_state:
    st.session_state.show_meds = False
if "show_records" not in st.session_state:
    st.session_state.show_records = False

dashboard_cols = st.columns(3)

with dashboard_cols[0]:
    if st.button(
        L('🔬\n\nLaboratory Tests\n\nView and manage patient lab results', 
          '🔬\n\nالتحاليل المخبرية\n\nعرض وإدارة نتائج مختبر المريضة'),
        key="lab_btn",
        use_container_width=True
    ):
        st.session_state.show_labs = not st.session_state.show_labs
        st.session_state.show_meds = False
        st.session_state.show_records = False

with dashboard_cols[1]:
    if st.button(
        L('💊\n\nMedications\n\nManage prescribed medications and dosage', 
          '💊\n\nالأدوية\n\nإدارة الأدوية الموصوفة والجرعات'),
        key="med_btn",
        use_container_width=True
    ):
        st.session_state.show_meds = not st.session_state.show_meds
        st.session_state.show_labs = False
        st.session_state.show_records = False

with dashboard_cols[2]:
    if st.button(
        L('📋\n\nMedical Records\n\nAccess complete patient medical history', 
          '📋\n\nالسجل الطبي\n\nالوصول إلى السجل الطبي الكامل للمريضة'),
        key="rec_btn",
        use_container_width=True
    ):
        st.session_state.show_records = not st.session_state.show_records
        st.session_state.show_labs = False
        st.session_state.show_meds = False

# Display content based on selection
if st.session_state.show_labs:
    with st.expander(L("🔬 Laboratory Tests Details", "🔬 تفاصيل التحاليل المخبرية"), expanded=True):
        st.markdown(f"### {L('Common Prenatal Laboratory Tests', 'التحاليل المخبرية الشائعة قبل الولادة')}")
        
        # Sample lab data
        lab_data = {
            L("Test Name", "اسم التحليل"): [
                L("Hemoglobin (Hb)", "الهيموجلوبين"),
                L("Blood Glucose", "سكر الدم"),
                L("Blood Pressure", "ضغط الدم"),
                L("Urine Protein", "بروتين البول"),
                L("Platelets", "الصفائح الدموية")
            ],
            L("Normal Range", "المعدل الطبيعي"): [
                "12-16 g/dL",
                "70-100 mg/dL",
                "90-120/60-80 mmHg",
                L("Negative", "سلبي"),
                "150-400 × 10³/µL"
            ],
            L("Status", "الحالة"): [
                L("Normal", "طبيعي"),
                L("Normal", "طبيعي"),
                L("Normal", "طبيعي"),
                L("Normal", "طبيعي"),
                L("Normal", "طبيعي")
            ]
        }
        
        df_labs = pd.DataFrame(lab_data)
        st.dataframe(df_labs, use_container_width=True, hide_index=True)
        
        st.info(L(
            "💡 **Note:** These are sample values. Actual patient lab results would be displayed here in a production environment.",
            "💡 **ملاحظة:** هذه قيم تجريبية. سيتم عرض نتائج المختبر الفعلية للمريضة هنا في بيئة الإنتاج."
        ))

if st.session_state.show_meds:
    with st.expander(L("💊 Medications Details", "💊 تفاصيل الأدوية"), expanded=True):
        st.markdown(f"### {L('Prescribed Medications', 'الأدوية الموصوفة')}")
        
        # Sample medication data
        med_data = {
            L("Medication", "الدواء"): [
                L("Prenatal Vitamins", "فيتامينات ما قبل الولادة"),
                L("Folic Acid", "حمض الفوليك"),
                L("Iron Supplement", "مكملات الحديد"),
                L("Calcium", "الكالسيوم")
            ],
            L("Dosage", "الجرعة"): [
                L("1 tablet daily", "قرص واحد يومياً"),
                "400 mcg " + L("daily", "يومياً"),
                "30 mg " + L("daily", "يومياً"),
                "1000 mg " + L("daily", "يومياً")
            ],
            L("Frequency", "التكرار"): [
                L("Once daily", "مرة يومياً"),
                L("Once daily", "مرة يومياً"),
                L("Once daily", "مرة يومياً"),
                L("Twice daily", "مرتين يومياً")
            ],
            L("Duration", "المدة"): [
                L("Throughout pregnancy", "طوال فترة الحمل"),
                L("First trimester", "الثلث الأول"),
                L("Throughout pregnancy", "طوال فترة الحمل"),
                L("Throughout pregnancy", "طوال فترة الحمل")
            ]
        }
        
        df_meds = pd.DataFrame(med_data)
        st.dataframe(df_meds, use_container_width=True, hide_index=True)
        
        st.info(L(
            "💡 **Note:** These are sample medications. Actual patient prescriptions would be displayed here in a production environment.",
            "💡 **ملاحظة:** هذه أدوية تجريبية. سيتم عرض الوصفات الطبية الفعلية للمريضة هنا في بيئة الإنتاج."
        ))

if st.session_state.show_records:
    with st.expander(L("📋 Medical Records Details", "📋 تفاصيل السجل الطبي"), expanded=True):
        st.markdown(f"### {L('Patient Medical History', 'السجل الطبي للمريضة')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{L('Personal Information', 'المعلومات الشخصية')}**")
            st.write(f"• {L('Age', 'العمر')}: 28 {L('years', 'سنة')}")
            st.write(f"• {L('Blood Type', 'فصيلة الدم')}: O+")
            st.write(f"• {L('Allergies', 'الحساسية')}: {L('None reported', 'لا يوجد')}")
            
            st.markdown(f"**{L('Previous Pregnancies', 'الحمل السابق')}**")
            st.write(f"• {L('Gravida', 'الحمل')}: 2")
            st.write(f"• {L('Para', 'الولادة')}: 1")
            st.write(f"• {L('Abortions', 'الإجهاض')}: 0")
        
        with col2:
            st.markdown(f"**{L('Chronic Conditions', 'الأمراض المزمنة')}**")
            st.write(f"• {L('Diabetes', 'السكري')}: {L('No', 'لا')}")
            st.write(f"• {L('Hypertension', 'ارتفاع ضغط الدم')}: {L('No', 'لا')}")
            st.write(f"• {L('Heart Disease', 'أمراض القلب')}: {L('No', 'لا')}")
            
            st.markdown(f"**{L('Recent Visits', 'الزيارات الأخيرة')}**")
            st.write(f"• {L('Last Visit', 'آخر زيارة')}: {L('2 weeks ago', 'منذ أسبوعين')}")
            st.write(f"• {L('Next Appointment', 'الموعد القادم')}: {L('1 week', 'أسبوع واحد')}")
        
        st.markdown("---")
        
        st.markdown(f"**{L('Notes', 'الملاحظات')}**")
        st.info(L(
            "Patient is in good general health. Regular prenatal checkups recommended. Continue current medication regimen.",
            "المريضة في صحة عامة جيدة. يُنصح بإجراء فحوصات منتظمة قبل الولادة. الاستمرار في نظام الدواء الحالي."
        ))
        
        st.info(L(
            "💡 **Note:** This is sample medical record data. Actual patient records would be displayed here in a production environment.",
            "💡 **ملاحظة:** هذا سجل طبي تجريبي. سيتم عرض السجل الطبي الفعلي للمريضة هنا في بيئة الإنتاج."
        ))

# =============================
# History (styled container)
# =============================
st.markdown(f"<h2 class='section-header'>{L('Patient History', 'سجل الحالات')}</h2>", unsafe_allow_html=True)
q = st.text_input(L("Search (ID/Name)", "بحث (رقم/اسم)"), key="hist_q")
df = st.session_state.history.copy()
if q:
    df = df[df["patient_name"].str.contains(q, case=False, na=False) | df["patient_id"].str.contains(q, case=False, na=False)]

# Display the history table
st.dataframe(
    df[["timestamp", "patient_id", "patient_name", "risk_level", "score_pct", "explanation"]]
    .sort_values("timestamp", ascending=False),
    use_container_width=True
)

# CSV Export
col_map_en = {
    "timestamp": "timestamp", "patient_id": "patient_id", "patient_name": "patient_name",
    "risk_level": "risk_level", "score_pct": "score_pct", "explanation": "explanation",
    "gestational_weeks": "gestational_weeks", "babyweight": "babyweight", "bmi": "bmi", "height": "height",
    "systolic_bp": "systolic_bp", "diastolic_bp": "diastolic_bp", "prenatal_visits": "prenatal_visits",
    "emergency_visits": "emergency_visits", "inpatient_visits": "inpatient_visits",
    "diabetes": "diabetes", "hypertension": "hypertension", "twins": "twins", "deliverytype": "deliverytype"
}

col_map_ar = {
    "timestamp": "التاريخ", "patient_id": "رقم المريضة", "patient_name": "اسم المريضة",
    "risk_level": "مستوى الخطورة", "score_pct": "المؤشر", "explanation": "توضيح",
    "gestational_weeks": "عمر الحمل (أسابيع)", "babyweight": "وزن الطفل", "bmi": "مؤشر كتلة الجسم", "height": "الطول",
    "systolic_bp": "الضغط الانقباضي", "diastolic_bp": "الضغط الانبساطي", "prenatal_visits": "زيارات قبل الولادة",
    "emergency_visits": "زيارات الطوارئ", "inpatient_visits": "الزيارات الداخلية",
    "diabetes": "سكري", "hypertension": "ارتفاع ضغط", "twins": "توأم", "deliverytype": "نوع الولادة"
}

col_map = col_map_ar if AR else col_map_en

# Ensure the DataFrame has the expected columns
expected_columns = [
    "timestamp", "patient_id", "patient_name", "risk_level", "score_pct", "explanation",
    "gestational_weeks", "babyweight", "bmi", "height", "systolic_bp", "diastolic_bp", "prenatal_visits",
    "emergency_visits", "inpatient_visits", "diabetes", "hypertension", "twins", "deliverytype"
]

# Create a copy of the DataFrame with only the expected columns
df_renamed = st.session_state.history[expected_columns].copy()

# Rename the columns
df_renamed = df_renamed.rename(columns=col_map)

# Write to BytesIO with UTF-8 encoding
csv_bytes = io.BytesIO()
df_renamed.to_csv(csv_bytes, index=False, encoding="utf-8-sig")
csv_bytes.seek(0)

st.markdown('<div class="download-history-btn">', unsafe_allow_html=True)
st.download_button(
    label=L("⬇️ Download history (CSV)", "⬇️ تنزيل السجل (CSV)"),
    data=csv_bytes,
    file_name=L("patient_history.csv", "سجل_الحالات.csv"),
    mime="text/csv",
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="small">{L("Results are stored temporarily. Export CSV to keep them.", "النتائج تُحفظ مؤقتًا. صدّر CSV للاحتفاظ بها.")}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
