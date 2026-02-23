import sys
import os

# Get project root (Brain_MRI folder)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import re
import plotly.express as px
import plotly.graph_objects as go
from jose import jwt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils.gradcam_utils import get_gradcam_heatmap, overlay_heatmap
import cv2

from utils.gradcam_utils import get_gradcam_heatmap, overlay_heatmap

# ---------------- CONFIG ---------------- #
BACKEND_URL = "http://127.0.0.1:8000"
SECRET_KEY = "supersecretkey123"
ALGORITHM = "HS256"

st.set_page_config(
    page_title="AI Brain Tumor Clinical System",
    page_icon="🧠",
    layout="wide"
)

# ---------------- REMOVE STREAMLIT DEFAULT HEADER ---------------- #
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
.block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- MODERN UI ---------------- #
def load_ui():
    st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    color: white !important;
}

.main {
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.glass {
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 25px;
    color: white !important;
}

.stTextInput>div>div>input {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

load_ui()

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_ai_model():
    return load_model("../model/brain_tumor_model.h5")

model = load_ai_model()

# ---------------- TRUST FUNCTION ---------------- #
def calculate_time_to_trust(confidence):
    trust_score = abs(confidence - 0.5) * 2
    trust_score = min(max(trust_score, 0), 1)
    return 1 - trust_score, trust_score

# ---------------- AI EXPLANATION ---------------- #
def generate_ai_explanation(tumor_prob, trust_score):
    if tumor_prob > 0.75:
        severity = "high likelihood"
    elif tumor_prob > 0.55:
        severity = "moderate likelihood"
    else:
        severity = "low likelihood"

    confidence_level = "strong" if trust_score > 0.7 else "moderate" if trust_score > 0.4 else "low"

    explanation = f"""
    🧠 AI Clinical Interpretation:

    The model indicates a {severity} of tumor presence.
    The prediction confidence is {confidence_level}, based on internal probability calibration.

    Trust Score: {trust_score:.2f}

    This AI output should assist clinical decision-making but must be validated by a certified radiologist.
    """

    return explanation

# ---------------- PHONE VALIDATION ---------------- #
def is_valid_phone(phone):
    return re.fullmatch(r"[6-9]\d{9}", phone) is not None

# ---------------- SESSION ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.role = None

# ---------------- LOGIN ---------------- #
def login():
    st.markdown("<h1 style='text-align:center;'>🏥 Brain MRI Clinical AI</h1>", unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        response = requests.post(
            f"{BACKEND_URL}/login",
            data={"username": username, "password": password}
        )

        if response.status_code == 200:
            token = response.json()["access_token"]
            decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            st.session_state.logged_in = True
            st.session_state.token = token
            st.session_state.username = decoded.get("sub")
            st.session_state.role = decoded.get("role")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- NAVBAR ---------------- #
col1, col2 = st.columns([8,2])
with col1:
    st.markdown("### 🧠 Brain MRI Clinical Dashboard")
with col2:
    st.markdown(f"<div class='role'>{st.session_state.role.upper()}</div>", unsafe_allow_html=True)

# ---------------- TABS ---------------- #
tabs = st.tabs(["🧠 Diagnosis", "📋 My Patients", "📊 Analytics", "⚙ Settings"])

# ==============================
# 🧠 DIAGNOSIS
# ==============================
with tabs[0]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

    colA, colB = st.columns(2)
    with colA:
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Age", min_value=1, max_value=120)
    with colB:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        patient_phone = st.text_input("Phone Number")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True)

        img_resized = img.resize((224,224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        raw_output = model.predict(img_array)[0][0]
        tumor_prob = float(raw_output)
        no_tumor_prob = 1 - tumor_prob

        result = "Tumor Detected" if tumor_prob > 0.5 else "No Tumor Detected"
        _, trust_score = calculate_time_to_trust(tumor_prob)

        st.success(result)

        col1, col2, col3 = st.columns(3)
        col1.metric("Tumor Probability", f"{tumor_prob:.4f}")
        col2.metric("No Tumor Probability", f"{no_tumor_prob:.4f}")
        col3.metric("Trust Score", f"{trust_score:.4f}")

        # ---------------- GRAD-CAM ---------------- #
        heatmap = get_gradcam_heatmap(model, img_array)
        original_img = np.array(img_resized)
        overlay_img = overlay_heatmap(original_img, heatmap)
        st.subheader("🔥 Grad-CAM Visualization")

        c1, c2 = st.columns(2)
        c1.image(original_img, width=300)
        c2.image(overlay_img, width=300)

        # -------- AI Explanation -------- #
        explanation = generate_ai_explanation(tumor_prob, trust_score)
        st.markdown("### 🧠 AI Clinical Interpretation")
        st.info(explanation)

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=tumor_prob,
            title={'text': "Tumor Probability"},
            gauge={'axis': {'range': [0, 1]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("💾 Save Patient"):
            if not patient_name or not is_valid_phone(patient_phone):
                st.error("Complete all fields correctly.")
            else:
                payload = {
                    "patient_name": patient_name,
                    "age": int(patient_age),
                    "gender": patient_gender,
                    "phone": patient_phone,
                    "prediction": result,
                    "tumor_probability": tumor_prob,
                    "trust_score": trust_score
                }

                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.post(
                    f"{BACKEND_URL}/save_patient",
                    json=payload,
                    headers=headers
                )

                if response.status_code == 200:
                    st.success("Patient saved successfully.")
                else:
                    st.error("Server error.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# 📋 MY PATIENTS
# ==============================
with tabs[1]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = requests.get(f"{BACKEND_URL}/my_patients", headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data:
            st.dataframe(data, use_container_width=True)
        else:
            st.info("No patient records.")
    else:
        st.error("Unable to fetch records.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# 📊 ANALYTICS
# ==============================
with tabs[2]:
    if st.session_state.role != "admin":
        st.warning("Admin only access")
    else:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(f"{BACKEND_URL}/all_patients", headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data:
                tumor_count = sum(d["prediction"]=="Tumor Detected" for d in data)
                no_tumor_count = len(data) - tumor_count

                fig = px.pie(
                    names=["Tumor", "No Tumor"],
                    values=[tumor_count, no_tumor_count],
                    title="Tumor Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Total Patients", len(data))
            else:
                st.info("No data available")

        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# ⚙ SETTINGS
# ==============================
with tabs[3]:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)