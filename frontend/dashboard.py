import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from gradcam_utils import get_gradcam_heatmap, calculate_time_to_trust
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Brain Tumor Clinical Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------- SIMPLE USER DATABASE ---------------- #
# (For demo purpose — later we can connect MongoDB)
USERS = {
    "doctor1": "1234",
    "admin": "admin123"
}

# ---------------- SESSION STATE INIT ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# ---------------- LOGIN FUNCTION ---------------- #
def login():
    st.title("🏥 Hospital AI Login Portal")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# ---------------- LOGOUT FUNCTION ---------------- #
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

# ---------------- IF NOT LOGGED IN ---------------- #
if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {
    background-color: #f4f8fb;
}
h1, h2, h3 {
    color: #0b3d91;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.tumor {
    background-color: #ff4b4b;
    color: white;
}
.no-tumor {
    background-color: #2ecc71;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🏥 Hospital AI System")
st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
st.sidebar.button("Logout", on_click=logout)

st.sidebar.markdown("""
**Model:** ResNet50  
**Mode:** Clinical Assist  
""")

# ---------------- LOAD MODEL ---------------- #
model = load_model('brain_tumor_model.h5')

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align:center;'>🧠 AI Brain Tumor Clinical Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Secure Hospital Diagnostic System</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader("📤 Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    img = Image.open(uploaded_file)

    with col1:
        st.image(img, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    with st.spinner("🔬 Analyzing MRI Scan..."):
        prediction = model.predict(img_array)[0][0]

    confidence = float(prediction)
    result = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"

    with col2:
        if prediction > 0.5:
            st.markdown(f"<div class='result-box tumor'>{result}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box no-tumor'>{result}</div>", unsafe_allow_html=True)

        st.progress(confidence)
        st.write(f"Confidence: {confidence:.2f}")

    st.markdown("---")

    # Grad-CAM
    st.subheader("📊 Explainable AI (Grad-CAM)")
    heatmap = get_gradcam_heatmap(model, img_array)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_colored, 0.4, 0)
    st.image(overlay, caption="Tumor Region Highlighted", use_column_width=True)

    st.markdown("---")

    # Trust Score
    st.subheader("⏱ AI Trust Assessment")
    time_to_trust, trust_score = calculate_time_to_trust(confidence)

    st.write(f"Time-to-Trust: {time_to_trust:.2f}")
    st.write(f"Trust Score: {trust_score:.2f}")

    if trust_score < 0.5:
        st.warning("⚠ Recommend expert review.")
    else:
        st.success("✅ Suitable for clinical support.")

    st.markdown("---")

    # Save Results
    if st.button("💾 Save Clinical Report"):
        os.makedirs('results', exist_ok=True)
        plt.imsave('results/heatmap.png', overlay)
        with open('results/prediction.txt', 'w') as f:
            f.write(
                f"Doctor: {st.session_state.username}\n"
                f"Prediction: {result}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Time-to-Trust: {time_to_trust:.2f}\n"
                f"Trust Score: {trust_score:.2f}"
            )
        st.success("Clinical report saved.")

st.markdown("---")
st.markdown("<center>© 2026 Secure AI Clinical System</center>", unsafe_allow_html=True)
