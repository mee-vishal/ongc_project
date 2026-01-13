import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# -----------------------------
# Load trained ROP model
# -----------------------------
with open("rf_model_depth_global.pkl", "rb") as f:
    rop_model = pickle.load(f)

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Well Performance Hub", layout="centered")

# -----------------------------
# Header with ONGC logo
# -----------------------------
col1, col2 = st.columns([1, 4])

with col1:
    logo = Image.open("on1362o755-ongc-logo-ongc-pixelmate-expo (1).png")
    st.image(logo, width=90)

with col2:
    st.markdown(
        """
        <h3 style='margin-bottom:0;'>Oil and Natural Gas Corporation Limited</h3>
        <p style='margin-top:0; color:gray;'>Well Performance Analytics Platform</p>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------
# Title
# -----------------------------
st.title("🛢️ Well Performance Hub – ROP Advisory System")
st.write("Machine-learning based drilling performance prediction")

# =========================================================
# ROP Prediction Section
# =========================================================
st.subheader("🔧 Predict ROP from Drilling Parameters")

depth = st.number_input("Depth (m)", 0.0, 10000.0, 2500.0, 1.0)
wob = st.number_input("WOB (klb)", 0.0, 50.0, 5.0, 0.1)
rpm = st.number_input("Rotary RPM", 0.0, 200.0, 75.0, 0.1)
flow = st.number_input("Flow In Rate (galUS/min)", 0.0, 2000.0, 600.0, 1.0)

if st.button("📈 Predict ROP"):
    X_input = pd.DataFrame({
        "Depth": [depth],
        "WOB": [wob],
        "RPM": [rpm],
        "Flow": [flow]
    })

    rop_pred = rop_model.predict(X_input)[0]

    st.success(f"✅ Predicted ROP: **{rop_pred:.2f} m/hr**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>For academic & research use | ONGC Well Performance Analytics</p>",
    unsafe_allow_html=True
)
