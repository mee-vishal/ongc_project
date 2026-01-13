import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# -----------------------------
# Load trained models
# -----------------------------
with open("rf_model_depth_global.pkl", "rb") as f:
    rop_model = pickle.load(f)

with open("rf_model_rop_to_wob.pkl", "rb") as f:
    wob_model = pickle.load(f)

with open("rf_model_rop_to_rpm.pkl", "rb") as f:
    rpm_model = pickle.load(f)

with open("rf_model_rop_to_flow.pkl", "rb") as f:
    flow_model = pickle.load(f)

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
st.write("Forward & inverse machine-learning based drilling performance analysis")

# =========================================================
# SECTION 1: ROP Prediction (Forward Model)
# =========================================================
st.subheader("🔧 Predict ROP from Drilling Parameters")

depth = st.number_input("Depth (m)", 0.0, 10000.0, 2500.0, 1.0)
wob = st.number_input("WOB (klb)", 0.0, 50.0, 5.0, 0.1)
rpm = st.number_input("Rotary RPM", 0.0, 200.0, 75.0, 0.1)
flow = st.number_input("Flow In Rate (galUS/min)", 0.0, 2000.0, 600.0, 1.0)

if st.button("📈 Predict ROP"):
    X_rop = pd.DataFrame({
        "Depth": [depth],
        "WOB": [wob],
        "RPM": [rpm],
        "Flow": [flow]
    })

    rop_pred = rop_model.predict(X_rop)[0]

    st.success(f"✅ Predicted ROP: **{rop_pred:.2f} m/hr**")

st.markdown("---")

# # =========================================================
# # SECTION 2: Parameter Recommendation (Inverse Models)
# # =========================================================
# st.subheader("🎯 Recommend Drilling Parameters for Target ROP")

# depth_inv = st.number_input(
#     "Depth for Recommendation (m)",
#     0.0, 10000.0, 2500.0, 1.0,
#     key="depth_inv"
# )

# target_rop = st.number_input(
#     "Desired ROP (m/hr)",
#     0.1, 100.0, 10.0, 0.1
# )

# if st.button("🧠 Recommend Parameters"):

#     X_inv = pd.DataFrame({
#         "Depth": [depth_inv],
#         "ROP": [target_rop]
#     })

#     rec_wob = wob_model.predict(X_inv)[0]
#     rec_rpm = rpm_model.predict(X_inv)[0]
#     rec_flow = flow_model.predict(X_inv)[0]

#     st.success("✅ Recommended Operating Parameters")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric("WOB (klb)", f"{rec_wob:.2f}")

#     with col2:
#         st.metric("RPM", f"{rec_rpm:.1f}")

#     with col3:
#         st.metric("Flow (galUS/min)", f"{rec_flow:.0f}")

#     st.warning(
#         "⚠️ These are **data-driven recommendations**. "
#         "Final drilling decisions must be validated by drilling engineers."
#     )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>For academic & research use | ONGC Well Performance Analytics</p>",
    unsafe_allow_html=True
)
