import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Load trained model
# -----------------------------
with open("rf_model_global.pkl", "rb") as f:
    rf_model = pickle.load(f)

# -----------------------------
# Column mapping
# -----------------------------
column_map = {
    "Bit Weight(klb)": "WOB",
    "Rotary RPM(RPM)": "RPM",
    "Flow In Rate(galUS/min)": "Flow"
}

# -----------------------------
# Page config
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
st.title("🛢️ Well Performance Hub – ROP Analysis")
st.write("Analyze Rate of Penetration (ROP) using drilling parameters")

# -----------------------------
# Input section (NUMBER INPUTS)
# -----------------------------
st.subheader("🔧 Input Drilling Parameters")

wob = st.number_input(
    "WOB (Bit Weight in klb)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.1
)

rpm = st.number_input(
    "Rotary RPM",
    min_value=0.0,
    max_value=200.0,
    value=75.0,
    step=0.1
)

flow = st.number_input(
    "Flow In Rate (galUS/min)",
    min_value=0.0,
    max_value=2000.0,
    value=600.0,
    step=1.0
)

# -----------------------------
# Predict button
# -----------------------------
if st.button("📈 Analyze Well Performance"):

    # Prepare input
    test_df = pd.DataFrame({
        "Bit Weight(klb)": [wob],
        "Rotary RPM(RPM)": [rpm],
        "Flow In Rate(galUS/min)": [flow]
    })

    test_df = test_df.rename(columns=column_map)

    # Predict
    rop_pred = rf_model.predict(test_df)[0]

    st.success(f"✅ Predicted ROP: **{rop_pred:.2f} m/hr**")

    # -----------------------------
    # Graphs
    # -----------------------------
    st.subheader("📊 Performance Visualization")

    # Graph 1: ROP vs Data Point Index
    fig1, ax1 = plt.subplots()
    ax1.scatter([1], [rop_pred], s=80)
    ax1.set_xlabel("Data Point Index")
    ax1.set_ylabel("ROP (m/hr)")
    ax1.set_title("ROP vs Data Point")
    st.pyplot(fig1)

    # Graph 2: Actual vs Predicted (single-point dot)
    fig2, ax2 = plt.subplots()
    ax2.scatter(["Predicted ROP"], [rop_pred], s=80)
    ax2.set_ylabel("ROP (m/hr)")
    ax2.set_title("Actual vs Predicted ROP")
    st.pyplot(fig2)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>For academic & research use | Oil & Gas Analytics</p>",
    unsafe_allow_html=True
)
