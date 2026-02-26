import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a telecom customer will churn based on account details.")
st.divider()

# -------------------------------
# Define Base Directory
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Load Model Safely
# -------------------------------
model_path = os.path.join(BASE_DIR, "models", "churn_model_small.pkl")
model = joblib.load(model_path)

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔢 Enter Customer Details")
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=800.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):
    input_data = pd.DataFrame([[tenure, monthly, total]],
                              columns=['tenure', 'MonthlyCharges', 'TotalCharges'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.divider()
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Customer Will Churn ({probability:.2f}% risk)")
    else:
        st.success(f"✅ Customer Will Stay ({100-probability:.2f}% retention likelihood)")
    st.progress(int(probability))

# -------------------------------
# Load Dataset Safely
# -------------------------------
try:
    csv_path = os.path.join(BASE_DIR, "data", "telco_churn.csv")
    df = pd.read_csv(csv_path)

    # Fix churn column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Fix TotalCharges column
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # -------------------------------
    # Monthly Charges vs Churn Chart
    # -------------------------------
    st.divider()
    st.subheader("📊 Monthly Charges vs Churn Analysis")

    fig, ax = plt.subplots()
    stayed = df[df["Churn"] == 0]["MonthlyCharges"]
    churned = df[df["Churn"] == 1]["MonthlyCharges"]
    ax.hist(stayed, alpha=0.6, label="Stayed")
    ax.hist(churned, alpha=0.6, label="Churned")
    ax.set_title("Monthly Charges Distribution by Churn Status")
    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Number of Customers")
    ax.legend()
    st.pyplot(fig)
    st.caption("Observation: Higher monthly charges show increased churn tendency.")

except FileNotFoundError:
    st.error("⚠️ Dataset file not found. Please upload 'telco_churn.csv' in the data folder of your repo.")