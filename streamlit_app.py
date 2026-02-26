import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a telecom customer will churn based on account details.")

st.divider()

# -------------------------------
# Load Model Safely
# -------------------------------
# Get absolute path to the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "churn_model_small.pkl")

# Load the model
model = joblib.load(model_path)

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔢 Enter Customer Details")

tenure = st.number_input("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 800.0)

# -------------------------------
# Prediction with Probability
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
# Load Dataset for Analytics
# -------------------------------
st.divider()
st.subheader("📊 Monthly Charges vs Churn Analysis")

df = pd.read_csv(os.path.join(BASE_DIR, "data", "telco_churn.csv"))

# Convert Churn Yes/No → 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Histogram of Monthly Charges
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
st.caption("Insight: Higher monthly charges correlate with increased churn risk.")

# -------------------------------
# Feature Importance Graph
# -------------------------------
st.divider()
st.subheader("⭐ Feature Importance")

try:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = ['Tenure', 'MonthlyCharges', 'TotalCharges']

        fig2, ax2 = plt.subplots()
        ax2.barh(features, importances)
        ax2.set_title("Model Feature Importance")

        st.pyplot(fig2)

        top_feature = features[np.argmax(importances)]
        st.info(f"Most influential factor: **{top_feature}**")
    else:
        st.warning("Feature importance not available for this model.")
except:
    st.warning("Feature importance could not be generated.")

st.divider()
st.caption("Built using Python, ML, and real telecom customer data.")