import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a telecom customer will churn based on account details.")

st.divider()

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("models/churn_model.pkl")

# -------------------------------
# USER INPUT SECTION
# -------------------------------
st.subheader("🔢 Enter Customer Details")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=800.0)

# -------------------------------
# PREDICTION SECTION
# -------------------------------
if st.button("Predict Churn"):

    input_data = pd.DataFrame([[tenure, monthly, total]],
                              columns=['tenure', 'MonthlyCharges', 'TotalCharges'])

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer Will Churn")
        st.write("High risk customer. Retention strategy recommended.")
    else:
        st.success("✅ Customer Will Stay")
        st.write("Customer likely to remain with company.")

# -------------------------------
# ANALYTICS SECTION
# -------------------------------
st.divider()
st.subheader("📊 Monthly Charges vs Churn Analysis")

# Load dataset
df = pd.read_csv("data/telco_churn.csv")

# Convert Churn column Yes/No → 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Fix TotalCharges column (sometimes string)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df = df.dropna()

# Plot graph
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

st.caption("Observation: Customers with higher monthly charges show higher churn tendency.")