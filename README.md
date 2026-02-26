# 📊 Customer Churn Prediction System

An end-to-end **Customer Churn Prediction System** built using Python, Machine Learning, and Streamlit.  
The project predicts telecom customer churn and provides interactive analytics to identify high-risk customers.

---

## 🚀 Features

- ML-based churn prediction using Random Forest Classifier  
- Interactive **Streamlit dashboard** for real-time prediction  
- Displays **churn probability** and retention likelihood  
- Behavioral analysis: Monthly charges vs churn  
- Feature importance visualization to identify key drivers of churn  
- Cleaned real-world telecom dataset (7,000+ customers)  

---

## 📂 Project Structure
customer-churn-prediction-system/
│
├── streamlit_app.py # Main Streamlit application
├── requirements.txt # Dependencies
├── README.md # This file
├── models/
│ └── churn_model_small.pkl # Compressed ML model
└── data/
└── telco_churn.csv # Dataset

---

## 🧠 Model Details

- Algorithm: Random Forest Classifier  
- Accuracy: ~75–85% depending on features  
- Class imbalance handled with stratified sampling & class weighting  

---

## 🎨 Analytics & Insights

- Customers with **higher monthly charges** are more likely to churn  
- Short-tenure customers are at higher risk  
- Most influential feature: **Monthly Charges**  

---

## ⚙️ Installation & Run Locally

Clone the repo:

```bash
git clone https://github.com/AtharvKadam1858/customer-churn-prediction-system.git
cd customer-churn-prediction-system
Install dependencies:

pip install -r requirements.txt
```
Run app:

streamlit run streamlit_app.py
🌐 Live Demo

Check the live app here: Streamlit App

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Streamlit

Matplotlib

📎 Author

Atharv Kadam
B.Tech CSE (AI & ML)

🔗 GitHub Repository

https://github.com/AtharvKadam1858/customer-churn-prediction-system.git
