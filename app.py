import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Telecom Customer Churn Prediction")
st.write("Predict whether a telecom customer will churn using Machine Learning.")

st.metric("Model Accuracy","74%")
st.metric("ROC AUC Score","0.84")

st.divider()

# CUSTOMER INFORMATION
st.header("Customer Information")

gender = st.selectbox("Gender", ["Male","Female"])
senior = st.selectbox("Senior Citizen", ["No","Yes"])
partner = st.selectbox("Partner", ["No","Yes"])
dependents = st.selectbox("Dependents", ["No","Yes"])

# SERVICE DETAILS
st.header("Service Details")

internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
online_security = st.selectbox("Online Security", ["No","Yes"])
tech_support = st.selectbox("Tech Support", ["No","Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No","Yes"])

# BILLING DETAILS
st.header("Billing Details")

contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])

payment = st.selectbox(
"Payment Method",
[
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
]
)

tenure = st.slider("Tenure (months)",0,72,12)
monthly_charges = st.number_input("Monthly Charges",0.0,200.0,50.0)
total_charges = st.number_input("Total Charges",0.0,10000.0,500.0)

st.divider()

# FEATURES USED BY MODEL
MODEL_FEATURES = [
"SeniorCitizen",
"tenure",
"MonthlyCharges",
"TotalCharges",
"gender_Male",
"Partner_Yes",
"Dependents_Yes",
"InternetService_Fiber optic",
"InternetService_No",
"OnlineSecurity_Yes",
"TechSupport_Yes",
"StreamingTV_Yes",
"Contract_One year",
"Contract_Two year",
"PaymentMethod_Electronic check",
"PaymentMethod_Mailed check",
"PaymentMethod_Credit card (automatic)"
]

# PREDICTION
if st.button("Predict Churn"):

    data = {
    "SeniorCitizen":1 if senior=="Yes" else 0,
    "tenure":tenure,
    "MonthlyCharges":monthly_charges,
    "TotalCharges":total_charges,
    "gender_Male":1 if gender=="Male" else 0,
    "Partner_Yes":1 if partner=="Yes" else 0,
    "Dependents_Yes":1 if dependents=="Yes" else 0,
    "InternetService_Fiber optic":1 if internet=="Fiber optic" else 0,
    "InternetService_No":1 if internet=="No" else 0,
    "OnlineSecurity_Yes":1 if online_security=="Yes" else 0,
    "TechSupport_Yes":1 if tech_support=="Yes" else 0,
    "StreamingTV_Yes":1 if streaming_tv=="Yes" else 0,
    "Contract_One year":1 if contract=="One year" else 0,
    "Contract_Two year":1 if contract=="Two year" else 0,
    "PaymentMethod_Electronic check":1 if payment=="Electronic check" else 0,
    "PaymentMethod_Mailed check":1 if payment=="Mailed check" else 0,
    "PaymentMethod_Credit card (automatic)":1 if payment=="Credit card (automatic)" else 0
    }

 df = pd.DataFrame([data])

for col in MODEL_FEATURES:
    if col not in df.columns:
        df[col] = 0

df = df[MODEL_FEATURES]

# Predict 
pred = model.predict(df)
prob = model.predict_proba(df)

    churn_prob = prob[0][1]*100

    st.subheader("Prediction Result")

    if pred[0]==1:
        st.error(f"Customer likely to churn ({churn_prob:.2f}% risk)")
    else:
        st.success(f"Customer likely to stay ({100-churn_prob:.2f}% retention probability)")

    st.subheader("Churn Probability")

    chart = pd.DataFrame({
    "Status":["Stay","Churn"],
    "Probability":[100-churn_prob,churn_prob]
    })

    st.bar_chart(chart.set_index("Status"))