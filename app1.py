import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Credit Card Fraud Detection")
st.write("Enter transaction data to check for fraud.")

# Input fields
v_features = [f'V{i}' for i in range(1, 29)]
inputs = {}
for v in v_features:
    inputs[v] = st.number_input(v, value=0.0)

amount = st.number_input("Amount", value=0.0)
inputs["normAmount"] = (amount - 88.35) / 250.12  # Approx mean/std used

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# Predict
if st.button("Predict"):
    result = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]
    if result[0] == 1:
        st.error(f"🚨 Fraud Detected with {prob*100:.2f}% confidence")
    else:
        st.success(f"✅ Transaction is Legitimate with {100 - prob*100:.2f}% confidence")
