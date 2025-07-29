import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

st.title("📰 News Article Classifier")
st.markdown("Enter a news article below and find out if it's **Fake** or **Real**!")

input_text = st.text_area("Breaking news: Government announces big economic reform package.")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Error.")
    else:
        text_vec = vectorizer.transform([input_text])
        prediction = model.predict(text_vec)[0]
        label = "Real ✅" if prediction == 1 else "Fake ❌"
        st.success(f"Prediction: **{label}**")
