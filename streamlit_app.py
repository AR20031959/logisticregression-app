
# ============================================================
# streamlit_app.py
# Streamlit UI for Titanic Survival Prediction
# ============================================================

import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger would survive the Titanic disaster.")

st.markdown("---")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)
sex = st.selectbox("Gender", ["Male", "Female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encoding
sex_male = 1 if sex == "Male" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

features = np.array([[
    pclass, age, sibsp, parch, fare,
    sex_male, embarked_q, embarked_s
]])

if st.button("Predict Survival"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.success(f"✅ Passenger Survived (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Passenger Did Not Survive (Probability: {probability:.2f})")

st.markdown("---")
st.caption("Model: Logistic Regression | Dataset: Titanic")
