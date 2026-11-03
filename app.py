
import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
with open("heart_disease_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's medical information below.")
st.info(
    "Disclaimer: This application is an educational Machine Learning "
    "demonstration and is not a substitute for professional medical "
    "diagnosis or advice."
)
# Patient information
st.subheader("Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex_option = st.selectbox("Sex", ["Female", "Male"])

sex = 0 if sex_option == "Female" else 1
# Medical information
st.subheader("Medical Information")

cp = st.number_input("Chest Pain Type (cp)", min_value=0, step=1, value=0)
trestbps = st.number_input("Resting Blood Pressure", min_value=0, step=1, value=120)
chol = st.number_input("Cholesterol", min_value=0, step=1, value=200)
fbs = st.number_input("Fasting Blood Sugar (fbs)", min_value=0, step=1, value=0)
restecg = st.number_input("Resting ECG (restecg)", min_value=0, step=1, value=0)
thalach = st.number_input("Maximum Heart Rate", min_value=0, step=1, value=150)
exang = st.number_input("Exercise Induced Angina (exang)", min_value=0, step=1, value=0)
oldpeak = st.number_input("Oldpeak", min_value=0.0, step=0.1, value=0.0)
slope = st.number_input("Slope", min_value=0, step=1, value=0)
ca = st.number_input("Number of Major Vessels (ca)", min_value=0, step=1, value=0)
thal = st.number_input("Thal", min_value=0, step=1, value=0)

# Prediction
if st.button("Predict"):

    user_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    prediction = model.predict(user_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
