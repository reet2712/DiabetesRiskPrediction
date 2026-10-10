
import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and feature names
model = joblib.load("best_diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")
feature_columns = joblib.load("diabetes_feature_columns.pkl")

# Configure the Streamlit page
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# Main heading
st.title("Type 2 Diabetes Risk Prediction")
st.write("Enter the health measurements below to get a model prediction.")

# Create input fields
with st.form("diabetes_form"):

    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=1.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=1.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    insulin = st.number_input(
        "Insulin Level",
        min_value=0.0,
        max_value=900.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        max_value=100.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    submit = st.form_submit_button("Predict")

# Make a prediction when the button is clicked
if submit:

    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    input_df = pd.DataFrame([input_data])

    # Keep features in the same order used during training
    input_df = input_df[feature_columns]

    # Apply the saved scaler
    input_scaled = scaler.transform(input_df)

    # Predict the class
    prediction = model.predict(input_scaled)[0]

    # Display the result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The model predicts a positive diabetes classification.")
    else:
        st.success("The model predicts a negative diabetes classification.")

    st.caption(
        "This is an educational machine-learning prediction, "
        "not a medical diagnosis. Consult a qualified healthcare professional "
        "for medical assessment."
    )
