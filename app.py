import streamlit as st
import numpy as np
import joblib

model = joblib.load("model_logistic.pkl")
scaler = joblib.load("scaler.pkl")


st.title("🧠Liver Disease Prediction")

menu = st.sidebar.radio("Navigation", 
                        [
                            "🏠Home",
                            "🩺Prediction",
                            "📈Dataset",
                            "📊Model Performance",
                            "ℹ️About Liver Disease",
                            "🛡️Prevention of Liver Disease",
                            "📝About Project",
                            ])

Age = st.number_input("Age", min_value=0.0, max_value=200.00)

Gender = st.number_input('Gender', min_value=0.00)

Bmi = st.number_input("BMI", min_value=0.00)

AlcoholConsumption = st.number_input("Alcohol_consumption", min_value=0.00)

Smoking = st.number_input("Smoking", min_value=0.00)

GeneticRisk = st.number_input("Geneticrisk", min_value=0.00)

PhysicalActivity = st.number_input("Physical_Activity", min_value=0.00)

Diabetes = st.number_input("Diabetes", min_value=0.00)

Hypertension = st.number_input("Hypertension", min_value=0.00)

LiverFunctionTest  = st.number_input("LiverFunctionTest", min_value=0.00)

if st.button("PREDICTION"):
    data = np.array([
        [
            Age,
            Gender,
            Bmi,
            AlcoholConsumption,
            Smoking,
            GeneticRisk,
            PhysicalActivity,
            Diabetes,
            Hypertension,
            LiverFunctionTest,
        ]
    ])
    data = scaler.transform(data)
    
    prediction = model.predict(data)

    if prediction[0] == 1:

        st.error("❌The Person has Liver Disease")
    else:
        st.success("✅The person has no Liver Disease detected")