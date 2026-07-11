import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load("breast_cancer.pkl")
scaler = joblib.load("scaler_breast.pkl")

st.title("🩺Breast Cancer Prediction")

data = pd.read_csv("breast-cancer.csv")
uploaded_file = st.file_uploader(
    "Upload your input CSV file",
    type=["csv"]
)
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.dataframe(data)

    if "id"and 'diagnosis' in data.columns:
        data = data.drop(columns=['id', 'diagnosis'], axis=0)
    data = data[scaler.feature_names_in_]

    scaled_data = scaler.transform(data)

    predictions = model.predict(scaled_data)

    data['Prediction'] = predictions

    st.success("Prediction Completed!")

    st.dataframe(data)

    csv = data.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="breast_cancer_predictions.csv",
        mime="text/csv"
    )