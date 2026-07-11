import streamlit as st
import numpy as np
import joblib
import warnings as wr
wr.filterwarnings('ignore')

st.title("🩺BREAST CANCER DISEASE DETECTION")

model = joblib.load("breast_cancer.pkl")
scaler = joblib.load("scaler_breast.pkl")
#texture_mean	perimeter_mean	area_mean	smoothness_mean	compactness_mean	concavity_mean	concave points_mean	symmetry_mean	fractal_dimension_mean	radius_se	texture_se	perimeter_se	area_se	smoothness_se	compactness_se	concavity_se	concave points_se	symmetry_se	fractal_dimension_se	radius_worst	texture_worst	perimeter_worst	area_worst	smoothness_worst	compactness_worst	concavity_worst	concave points_worst	symmetry_worst	fractal_dimension_worst

radius_mean = st.number_input("radius_mean", min_value=0.00)
texture_mean = st.number_input("textture_mean", min_value=0.00)
perimeter_mean = st.number_input("perimeter_mean", min_value=0.00)
area_mean	= st.number_input("area_mean", min_value=0.00)
smoothness_mean = st.number_input("smoothness_mean", min_value=0.00)
compactness_mean = st.number_input("compactness_mean", min_value=0.00)	
concavity_mean	 = st.number_input("concavity_mean", min_value=0.00)
concave_points_mean = st.number_input("concave_point_mean", min_value=0.00)
symmetry_mean	 = st.number_input("symmetry_mean", min_value=0.00)
fractal_dimension_mean = st.number_input("fraction_dim_mean", min_value=0.00)

radius_se	= st.number_input("radius_se", min_value=0.00)
texture_se	= st.number_input("texture_se", min_value=0.00)
perimeter_se = st.number_input("preimeter_se", min_value=0.00)	
area_se	= st.number_input("area_se", min_value=0.00)
smoothness_se	= st.number_input("smmothness_se", min_value=0.00)
compactness_se	= st.number_input("campactness_se", min_value=0.00)
concavity_se	= st.number_input("concavity_Se", min_value=0.00)
concave_points_se = st.number_input("concave_point_Se", min_value=0.00)	
symmetry_se = st.number_input("symmetry_se", min_value=0.00)
fractal_dimension_se = st.number_input("fraction_dim_se", min_value=0.00)

radius_worst	= st.number_input("radius_worst", min_value=0.00)
texture_worst	= st.number_input("texture_worst", min_value=0.00)
perimeter_worst = st.number_input("perimeter_worst", min_value=0.00)

area_worst	 = st.number_input("area_worst", min_value=0.00)
smoothness_worst	 = st.number_input("smoothness_worst", min_value=0.00)
compactness_worst	= st.number_input("compactness_worst", min_value=0.00)
concavity_worst	= st.number_input("concavity_worst", min_value=0.00)
concave_points_worst	= st.number_input("concave_points_worst", min_value=0.00)
symmetry_worst	 = st.number_input("symmetry_worst", min_value=0.00)
fractal_dimension_worst = st.number_input("fraction_dim_worst", min_value=0.00)

if st.button('BREAST_PREDICTION'):
    data = np.array([
        [
            radius_mean,
            texture_mean,
            perimeter_mean,

            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,
            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_mean,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,
            radius_mean,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst

            
            

            

        ]
    ])
    data = scaler.transform(data)
    prediction = model.predict(data)
    if prediction[0] == 0:
        st.success("✅The person has no breast cancer")
    else:
        st.error("❌The person has a breast cancer")