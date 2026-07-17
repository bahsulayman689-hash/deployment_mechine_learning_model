import pandas as pd
import numpy as np
import joblib
import streamlit as st

model = joblib.load("model_mail.pkl")
feature_extraction = joblib.load("feature_mail.pkl")



st.set_page_config(
    page_title="Spam Email Detector",
    page_icon='📧',
    layout='centered'
)
st.title("📧CLASSIFIER THE MESSAGE EITHER SPAM OR HAM MAIL")
st.write("Enter an email or SMS message to determine whether it is Spam or Ham.")

meassage = st.text_area("Enter your meassage", height=180)

if st.button("predict"):
    if meassage.strip() == "":
        st.warning("please enter a meassage")
    else:
        meassage_vector = feature_extraction.transform([meassage])

        prediction = model.predict(meassage_vector)[0]

        if prediction == 0:
            st.error("🚨Spam mesaage")
        else:
            st.success("✅Ham (Not Spam)")