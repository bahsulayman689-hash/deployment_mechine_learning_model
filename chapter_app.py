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

st.markdown("""
Detect whether an email or SMS is **Spam** or **Ham**
using a Machine Learning model trained with **TF-IDF**
and **Logistic Regression**.
""")

message = st.text_area(
    "✍️ Enter your email or SMS",
    placeholder="Paste your email or SMS message here...",
    height=220
)

if st.button("predict"):
    if message.strip() == "":
        st.warning("please enter a meassage")
    else:
        message_vector = feature_extraction.transform([message])

        prediction = model.predict(message_vector)[0]

        col1, col2 = st.columns(2)

        with col1:
            predict = st.button("🔍 Analyze")

        with col2:
            clear = st.button("🗑️ Clear")

        if prediction == 0:
            st.error("🚨Spam mesaage")
            st.write("This message contains characteristics commonly found in spam.")
        else:
            st.success("✅Ham (Not Spam)")
            st.write("This message appears to be legitimate.")
with st.sidebar:

    st.title("📧 Spam Detector")

    st.markdown("""
### Model

- Logistic Regression
- TF-IDF Vectorizer

### Features

- Email Detection
- SMS Detection
- Fast Prediction

### Built With

- Python
- Streamlit
- Scikit-learn
""")
    examples = [
    "Congratulations! You've won a free iPhone. Click here to claim.",
    "Hi, are we still meeting at 3 PM today?",
    "Your bank account has been suspended. Verify immediately."
]

selected = st.selectbox("📌 Try an example", [""] + examples)

if selected:
    message = selected
with st.expander("🧠 How it works"):

    st.write("""
This application uses:

- TF-IDF Vectorization
- Logistic Regression
- Supervised Machine Learning

The model converts text into numerical features
before predicting whether the message is Spam or Ham.
""")
col1, col2 = st.columns(2)

col1.metric("Training Accuracy", "98.2%")
col2.metric("Algorithm", "Logistic Regression")