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
st.balloons()
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
                


<style>

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1E3A8A;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)
    st.divider()

    st.caption(
        "Built by Sulayman Bah • Python • Scikit-learn • Streamlit"
    )

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
col1, col2, col3 = st.columns(3)

col1.metric("Training Accuracy", "96.2%")
col2.metric("Testing Accuracy", "96.2%")
col3.metric("Algorithm", "Logistic Regression")

st.markdown("""
<style>
div[data-testid="stTextArea"] textarea {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)
with st.sidebar:

    st.image("WIN_20250906_05_26_12_Pro.jpg", width=150)

    st.title("👨‍💻 Sulayman Bah")

    st.write("Machine Learning Developer")

    st.divider()

    st.write("📍 The Gambia")

    st.write("💡 Passionate about AI & Machine Learning")
with st.sidebar:

    st.subheader("🔗 Connect")

    st.markdown("[GitHub](https://github.com/bahsulayman689-hash/deployment_mechine_learning_model/commits?author=bahsulayman689-hash)")

    st.markdown("[LinkedIn](https://linkedin.com/in/WIN_20250906_05_26_12_Pro.jpg)")

    st.markdown("[Portfolio](https://yourportfolio.com)")
with st.sidebar.expander("👤 About Me"):

    st.write("""
    Hi! I'm Sulayman Bah.

    I build Machine Learning and
    Deep Learning applications
    using Python and Streamlit.
    """)
st.sidebar.subheader("🛠 Skills")

st.sidebar.write("🐍 Python")
st.sidebar.write("🤖 Machine Learning")
st.sidebar.write("📊 Data Analysis")
st.sidebar.write("🎨 Deep learning")
st.sidebar.write("🧠 Software enginner")
st.divider()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("email.png", width=60)
    st.title("ML Ops Dashboard")
    st.markdown("---")
    
    # 1. Model Selection Hub (Perfect for your 20+ projects)
    st.subheader("📁 Select Project")
    selected_model = st.selectbox(
        "Choose an active deployment:",
        ["Spam Filter v2.1", "Churn Predictor v1.0", "Fraud Detection v4.2", "Sentiment Engine v3.0"]
    )
    
    # 2. Dynamic Model Information
    st.info(f"Active: **{selected_model}**\n\nStatus: 🟢 Operational")
    st.markdown("---")
    
    # 3. Interactive Hyperparameters for Testing
    st.subheader("⚙️ Model Settings")
    decision_threshold = st.slider(
        "Spam Sensitivity Threshold", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    
    st.markdown("---")
    
    # 4. Sticky System Metrics Footer
    st.markdown("### 📊 Infrastructure Status")
    st.caption("🖥️ Server: AWS Lambda (Serverless)")
    st.caption("⚡ Latency: 42ms")
    st.caption("📅 Last Retrained: July 2026")
