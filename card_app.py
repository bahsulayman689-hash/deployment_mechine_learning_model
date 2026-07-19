import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="💳 Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# --- LOADING THE TRAINED PIPELINE ARTIFACTS ---
@st.cache_resource
def load_deployment_assets():
    model = joblib.load("credit_model.pkl")
    scaler = joblib.load("credit_scaler.pkl")
    return model, scaler
col_main, col_right = st.columns(2)
with col_main:
    st.title("💳legit and fradulent classifier mechine engine detection")
with col_right:
    st.image("credit-cart.png", width=220)
try:
    model, scaler = load_deployment_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.sidebar.error("⚠️ Model Assets Status: Awaiting deployment weights...")

# --- SIDEBAR PRESENTATION CONTROL ---
with st.sidebar:
    st.image("WIN_20250906_05_26_12_Pro.jpg", width=130)
    st.title("👨‍💻 Sulayman Bah")
    st.caption("Machine Learning & Deep Learning Engineer")
    st.divider()
    
    st.title("ML Ops Hub Dashboard")
    st.subheader("📁 Project Registry")
    selected_project = st.selectbox(
        "Select Active Framework Deployment:",
        ["Credit Card Fraud Tracker", "Spam Filter v2.1", "Fashion CNN Classifier"]
    )
    st.info(f"Active Workspace: **{selected_project}**\n\nStatus: 🟢 Production Ready")
    st.divider()
    st.caption("Built by Sulayman Bah • Python • Scikit-learn • Streamlit")

# --- MAIN PAGE HUB LAYOUT ---
st.title("💳 REAL-TIME CREDIT CARD FRAUD DETECTION INSTANCE")
st.markdown("""
### 🔬 Financial Telemetry Matrix Analysis
This classification layer routes live financial transactions through a Logistic Regression decision boundary. 
The system evaluates transactions against 28 PCA structural features alongside standardized chronological and valuation metrics.
""")

tab1, tab2 = st.tabs(["✨ Single Vector Evaluation", "📂 Batch CSV File Processing"])
# --- USER INPUT FEED ZONE ---
st.write("### 📥 Transaction Stream Vector Feed")
st.write("Paste a raw comma-separated transaction array containing **30 values** (Time, V1 to V28, Amount):")

# Pre-packaged transaction test cases from the dataset
legit_sample = "0,1.19185711131486,0.26615071205963,0.16648011335321,0.448154078460911,0.0600176492822243,-0.0823608088155687,-0.0788029833323113,0.0851016549148104,-0.255425128109186,-0.166974414004614,1.61272666105479,1.06523531137287,0.48909501589608,-0.143772296441519,0.635558093258208,0.463917041022171,-0.114804663102346,-0.183361270123994,-0.145783041325259,-0.0690831352230203,-0.225775248033138,-0.638671952771851,0.101288021253234,-0.339846475529127,0.167170404418143,0.125894532368176,-0.00898309914322813,0.0147241691924927,2.69"
fraud_sample = "406,-2.3122265423263,1.9519920106412,-1.6098507328042,3.9979055875411,-0.5221878641258,-1.4265452310245,-2.5373873421258,1.3916573412548,-2.7700892341258,-2.7722721024512,3.2020331254871,-2.8999071024581,-0.5952223412581,-4.2892541245812,0.3897210245812,-1.1407471245812,-2.8300551245812,-0.0168223412581,0.4169561024581,0.1269112458124,0.5172321024581,-0.0350491024581,-0.4652111024581,0.3201981024581,0.0445191024581,0.1778401024581,0.2611451024581,-0.1432761024581,0.00"

selected_preset = st.selectbox(
    "📌 Quick-Inject Sample Diagnostic Presets",
    ["Select a transaction profile...", "✅ Verified Legitimate Profile", "🚨 Verified Flagged Fraud Profile"]
)

default_text = ""
if selected_preset == "✅ Verified Legitimate Profile":
    default_text = legit_sample
elif selected_preset == "🚨 Verified Flagged Fraud Profile":
    default_text = fraud_sample

credit_card_input = st.text_area(
    "✍️ Comma-Separated Data Stream",
    value=default_text,
    placeholder="Paste row element numbers here...",
    height=120
)
#tab1, tab2 = st.tabs(["✨ Single Vector Evaluation", "📂 Batch CSV File Processing"])
# --- INTERFERENCE EVALUATION ENGINE ---
with tab1:
    if st.button("Evaluate Vector Security Signature", type="primary"):
        if not model_loaded:
            st.error("❌ Model execution halted. Save your pkl model binaries to activate evaluation.")
        elif credit_card_input.strip() == "":
            st.warning("Please inject data parameters first.")
        else:
            try:
                # 1. Standard parsing matching local training file
                parsed_number = [float(x) for x in credit_card_input.split(",")]
                
                if len(parsed_number) != 30:
                    st.error(f"📐 Structural Shape Mismatch: The input array contains {len(parsed_number)} features, but your network model strictly requires exactly 30 parameters.")
                else:
                    # 2. Match column alignment exactly to avoid dimension crashes
                    raw_time = parsed_number[0]
                    pca_features = parsed_number[1:29]
                    raw_amount = parsed_number[29]
                    
                    # Transform variables through the single-column trained scalar shape
                    scaled_time = scaler.transform(np.asarray([[raw_time]]))[0][0]
                    scaled_amount = scaler.transform(np.asarray([[raw_amount]]))[0][0]
                    
                    # Assemble array into 30 feature shape matrix
                    final_features = [scaled_time] + pca_features + [scaled_amount]
                    credit_card_reshape = np.asarray(final_features).reshape(1, -1)
                    col1, col2 =st.columns(2)
                    with col1:
                        st.button("anaylsis")
                    with col2:
                        st.button("clear")
                    # 3. Model Predict Passes
                    prediction = model.predict(credit_card_reshape)[0]
                    #probabilities = model.predict_proba(credit_card_reshape)[0]
                    
                    st.write("### 📊 Classification Outcome")
                    
                    if prediction == 0:
                        st.success("✅ TRANSACTION AUTHORIZED")
                        
                        st.info(f"Verified Safe Asset. Risk Threat Matrix Confidence")
                        st.balloons()
                    else:
                        st.error("🚨 SUSPICIOUS INCIDENT BLOCKED")
                        st.info(f"Verified Fraudulent Transaction Profile. System Certainty")
                        st.snow()
            except ValueError:
                st.error("❌ Array Processing Failure: The string contains textual characters or bad symbols that cannot be translated into numerical float metrics.")
with tab2:
    st.write("### 📂 Upload Multi-Row Transaction CSV File")
    st.markdown("""
    Upload a **CSV file** with your transactions. The file **must contain exactly 30 columns** 
    in order: `Time`, `V1` to `V28`, and `Amount`. Do not include a header row if it contains text labels.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="tab2_uploader")
    
    if uploaded_file is not None:
        if not model_loaded:
            st.error("❌ Model weights not loaded. Upload cannot process.")
        else:
            try:
                # Load CSV without headers to seamlessly parse pure data matrices
                header_check = pd.read_csv(uploaded_file, nrows=1)
                                # Check if 'Class' column is present in the file headers
                has_class = 'Class' in header_check.columns
                expected_raw_columns = 31 if has_class else 30
                
                # Verify raw column count before processing memory chunks
                if header_check.shape[1] != expected_raw_columns:
                    st.error(f"📐 File Structure Error: Found {header_check.shape[1]} columns. Expected {expected_raw_columns} columns.")
                else:
                    # Reset buffer pointer and read full large dataset with optimized memory engine
                    uploaded_file.seek(0)
                    with st.spinner("⚡ Loading large telemetry matrix file into memory engine..."):
                        df = pd.read_csv(uploaded_file, engine='c', low_memory=False)

                                # 2. Check if a 'Class' column exists by text name or index position
                if 'Class' in df.columns:
                    df_features = df.drop(columns=['Class'])
                    st.info("💡 Target column 'Class' detected and dropped automatically.")
               
                else:
                    df_features = df.copy()
                
                if df_features.shape[1] != 30:
                    st.error(f"📐 File Shape Mismatch: Your file has {df_features.shape[1]} columns, but the pipeline requires exactly 30.")
                else:
                    st.success(f"CSV file successfully loaded with **{df_features.shape[0]} rows** to process!")
                    
                    if st.button("🚀 Process Batch Run Pipeline", type="primary"):
                        with st.spinner("Executing pipeline matrix scaling and model routing..."):
                            # Separate Time, PCA features, and Amount arrays
                            raw_times = df_features.iloc[:, [0]].values
                            pca_features = df_features.iloc[:, 1:29].values
                            raw_amounts = df_features.iloc[:, [29]].values
                            
                            # Scale arrays uniformly via the saved transformer
                            scaled_times = scaler.transform(raw_times)
                            scaled_amounts = scaler.transform(raw_amounts)
                            
                            # Re-stitch data together into structural shapes matching training matrix
                            processed_features = np.hstack((scaled_times, pca_features, scaled_amounts))
                            
                            # Compute Batch Predictions
                            predictions = model.predict(processed_features)
                            df_output = df.copy()
                            # Append Results directly back onto user dataframe output view
                            df_output['Prediction Output'] = predictions
                            df_output['Prediction Output'] = df_output['Prediction Output'].map({0: "✅ Authorized (Legit)", 1: "🚨 Blocked (Fraud)"})
                            
                            st.write("### 📈 Batch Run Results Sample")
                            st.dataframe(df_output.head(20))
                            
                            # Count metrics
                            fraud_count = int((predictions == 1).sum())
                            legit_count = int((predictions == 0).sum())
                            
                            c_b1, c_b2 = st.columns(2)
                            c_b1.metric("Legitimate Identified", legit_count)
                            c_b2.metric("Fraudulent Flagged", fraud_count, delta=f"{fraud_count} Flagged", delta_color="inverse")
                            
                            # Create a download button for the generated output CSV
                            csv_buffer = df_output.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Full Batch Evaluation Report",
                                data=csv_buffer,
                                file_name="fraud_batch_evaluation_report.csv",
                                mime="text/csv"
                            )
                            st.balloons()
                            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    

#
st.balloons()

# --- METRICS BOARD FOOTER ---
st.divider()
st.subheader("Model Evaluation Logs (Undersampled Balanced Set)")
c1, c2, c3 = st.columns(3)
c1.metric(label="Model Accuracy", value="94.41%")
c2.metric(label="Downsampled Balance Scale", value="1:1 (984 rows)")
c3.metric(label="Model Core Kernel", value="Logistic Regression")

# Custom Dashboard Dark Theme Injections
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0F172A;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC;
}
div[data-testid="stTextArea"] textarea {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
