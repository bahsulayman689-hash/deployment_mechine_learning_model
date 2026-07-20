# 💳 Real-Time Credit Card Fraud Detection Pipeline & ML Ops Hub

A production-ready, highly optimized Streamlit dashboard that serves a machine learning classification engine for financial telemetry data. The system routes real-time individual transaction vectors or processes massive batch transactions simultaneously through a calibrated Logistic Regression decision boundary.

Developed by **Sulayman Bah** • *Machine Learning & Deep Learning Engineer*

---

## 🚀 Key Engineering & Architecture Features

### ⚡ 1. Large-Scale Data Optimization (140MB+ Files)
- **Vectorized Inference Loops:** Replaced slow iterative row processing with high-performance NumPy horizontal stacks (`np.hstack`). This enables instantaneous parallel calculation over thousands of data entries.
- **Fast Storage Engines:** Configured the Pandas engine using low-memory layout strategies (`engine='c'`, `low_memory=False`) to process heavy batch uploads (such as the standard 143.8MB Kaggle dataset) without risking runtime memory overflows.
- **Dynamic Header Audits:** Implemented an upfront row-zero lookup (`nrows=1`) to dynamically detect file layouts, automatically stripping target ground-truth vector arrays (`Class`) without breaking user workflows.

### 🛠️ 2. Scikit-Learn Framework Migration Upgrades
- **Version Compatibility:** Patched runtime deprecation exceptions by upgrading the `LogisticRegression` configuration to match modern Scikit-Learn environments, removing deprecated parameter footprints like `multi_class`.
- **Dimensionality Safety:** Enforced explicit 2D column formatting (`np.array([[value]])`) across raw transaction scalars (`Time`, `Amount`) to eliminate dimensional fragmentation during scaling pipelines.

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit Dashboard Application
├── credit_model.pkl            # Trained Logistic Regression Model Weights
├── credit_scaler.pkl           # Pre-fitted StandardScaler Model Pipeline
├── README.md                   # Project Documentation Engine
└── requirements.txt            # Project Software Dependencies
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the Workspace
```bash
git clone https://github.com
cd fraud-detection-dashboard
```

### 2. Configure Your Virtual Environment
```bash
python -bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Software Dependencies
Ensure your environment meets the structural library constraints:
```bash
pip install -r requirements.txt
```

### 4. Boot Up the Machine Learning Application Instance
```bash
streamlit run app.py
```

---

## 📊 Model Evaluation Summary

The core evaluation metrics for the predictive backend instance (trained on an undersampled, balanced financial telemetry framework) include:

- **Model Core Kernel:** Logistic Regression Classifier
- **Model Baseline Accuracy:** 94.41%
- **Downsampled Scale Balance:** 1:1 Distribution Pattern (984 Balanced Telemetry Rows)
- **Feature Matrix Inputs:** 28 Structural Anonymized PCA Elements + Standardized Chronological & Valuation Metrics

---

## 🤝 Project Registry Workspace Core
This interface operates as part of an **ML Ops Hub Deployment**. The dashboard infrastructure includes template parameters ready to scale additional pipeline workflows, such as:
- **Spam Filter v2.1** (NLP Text Tokenization Matrix Router)
- **Fashion CNN Classifier** (Computer Vision Tensor Deep Learning Network)

---
*Built with passion by Sulayman Bah • Python • Scikit-learn • Streamlit*
