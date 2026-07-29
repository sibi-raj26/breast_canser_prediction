import os
import joblib
import numpy as np
import streamlit as st

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Prediction",
        "Model Information",
        "About"
    ]
)

# ---------------------------------
# Home Page
# ---------------------------------
if page == "Home":

    st.title("🩺 Breast Cancer Prediction System")

    st.markdown("---")

    st.markdown("""
## Early Detection Saves Lives

This application uses a **Machine Learning Logistic Regression Model**
to predict whether a breast tumor is **Benign** or **Malignant**.
""")

    c1, c2, c3 = st.columns(3)

    c1.metric("Model", "Logistic Regression")
    c2.metric("Accuracy", "97.37%")
    c3.metric("Features", "8")

    st.markdown("---")

    st.subheader("Technologies Used")

    st.markdown("""
- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Plotly
""")

# ---------------------------------
# Prediction Page
# ---------------------------------
elif page == "Prediction":

    st.title("🔍 Breast Cancer Prediction")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(BASE_DIR,  "cancer_model.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "trained_model", "scaler.pkl")
    FEATURE_PATH = os.path.join(BASE_DIR, "trained_model", "feature_names.pkl")

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_PATH)

    except Exception as e:
        st.error("❌ Model files not found.")
        st.error(str(e))
        st.stop()

    st.subheader("Enter Tumor Details")

    default_values = {
        "mean radius": 14.0,
        "mean texture": 19.0,
        "mean perimeter": 90.0,
        "mean area": 650.0,
        "mean smoothness": 0.09,
        "worst radius": 16.0,
        "worst perimeter": 107.0,
        "worst area": 880.0,
    }

    input_values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        default = default_values.get(feature, 0.0)

        if i % 2 == 0:
            with col1:
                value = st.number_input(
                    feature.title(),
                    value=float(default),
                    step=0.01
                )
        else:
            with col2:
                value = st.number_input(
                    feature.title(),
                    value=float(default),
                    step=0.01
                )

        input_values.append(value)

    if st.button("🔍 Predict", use_container_width=True):

        data = np.array(input_values).reshape(1, -1)

        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0]

        confidence = max(probability) * 100

        st.markdown("---")

        if prediction == 1:
            st.success("✅ Benign Tumor")
        else:
            st.error("❌ Malignant Tumor")

        st.metric("Confidence", f"{confidence:.2f}%")

        st.progress(confidence / 100)

        st.subheader("Prediction Probability")

        st.write(f"🔴 Malignant : {probability[0]*100:.2f}%")
        st.write(f"🟢 Benign : {probability[1]*100:.2f}%")

# ---------------------------------
# Model Information
# ---------------------------------
elif page == "Model Information":

    st.title("📊 Model Information")

    st.markdown("""
**Dataset:** Breast Cancer Wisconsin Dataset

**Algorithm:** Logistic Regression

**Features:** 8

**Scaling:** StandardScaler

**Classification:** Binary Classification
""")

# ---------------------------------
# About
# ---------------------------------
elif page == "About":

    st.title("ℹ️ About")

    st.markdown("""
### Breast Cancer Prediction System

This project predicts whether a breast tumor is **Benign** or **Malignant**
using a Logistic Regression Machine Learning model.

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Plotly

### Developer

Your Name
""")