LIVE LINK: https://breastcanserprediction-o5g7qxuzrqmdd9dnacx849.streamlit.app/

# 🩺 Breast Cancer Prediction System

A Machine Learning web application built with **Streamlit** that predicts whether a breast tumor is **Benign** or **Malignant** using a Logistic Regression model.

---

## 📌 Features

- 🩺 Breast cancer prediction
- 🤖 Machine Learning (Logistic Regression)
- 📊 8 important medical features
- ⚡ Fast predictions
- 🎨 Simple and user-friendly Streamlit interface
- 📈 Displays prediction confidence

---

## 📂 Project Structure

```
breast_cancer_prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── trained_model/
│   ├── cancer_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── utils/
```

---

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-Learn
- NumPy
- Pandas
- Joblib

---

## 📊 Machine Learning Model

- Algorithm: Logistic Regression
- Dataset: Breast Cancer Wisconsin Dataset
- Feature Scaling: StandardScaler
- Selected Features: 8

### Features Used

- Mean Radius
- Mean Texture
- Mean Perimeter
- Mean Area
- Mean Smoothness
- Worst Radius
- Worst Perimeter
- Worst Area

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/sibi-raj26/breast_cancer_prediction.git
```

### Go to Project Folder

```bash
cd breast_cancer_prediction
```

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧠 Train the Model

To retrain the machine learning model:

```bash
python train_model.py
```

This creates:

```
trained_model/
├── cancer_model.pkl
├── scaler.pkl
└── feature_names.pkl
```

---

## 📈 Model Performance

- Algorithm: Logistic Regression
- Accuracy: **97.37%**
- Binary Classification
- Fast Prediction

---

## 📷 Application Pages

- 🏠 Home
- 🔍 Prediction
- 📊 Model Information
- ℹ️ About

---

## 👨‍💻 Developer

**Vijay**

GitHub:
https://github.com/sibi-raj26

---

## 📄 License

This project is created for educational and learning purposes.
