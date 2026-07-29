import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():

    print("Loading Breast Cancer Dataset...")

    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Top 8 Important Features
    selected_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "worst radius",
        "worst perimeter",
        "worst area"
    ]

    X = df[selected_features]
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=10000)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"Accuracy : {accuracy:.4f}")

    os.makedirs("trained_model", exist_ok=True)

    joblib.dump(model, "trained_model/cancer_model.pkl")
    joblib.dump(scaler, "trained_model/scaler.pkl")
    joblib.dump(selected_features, "trained_model/feature_names.pkl")

    print("Model Saved Successfully.")


if __name__ == "__main__":
    train_model()