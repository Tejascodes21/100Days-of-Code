import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_evaluate(data_path, model_path, metrics_path):
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    df = pd.read_csv(data_path)

    X = df.drop("default", axis=1)
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    
    joblib.dump(model, model_path)


    with open(metrics_path, "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=4)

    print(f" Model saved at: {model_path}")
    print(f" Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    train_and_evaluate(
        data_path="data/train_data.csv",
        model_path="models/model_v1.pkl",
        metrics_path="models/metrics_v1.json"
    )
