import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def validate_model():
    model = joblib.load("model/model.pkl")

    df = pd.read_csv("data/train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    assert acc >= 0.7, f" Model accuracy too low: {acc}"

    print(f" Model performance check passed (Accuracy={acc:.2f})")
