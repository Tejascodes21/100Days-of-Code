
import pandas as pd
import joblib
from feature_store import build_features, FEATURE_VERSION

# Load model bundle
bundle = joblib.load("model.pkl")
model = bundle["model"]
trained_feature_version = bundle["feature_version"]

# Consistency check
if trained_feature_version != FEATURE_VERSION:
    raise ValueError(" Feature version mismatch detected!")


df = pd.read_csv("data/inference_data.csv")

X = build_features(df)

prediction = model.predict(X)
probability = model.predict_proba(X)[:, 1]

for i in range(len(prediction)):
    print(
        f"Prediction: {prediction[i]} | "
        f"Risk Score: {round(probability[i], 3)}"
    )
