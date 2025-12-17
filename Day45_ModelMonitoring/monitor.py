import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from datetime import datetime

MODEL_THRESHOLD = 0.75  # alert threshold

model = joblib.load("model.pkl")
prod_data = pd.read_csv("data/production_data.csv")

X_prod = prod_data.drop("default", axis=1)
y_prod = prod_data["default"]

preds = model.predict(X_prod)
accuracy = accuracy_score(y_prod, preds)

log_entry = {
    "timestamp": datetime.now(),
    "accuracy": round(accuracy, 3),
    "alert": "YES" if accuracy < MODEL_THRESHOLD else "NO"
}

log_df = pd.DataFrame([log_entry])

try:
    existing = pd.read_csv("metrics_log.csv")
    log_df = pd.concat([existing, log_df], ignore_index=True)
except FileNotFoundError:
    pass

log_df.to_csv("metrics_log.csv", index=False)

print(f" Current Accuracy: {accuracy:.2f}")

if accuracy < MODEL_THRESHOLD:
    print(" ALERT: Model performance dropped below threshold!")
else:
    print(" Model performing within acceptable range.")
