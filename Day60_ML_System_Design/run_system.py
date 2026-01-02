import pandas as pd
from system.data_validation import validate_data
from system.train_model import train_model
from system.inference import load_model, predict
from system.drift_check import check_drift
from system.retrain_decision import should_retrain

# Load training data
train_df = pd.read_csv("data/train_data.csv")
X_train = train_df[["age", "income", "loan_amount"]]
y_train = train_df["default"]

validate_data(X_train)

# Train model
model = train_model(X_train, y_train)

# Load live data
live_df = pd.read_csv("data/live_data.csv")
validate_data(live_df)

# Inference
preds = predict(model, live_df)
print(" Live Predictions:", preds.tolist())

# Drift detection
train_mean = X_train["income"].mean()
live_mean = live_df["income"].mean()

drift_detected = check_drift(train_mean, live_mean)

# Retrain decision
should_retrain(drift_detected, accuracy=0.75)
