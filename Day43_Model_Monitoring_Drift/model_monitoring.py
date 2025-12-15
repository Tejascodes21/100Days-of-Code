# Day 43: Model Monitoring & Data Drift Detection

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime

np.random.seed(42)

train_data = pd.DataFrame({
    "income": np.random.normal(60000, 15000, 500),
    "loan_amount": np.random.normal(200000, 50000, 500),
    "credit_score": np.random.normal(700, 50, 500)
})


# 2. Simulate Production Data

prod_data = pd.DataFrame({
    "income": np.random.normal(52000, 17000, 500),   # drifted
    "loan_amount": np.random.normal(230000, 60000, 500),
    "credit_score": np.random.normal(680, 60, 500)   # drifted
})



# 3. Fake Model Predictions

def mock_predict_probability(df):
    """
    Simulates model prediction probabilities
    """
    score = (
        0.4 * (df["income"] / df["income"].max()) +
        0.4 * (df["credit_score"] / df["credit_score"].max()) -
        0.2 * (df["loan_amount"] / df["loan_amount"].max())
    )
    return np.clip(score, 0, 1)


prod_data["prediction_probability"] = mock_predict_probability(prod_data)
prod_data["timestamp"] = datetime.now()


# 4. Drift Detection (KS Test)

def detect_drift(train_col, prod_col, feature_name, alpha=0.05):
    stat, p_value = ks_2samp(train_col, prod_col)
    drift = p_value < alpha

    return {
        "feature": feature_name,
        "p_value": round(p_value, 4),
        "drift_detected": drift
    }


drift_results = []

for col in train_data.columns:
    result = detect_drift(train_data[col], prod_data[col], col)
    drift_results.append(result)


drift_report = pd.DataFrame(drift_results)



# 5. Monitoring Summary

print("\n MODEL MONITORING REPORT")
print("-" * 40)
print(drift_report)

if drift_report["drift_detected"].any():
    print("\n Drift detected! Model retraining recommended.")
else:
    print("\n No significant drift detected.")



# 6. Save Monitoring Logs

drift_report.to_csv("drift_report.csv", index=False)
prod_data.to_csv("production_predictions_log.csv", index=False)

print("\n Files saved:")
print(" drift_report.csv")
print(" production_predictions_log.csv")
