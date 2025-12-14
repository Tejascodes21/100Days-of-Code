# Day 42 - Model Robustness & Stress Testing
# Loan Default Prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# 1️ Load Data
df = pd.read_csv("loan_sample.csv")

X = df.drop("default", axis=1)
y = df["default"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns


# 2️ Preprocessing & Model

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = LogisticRegression(
    max_iter=300,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])


# 3️ Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)

baseline_probs = pipeline.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, baseline_probs)

print(f"\n Baseline ROC-AUC: {baseline_auc:.3f}")


# 4️ Stress Testing(Extreme Values)

stress_data = X_test.copy()

stress_data["income"] *= 0.4       
stress_data["loan_amount"] *= 1.5  
stress_data["credit_score"] -= 80   

stress_probs = pipeline.predict_proba(stress_data)[:, 1]
stress_auc = roc_auc_score(y_test, stress_probs)

print(f" Stress Test ROC-AUC: {stress_auc:.3f}")


# 5️ Noise Robustness Test

noise_data = X_test.copy()
noise_data[num_cols] = noise_data[num_cols] + np.random.normal(0, 0.1, noise_data[num_cols].shape)

noise_probs = pipeline.predict_proba(noise_data)[:, 1]
noise_auc = roc_auc_score(y_test, noise_probs)

print(f" Noisy Data ROC-AUC: {noise_auc:.3f}")


# 6️ Sensitivity Analysis

feature = "income"
sensitivity_data = X_test.copy()

sensitivity_data[feature] *= 1.1  # +10% income
sensitivity_probs = pipeline.predict_proba(sensitivity_data)[:, 1]

avg_change = np.mean(np.abs(sensitivity_probs - baseline_probs))

print(f" Avg Prediction Change after {feature} shift: {avg_change:.4f}")


with open("robustness_report.txt", "w") as f:
    f.write("Day 42 – Model Robustness & Stress Testing Report\n\n")
    f.write(f"Baseline ROC-AUC: {baseline_auc:.3f}\n")
    f.write(f"Stress Test ROC-AUC: {stress_auc:.3f}\n")
    f.write(f"Noisy Data ROC-AUC: {noise_auc:.3f}\n")
    f.write(f"Avg Prediction Sensitivity (income): {avg_change:.4f}\n")

print("\n Robustness report saved → robustness_report.txt")
