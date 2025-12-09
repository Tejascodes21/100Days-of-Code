# Day 37 - Model Explainability with SHAP (FINAL WORKING VERSION)

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("heart.csv")
print(df.columns)


target_col = "heart_disease" if "heart_disease" in df.columns else "target"

X = df.drop(columns=[target_col])
y = df[target_col]


# 2) Train Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(" Model trained & saved!")


# 3) SHAP Linear Explainer

explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

# For binary classification → shap_values[1] = positive class
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values


# 4) SHAP Summary Plot

plt.figure()
shap.summary_plot(shap_vals, X_test, feature_names=X.columns, show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()
print("SHAP Summary saved → shap_summary.png")


# 5) SHAP Force Plot (Individual Prediction)
sample = X_test[0:1]
sample_shap_vals = shap_vals[0]

plt.figure()
shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
    sample_shap_vals,
    feature_names=X.columns,
    matplotlib=True
)
plt.savefig("shap_force.png", bbox_inches="tight")
plt.close()
print(" SHAP Force plot saved → shap_force.png")

print("DONE! Model Explainability Completed Successfully.")
