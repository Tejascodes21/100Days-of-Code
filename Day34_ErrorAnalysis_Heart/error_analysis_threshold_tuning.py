# Day 34 - Error Analysis & Threshold Optimization (Heart Disease Prediction)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)


df = pd.read_csv("heart.csv")

X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

# Stratified split so both sets contain both classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2️ Train baseline model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probability scores

baseline_acc = accuracy_score(y_test, y_pred)
print(f"\nBaseline Accuracy: {baseline_acc:.3f}")

print("\n🔹 Classification Report (Default Threshold 0.5):")
print(classification_report(y_test, y_pred))


# 3️ Confusion Matrix - Where is the model failing?
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix (Default Threshold = 0.5)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 4️ Threshold Tuning
thresholds = np.arange(0.3, 0.71, 0.05)
acc_scores = []
recall_scores = []

for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    acc_scores.append(accuracy_score(y_test, y_pred_thr))
    recall_scores.append(
        (confusion_matrix(y_test, y_pred_thr)[1,1]) / sum(y_test)
    )

# Plot threshold vs performance
plt.figure(figsize=(8,4))
plt.plot(thresholds, acc_scores, marker='o', label="Accuracy")
plt.plot(thresholds, recall_scores, marker='s', label="Recall", linestyle='--')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning — Accuracy vs Recall")
plt.legend()
plt.show()


# 5️ ROC Curve + AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


print(" Error analysis complete!")
