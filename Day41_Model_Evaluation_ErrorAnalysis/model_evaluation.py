# Day 41 - Model Evaluation & Error Analysis (Loan Default Prediction)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)


df = pd.read_csv("loan_sample.csv")

X = df.drop("default", axis=1)
y = df["default"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

log_model = LogisticRegression(
    max_iter=300,
    class_weight="balanced"
)

tree_model = DecisionTreeClassifier(
    max_depth=4,
    class_weight="balanced",
    random_state=42
)

log_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", log_model)
])

tree_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", tree_model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


log_pipeline.fit(X_train, y_train)
tree_pipeline.fit(X_train, y_train)


def evaluate_model(name, pipeline):
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"\n {name} Evaluation")
    print("ROC-AUC:", round(roc, 3))
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_test, preds))

    return probs


log_probs = evaluate_model("Logistic Regression", log_pipeline)
tree_probs = evaluate_model("Decision Tree", tree_pipeline)


precision, recall, thresholds = precision_recall_curve(y_test, log_probs)

custom_threshold = 0.35
custom_preds = (log_probs >= custom_threshold).astype(int)

print("\n Threshold Tuning (Logistic Regression)")
print("Using threshold:", custom_threshold)
print(confusion_matrix(y_test, custom_preds))
print(classification_report(y_test, custom_preds))


error_df = X_test.copy()
error_df["actual"] = y_test.values
error_df["predicted"] = custom_preds

false_positives = error_df[(error_df.actual == 0) & (error_df.predicted == 1)]
false_negatives = error_df[(error_df.actual == 1) & (error_df.predicted == 0)]

print("\n False Positives:", len(false_positives))
print(" False Negatives:", len(false_negatives))

with open("evaluation_report.txt", "w") as f:
    f.write("Day 41 – Model Evaluation & Error Analysis\n\n")
    f.write(f"ROC-AUC (Logistic Regression): {roc_auc_score(y_test, log_probs):.3f}\n")
    f.write(f"ROC-AUC (Decision Tree): {roc_auc_score(y_test, tree_probs):.3f}\n")
    f.write(f"Threshold used: {custom_threshold}\n")
    f.write(f"False Positives: {len(false_positives)}\n")
    f.write(f"False Negatives: {len(false_negatives)}\n")

print("\n Evaluation report saved → evaluation_report.txt")
