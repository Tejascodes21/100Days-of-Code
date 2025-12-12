# Day 40 - Loan Default Preprocessing & Modeling (No imblearn required)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("\n Loading sample dataset...")

df = pd.read_csv("data/loan_data.csv")
print("Dataset Loaded Successfully!")
print(df.head())


# 1️ TRAIN–TEST SPLIT
# Split features and target correctly
X = df.drop("loan_default", axis=1)
y = df["loan_default"]

print("Feature columns:", X.columns)
print("Target sample:", y.head())


# Identify numeric & categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric Columns:", num_cols)
print("Categorical Columns:", cat_cols)


# 2️ PREPROCESSING PIPELINE

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


# 3️ MODEL (Handles imbalance natively)

model = LogisticRegression(
    max_iter=300,
    class_weight="balanced"   
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])


# 4️ SPLIT INTO TRAIN & TEST

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n Training the model...")
pipeline.fit(X_train, y_train)

print("\n Model training complete!")


# 5️ MODEL EVALUATION

y_pred = pipeline.predict(X_test)

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(pipeline, "loan_default_model.pkl")
print("\n Model saved as loan_default_model.pkl")

# Save clean processed dataset
df.to_csv("loan_cleaned.csv", index=False)
print(" Clean dataset saved → loan_cleaned.csv")

print("\n Day 40 Preprocessing + Modeling Completed Successfully!")

