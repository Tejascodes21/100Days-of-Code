# Day10_Data_Preprocessing_Pipeline.py
# Goal: Create a preprocessing pipeline for ML-ready data

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# 1️ Load dataset
df = pd.read_csv("cleaned_customer_data.csv")

print("Original Data Shape:", df.shape)
print(df.head())

# 2️ Identify categorical & numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("\nCategorical Columns:", list(categorical_cols))
print("Numerical Columns:", list(numerical_cols))

# 3️ Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('transformer', PowerTransformer(method='yeo-johnson'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 4️ Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# 5️ Split the data (if target column exists)
target_col = 'Purchased' if 'Purchased' in df.columns else None
if target_col:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train = df.copy()

# 6️ Build full pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# 7️ Fit & transform the data
processed_data = model_pipeline.fit_transform(X_train)

# 8️ Convert transformed data back to DataFrame
processed_df = pd.DataFrame(processed_data.toarray() if hasattr(processed_data, 'toarray') else processed_data)
print("\n Data after preprocessing:", processed_df.shape)

# 9️ Export final preprocessed dataset
processed_df.to_csv("Day10_Preprocessed_Data.csv", index=False)
print("Exported 'Day10_Preprocessed_Data.csv' successfully!")

