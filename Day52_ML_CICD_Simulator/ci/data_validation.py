import pandas as pd

def validate_data(df: pd.DataFrame):
    required_columns = ["age", "income", "loan_amount", "credit_score", "default"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df.isnull().sum().any():
        raise ValueError("Dataset contains missing values")

    print("Data validation passed")
