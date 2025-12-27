
import pandas as pd

FEATURE_VERSION = "v1"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared feature logic for both training and inference
    """

    df = df.copy()

    # Feature engineering
    df["income_to_loan_ratio"] = df["annual_income"] / (df["loan_amount"] + 1)
    df["credit_utilization"] = df["current_debt"] / (df["credit_limit"] + 1)

    features = [
        "age",
        "income_to_loan_ratio",
        "credit_utilization",
        "loan_tenure"
    ]

    return df[features]
