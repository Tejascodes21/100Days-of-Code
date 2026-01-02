def validate_data(df):
    required_columns = ["age", "income", "loan_amount"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Null values detected")

    print(" Data validation passed")
