import pandas as pd

EXPECTED_COLUMNS = ["age", "income", "credit_score"]

def validate_features():
    df = pd.read_csv("data/serve.csv")

    assert list(df.columns) == EXPECTED_COLUMNS, " Feature mismatch detected"

    print(" Feature consistency check passed")
