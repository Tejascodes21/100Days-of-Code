import pandas as pd

def validate_data():
    df = pd.read_csv("data/train.csv")

    assert df.isnull().sum().sum() == 0, " Missing values found"
    assert (df["age"] > 0).all(), " Invalid age values"

    print(" Data validation passed")
