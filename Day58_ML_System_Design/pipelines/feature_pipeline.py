import pandas as pd

def build_features(df):
    print(" Building features...")
    df["income_per_age"] = df["income"] / df["age"]

    feature_df = df.drop(columns=["approved"])
    feature_df["target"] = df["approved"]

    feature_df.to_csv("features/feature_store.csv", index=False)
    return feature_df
