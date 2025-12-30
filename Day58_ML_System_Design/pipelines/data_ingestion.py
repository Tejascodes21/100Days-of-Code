import pandas as pd

def ingest_data():
    print(" Ingesting raw data...")
    df = pd.read_csv("data/raw_data.csv")
    return df
