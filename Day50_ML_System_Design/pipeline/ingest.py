import pandas as pd

def load_data(path):
    print("Loading data...")
    df = pd.read_csv(path)
    return df
