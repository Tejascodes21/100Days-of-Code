from sklearn.model_selection import train_test_split

def preprocess(df):
    print("Preprocessing data...")

    X = df.drop("default", axis=1)
    y = df["default"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
