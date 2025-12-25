import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model(df):
    X = df.drop("default", axis=1)
    y = df["default"]

    # Stratified split (CRITICAL FIX)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # Scaling (boosts accuracy)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Balanced model (real-world fix)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model, X_val, y_val
