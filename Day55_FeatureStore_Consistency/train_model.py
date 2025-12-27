
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from feature_store import build_features, FEATURE_VERSION


df = pd.read_csv("data/train_data.csv")

X = build_features(df)
y = df["default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)


joblib.dump(
    {
        "model": model,
        "feature_version": FEATURE_VERSION
    },
    "model.pkl"
)

print(" Model trained successfully")
print(f" Feature version used: {FEATURE_VERSION}")
