import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import os


df = pd.read_csv("data/dataset_v1.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))


version = "v2"
os.makedirs("models", exist_ok=True)
model_path = f"models/model_{version}.pkl"
joblib.dump(model, model_path)

print(f"Model {version} trained | Accuracy: {accuracy:.3f}")
