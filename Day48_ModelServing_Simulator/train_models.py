import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset (reuse Day 47 dataset)
df = pd.read_csv("../Day47_ModelVersioning/data/dataset_v1.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model v1
model_v1 = LogisticRegression(max_iter=200, C=1.0)
model_v1.fit(X_train, y_train)
joblib.dump(model_v1, "models/model_v1.pkl")

# Model v2 (slightly different hyperparameter)
model_v2 = LogisticRegression(max_iter=200, C=2.0)
model_v2.fit(X_train, y_train)
joblib.dump(model_v2, "models/model_v2.pkl")

print(" model_v1 and model_v2 saved successfully")
