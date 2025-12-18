import pandas as pd
import joblib
import uuid
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


df = pd.read_csv("data/train_data.csv")
X = df.drop("default", axis=1)
y = df["default"]

RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

params = {
    "model": "LogisticRegression",
    "C": 1.0,
    "solver": "lbfgs"
}

model = LogisticRegression(
    C=params["C"],
    solver=params["solver"],
    max_iter=1000,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

run_id = str(uuid.uuid4())[:8]
timestamp = datetime.now()

os.makedirs("models", exist_ok=True)
os.makedirs("experiments", exist_ok=True)


model_path = f"models/model_{run_id}.pkl"
joblib.dump(model, model_path)


run_log = {
    "run_id": run_id,
    "timestamp": timestamp,
    "model": params["model"],
    "C": params["C"],
    "solver": params["solver"],
    "accuracy": round(accuracy, 3),
    "f1_score": round(f1, 3),
    "model_path": model_path
}

runs_file = "experiments/runs.csv"

if os.path.exists(runs_file):
    existing = pd.read_csv(runs_file)
    updated = pd.concat([existing, pd.DataFrame([run_log])], ignore_index=True)
else:
    updated = pd.DataFrame([run_log])

updated.to_csv(runs_file, index=False)

print("Experiment logged successfully")
print(run_log)
