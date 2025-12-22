import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

def train_model(X_train, X_test, y_train, y_test):
    print("Training model...")

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_v1.pkl")

    print(f" Model saved | Accuracy: {acc:.2f}")
    return acc
