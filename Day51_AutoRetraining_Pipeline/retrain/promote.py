import joblib
import os

def promote_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/champion.pkl")
    print("Challenger promoted to Champion")
