import joblib
import os

def deploy_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/production.pkl")
    print(" Model deployed to production")
