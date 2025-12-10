# Day 38: ML Monitoring & Logging API

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
from datetime import datetime

# Load trained model
model = joblib.load("heart_model.pkl")
MODEL_VERSION = "v1.0"

# Log File setup
LOG_FILE = "logs.csv"

# Input schema
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    thalach: int
    oldpeak: float
    exang: int
    ca: int

app = FastAPI(title="Heart API with Monitoring")

@app.get("/")
def home():
    return {"message": "Monitoring Enabled! Visit /docs"}

@app.post("/predict")
def predict_heart(data: HeartData):
    input_data = np.array([[data.age, data.sex, data.cp, data.trestbps,
                            data.chol, data.thalach, data.oldpeak,
                            data.exang, data.ca]])

    # Predictions
    proba = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    result = "High Risk" if pred == 1 else "Low Risk"

    # Prepare row for logging
    record = {
        "timestamp": datetime.now(),
        "model_version": MODEL_VERSION,
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "thalach": data.thalach,
        "oldpeak": data.oldpeak,
        "exang": data.exang,
        "ca": data.ca,
        "prediction": result,
        "probability": round(float(proba), 3),
    }

    # Append to CSV log
    df = pd.DataFrame([record])
    df.to_csv(LOG_FILE, mode='a', header=not pd.io.common.file_exists(LOG_FILE), index=False)

    return {"Prediction": result, "Risk Probability": round(float(proba), 3),
            "Logged": True}

