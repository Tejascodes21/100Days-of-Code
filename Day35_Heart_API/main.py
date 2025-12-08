# Day 35: Heart Disease Prediction API using FastAPI


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib


model = joblib.load("heart_model.pkl")

# Input schema for API
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

# FastAPI App
app = FastAPI(
    title="Heart Disease Prediction API ",
    description="API to predict heart attack risk using ML model",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "API is running! Go to /docs for Swagger UI"}

@app.post("/predict")
def predict_heart_disease(data: HeartData):
    input_data = np.array([[data.age, data.sex, data.cp, data.trestbps,
                            data.chol, data.thalach, data.oldpeak,
                            data.exang, data.ca]])

    prediction_proba = model.predict_proba(input_data)[0][1]  # risk probability
    prediction = model.predict(input_data)[0]

    result = {
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "risk_probability": round(float(prediction_proba), 3)
    }

    return result
    