import joblib
import pandas as pd

def predict(input_data: dict):
    model = joblib.load("models/model.pkl")

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    return "Approved" if prediction == 1 else "Rejected"
