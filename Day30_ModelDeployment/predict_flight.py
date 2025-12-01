import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("trained_model.pkl")

new_data = pd.DataFrame({
    "airline": ["Indigo"],
    "origin": ["DEL"],
    "temperature": [19],
    "humidity": [68],
    "wind_speed": [14],
    "visibility": [3],
    "precipitation": [4]
})


prediction = model.predict(new_data)[0]

if prediction == 1:
    print(" Prediction: Flight is likely to be DELAYED")
else:
    print("Prediction: Flight will likely be ON-TIME ")
