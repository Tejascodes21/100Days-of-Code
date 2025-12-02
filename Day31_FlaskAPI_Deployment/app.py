from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("../Day30_ModelDeployment/trained_model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Flight Delay Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]

    result = "DELAYED" if prediction == 1 else "ON-TIME"
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
