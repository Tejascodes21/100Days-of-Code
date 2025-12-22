import joblib

def predict(sample):
    model = joblib.load("models/model_v1.pkl")
    prediction = model.predict([sample])[0]
    return prediction
