import joblib

def load_models():
    blue_model = joblib.load("models/model_v1.pkl")
    green_model = joblib.load("models/model_v2.pkl")
    return blue_model, green_model

def predict(model, input_data):
    return model.predict([input_data])[0]
