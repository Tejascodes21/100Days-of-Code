import joblib

def load_model():
    return joblib.load("system/model.pkl")

def predict(model, X):
    return model.predict(X)
