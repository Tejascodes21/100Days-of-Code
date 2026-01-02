import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "system/model.pkl")
    print(" Model trained and saved")
    return model
