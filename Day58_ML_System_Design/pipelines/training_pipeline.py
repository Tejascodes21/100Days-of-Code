import joblib
from sklearn.linear_model import LogisticRegression

def train_model(df):
    print(" Training model...")
    X = df.drop(columns=["target"])
    y = df["target"]

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")
    print(" Model saved")
