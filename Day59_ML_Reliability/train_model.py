import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/train.csv")

X = df.drop(columns=["target"])
y = df["target"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model/model.pkl")
print(" Model trained and saved")
