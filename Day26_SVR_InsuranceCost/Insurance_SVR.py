import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("Data/insurance.csv")

X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

# Columns
numeric_cols = ["age", "bmi", "children"]
categorical_cols = ["sex", "smoker", "region"]


# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Helper function for eval

def train_and_evaluate(model_name, model):
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("svr", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = (mean_squared_error(y_test, preds)) ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"\n{model_name} Performance")
    print("--------------------------------")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.2f}")

    return preds, pipeline



# 1️ Linear Kernel SVR

linear_preds, linear_model = train_and_evaluate(
    "Linear SVR",
    SVR(kernel="linear", C=1.0)
)



# 2️ RBF Kernel SVR

rbf_preds, rbf_model = train_and_evaluate(
    "RBF SVR",
    SVR(kernel="rbf", C=100, gamma=0.1)
)

# Performance Comparison plot
plt.figure(figsize=(6,4))
plt.scatter(y_test, linear_preds, color="blue", label="Linear SVR")
plt.scatter(y_test, rbf_preds, color="red", label="RBF SVR")
plt.xlabel("Actual Insurance Cost")
plt.ylabel("Predicted Insurance Cost")
plt.title("SVR Comparison: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
