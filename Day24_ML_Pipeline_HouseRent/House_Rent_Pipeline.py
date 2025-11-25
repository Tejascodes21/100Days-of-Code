import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset
df = pd.read_csv("data/house_rent.csv")

# Features
X = df[["city", "size_sqft", "rooms", "bathroom", "furnished", "floor"]]
y = df["rent"]


# Identify Categorical & Numerical columns
categorical_cols = ["city", "furnished"]
numeric_cols = ["size_sqft", "rooms", "bathroom", "floor"]


# Columntransformer Setup
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


# Pipeline: Preprocessing + model
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the pipeline
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred) ** 0.5)
r2 = r2_score(y_test, y_pred)

print("\nPipeline Model Results")
print("-----------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")


# Visualization
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Actual vs Predicted Rent (Pipeline Model)")
plt.grid(True)
plt.show()


joblib.dump(model, "rent_pipeline.pkl")
print("\nPipeline saved as rent_pipeline.pkl")
