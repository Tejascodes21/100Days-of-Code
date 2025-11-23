import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Data
df = pd.read_csv("data/house_rent.csv")

# Encode categorical values
label = LabelEncoder()
df["city_encoded"] = label.fit_transform(df["city"])
df["furnished_encoded"] = label.fit_transform(df["furnished"])

# Features and Target
X = df[["city_encoded", "size_sqft", "rooms", "bathroom", "furnished_encoded", "floor"]]
y = df["rent"]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# XGBoost model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# Evaluation

mae = mean_absolute_error(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred) ** 0.5)
r2 = r2_score(y_test, y_pred)

print("\nXGBoost Performance")
print("----------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")


# Feature Importance

print("\nFeature Importance Scores:")
for col, score in zip(X.columns, model.feature_importances_):
    print(f"{col} → {score:.3f}")

plt.figure(figsize=(8,5))
plot_importance(model)
plt.title("Feature Importance - House Rent Prediction (XGBoost)")
plt.show()


# Actual vs Predicted Plot

plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Actual vs Predicted Rent - XGBoost")
plt.grid(True)
plt.show()
