import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset
df = pd.read_csv("data/house_rent.csv")

# Encode categorical values
label = LabelEncoder()
df["city_encoded"] = label.fit_transform(df["city"])
df["furnished_encoded"] = label.fit_transform(df["furnished"])

# Features and Target
X = df[["city_encoded", "size_sqft", "rooms", "bathroom", "furnished_encoded", "floor"]]
y = df["rent"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline Model (Before Tuning)
baseline = XGBRegressor(random_state=42)
baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)

base_mae = mean_absolute_error(y_test, base_pred)
base_rmse = (mean_squared_error(y_test, base_pred)**0.5)
base_r2 = r2_score(y_test, base_pred)

print("\nBASE MODEL PERFORMANCE")
print("----------------------------")
print(f"MAE  : {base_mae:.2f}")
print(f"RMSE : {base_rmse:.2f}")
print(f"R²   : {base_r2:.2f}\n")



# Hyperparameter Tuning using RandomizedSearchCV
params = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9]
}

model = XGBRegressor(random_state=42)

tuner = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=15,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    random_state=42
)

print("\nRunning hyperparameter search... please wait.\n")
tuner.fit(X_train, y_train)

print("\nBEST PARAMETERS FOUND:")
print(tuner.best_params_)



# Final model with best parameters

best_model = tuner.best_estimator_
final_pred = best_model.predict(X_test)

final_mae = mean_absolute_error(y_test, final_pred)
final_rmse = (mean_squared_error(y_test, final_pred)**0.5)
final_r2 = r2_score(y_test, final_pred)

print("\nTUNED MODEL PERFORMANCE")
print("----------------------------")
print(f"MAE  : {final_mae:.2f}")
print(f"RMSE : {final_rmse:.2f}")
print(f"R²   : {final_r2:.2f}")



# Visualization: Actual vs Predicted
plt.scatter(y_test, final_pred, color="green")
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Actual vs Predicted Rent (Tuned XGBoost Model)")
plt.grid(True)
plt.show()
