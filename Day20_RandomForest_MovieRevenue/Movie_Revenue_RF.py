import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/movie_revenue.csv")

# Encode categorical column:genre
label = LabelEncoder()
df["genre_encoded"] = label.fit_transform(df["genre"])

# Features & target
X = df[["genre_encoded", "budget", "rating", "duration", "buzz", "year"]]
y = df["revenue"]


# Train/test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Random Forest model

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Results")
print("----------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")


# Feature importance

importances = rf.feature_importances_
for name, score in zip(X.columns, importances):
    print(f"{name}: {score:.3f}")


# Actual vs predicted plot

plt.scatter(y_test, y_pred, color="green")
plt.xlabel("Actual Revenue (Cr)")
plt.ylabel("Predicted Revenue (Cr)")
plt.title("Actual vs Predicted Movie Revenue")
plt.grid(True)
plt.show()
