import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset
df = pd.read_csv("data/movie_revenue.csv")

# Encode Genre (Categorical Feature)
label = LabelEncoder()
df["genre_encoded"] = label.fit_transform(df["genre"])

# Features & target
X = df[["genre_encoded", "budget", "rating", "duration", "buzz", "year"]]
y = df["revenue"]


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Gradient Boosting Model

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)


# Evaluation metrics

mae = mean_absolute_error(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nGradient Boosting Results")
print("----------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")


# Feature importance

print("\nFeature Importance:")
for name, score in zip(X.columns, gb_model.feature_importances_):
    print(f"{name} → {score:.3f}")


# Actual vs Predicted visualization

plt.scatter(y_test, y_pred, color="purple")
plt.xlabel("Actual Revenue (Cr)")
plt.ylabel("Predicted Revenue (Cr)")
plt.title("Actual vs Predicted Revenue - Gradient Boosting")
plt.grid(True)
plt.show()
