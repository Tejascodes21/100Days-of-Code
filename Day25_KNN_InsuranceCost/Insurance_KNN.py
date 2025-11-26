import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset
df = pd.read_csv("data/insurance.csv")

X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

# Columns by type
numeric_cols = ["age", "bmi", "children"]
categorical_cols = ["sex", "smoker", "region"]


# Preprocessing Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Try multiple K values (small dataset → limit neighbors)
k_values = [1, 3, 5, 7]
results = []

# Train-test split with more training samples
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


# Experiment with Different K Values

for k in k_values:

   
    if k > len(X_train):
        continue

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append((k, mae, rmse, r2))


# Print Performance Table

print("\nKNN Performance Comparison:")
print("K  |  MAE  |  RMSE  |  R²")
print("----------------------------")
for k, mae, rmse, r2 in results:
    print(f"{k:<2}| {mae:.2f} | {rmse:.2f} | {r2:.2f}")

# Best K based on MAE
best_k = sorted(results, key=lambda x: x[1])[0][0]
print(f"\nBest K found: {best_k}")

# Final model
final_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("knn", KNeighborsRegressor(n_neighbors=best_k))
])

final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)


# Visualization: Elbow curve

plt.figure(figsize=(6,4))
plt.plot([r[0] for r in results], [r[1] for r in results], marker="o")
plt.xlabel("K Value")
plt.ylabel("MAE")
plt.title("KNN - K vs MAE (Elbow Method)")
plt.grid(True)
plt.show()

# Actual vs Predicted Plot
plt.scatter(y_test, final_pred, color="green")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title(f"Actual vs Predicted Medical Costs (K={best_k})")
plt.grid(True)
plt.show()
