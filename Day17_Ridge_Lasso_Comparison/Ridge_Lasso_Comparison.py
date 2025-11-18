import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load dataset
df = pd.read_csv("car_prices.csv")

# features n target
X = df[["mileage", "engine_size", "age", "horsepower"]]
y = df["price"]

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# models
lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.5)

# train models
lr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# predictions
models = {
    "Linear Regression": lr,
    "Ridge Regression": ridge,
    "Lasso Regression": lasso
}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    results[name] = [mae, rmse, r2]

# Results DataFrame
results_df = pd.DataFrame(
    results, index=["MAE", "RMSE", "R² Score"]
).T

print("\nModel Comparison:\n")
print(results_df)

# coefficient comparison
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Linear": lr.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})

print("\nCoefficient Comparison:")
print(coeff_df)

# Visualization — Coefficient Shrinkage
plt.figure(figsize=(8,5))
sns.lineplot(data=coeff_df.set_index("Feature"))
plt.title("Coefficient comparison (Linear vs Ridge vs Lasso)")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.show()

# Visualization
results_df.plot(kind="bar", figsize=(8,5))
plt.title("Model Performance comparison")
plt.ylabel("Score / Error")
plt.grid(True)
plt.show()
