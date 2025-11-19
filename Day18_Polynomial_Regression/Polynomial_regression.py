import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Data
df = pd.read_csv("data/mileage_price.csv")

X = df[["mileage"]]
y = df["price"]

# Linear Regression(baseline)
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)


# Polynomial Regression (deg 2)
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)

poly2_model = LinearRegression()
poly2_model.fit(X_poly2, y)
poly2_pred = poly2_model.predict(X_poly2)


# Polynomial Regression (deg 3)
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)

poly3_model = LinearRegression()
poly3_model.fit(X_poly3, y)
poly3_pred = poly3_model.predict(X_poly3)


# Compare models
def evaluate(name, y, pred):
    mae = mean_absolute_error(y, pred)
    rmse = (mean_squared_error(y, pred))** 0.5
    r2 = r2_score(y, pred)

    print(f"\n{name} Results")
    print("-" * 25)
    print(f"MAE :{mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.3f}")

evaluate("Linear Regression", y, linear_pred)
evaluate("Polynomial (Degree 2)",y, poly2_pred)
evaluate("Polynomial (Degree 3)", y, poly3_pred)


# Plotting
plt.scatter(X, y, color="black", label="Actual Data")

plt.plot(X, linear_pred, label="Linear Fit", color="blue")
plt.plot(X, poly2_pred,label="Poly Degree 2", color="green")
plt.plot(X, poly3_pred, label="Poly Degree 3", color="red")

plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Mileage vs Price | Polynomial Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
