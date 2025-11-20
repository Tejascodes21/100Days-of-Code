import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/car_prices.csv")

# Encode the brand column(categorical)
label = LabelEncoder()
df["brand_encoded"] = label.fit_transform(df["brand"])

# Features and target
X = df[["brand_encoded", "mileage", "engine", "age", "horsepower"]]
y = df["price"]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Decision tree model
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# predictions
y_pred = tree.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nDecision Tree Results")
print("-----------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")


# Feature Importance
print("\nFeature Importance:")
for feature, score in zip(X.columns, tree.feature_importances_):
    print(f"{feature}: {score:.3f}")


# Plot the Decision Tree
plt.figure(figsize=(14, 7))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree for Car Price Prediction")
plt.show()


# Actual vs predicted

plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()
