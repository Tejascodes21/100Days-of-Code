import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("heart.csv")  

# Define input features exactly as used in API
features = ["age", "sex", "cp", "trestbps", "chol",
            "thalach", "oldpeak", "ca", "exang"]

X = df[features]
y = df["heart_disease"]  # Correct target column

# Create and train model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=600))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "heart_model.pkl")

print(" Model Trained & Saved Successfully!")
print(" Target Column: heart_disease")
print(" Features Used:", features)
print(" Model expects input shape:", model.n_features_in_)
