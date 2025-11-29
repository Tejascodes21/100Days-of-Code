import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# Load dataset
df = pd.read_csv("data/flight_delay.csv")

# Features and target
X = df.drop("delay", axis=1)
y = df["delay"]

# Categorical & Numerical columns
cat_cols = ["airline", "origin"]
num_cols = ["temperature", "humidity", "wind_speed", "visibility", "precipitation"]

# Preprocessing steps
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Logistic Regression Pipeline
model = Pipeline([
    ("prep", preprocess),
    ("logreg", LogisticRegression(max_iter=300))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model.fit(X_train, y_train)

# Predictions & probabilities
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:,1]

# Evaluation metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — Flight Delay Prediction")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression (Flight Delay)")
plt.legend()
plt.grid(True)
plt.show()
