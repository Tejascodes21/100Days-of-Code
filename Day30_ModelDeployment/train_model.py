import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("data/flight_delay.csv")

X = df.drop("delay", axis=1)
y = df["delay"]

num_cols = ["temperature", "humidity", "wind_speed", "visibility", "precipitation"]
cat_cols = ["airline", "origin"]

# Preprocessor for consistent deployment
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Best performing model from previous evaluation
model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train split
X_train, X_test, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)


joblib.dump(model, "trained_model.pkl")
print(" Flight Delay Prediction Model Saved Successfully!")
