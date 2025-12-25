import pandas as pd
from ci.data_validation import validate_data
from ci.model_tests import validate_model
from train_model import train_model

print(" Running ML CI/CD Pipeline...")

# Load data
df = pd.read_csv("data/dataset.csv")

# Step 1: Data Validation
validate_data(df)

# Step 2: Train Model
model, X_val, y_val = train_model(df)

# Step 3: Model Validation (CI Gate)
accuracy = validate_model(model, X_val, y_val)

print(f" Pipeline completed successfully with accuracy: {accuracy:.2f}")
